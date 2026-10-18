# SCI C++

Native C++17 port of Sub-compartment Identifier (SCI), using Straw C++ to read
`.hic` files directly. The pipeline is **log-weighted trans contacts → LINE
graph embedding → k-means++ clustering → compartment BED**. First-order,
second-order, and concatenated first/second-order embeddings are supported.
The Python and bundled LINE sources remain as references; their original
instructions are in [README.legacy.md](README.legacy.md).

Based on SCI by Haitham Ashoor and Sheng Li, and LINE by Jian Tang et al.

There is no Python, NumPy, scikit-learn, GSL, Java, intermediate BEDPE, or graph
text dependency at runtime. `.cool` input is not supported.

## Build

Requires CMake 3.16+, a C++17 compiler, libcurl, zlib, zstd, and the local Straw
source tree with `StrawRawReader`, `StrawNormalizationVectors`, and the v10 API.
The default Straw path matches this workspace's sibling `hic-infra` checkout.
Only its library sources are compiled; Straw's executables and tests are not
added to this build, and the Straw source tree is not modified.

```sh
cd /path/to/subcompartments/sci
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DSCI_NATIVE=ON
cmake --build build -j
ctest --test-dir build --output-on-failure
```

For another checkout location, add
`-DSTRAW_SOURCE_DIR=/path/to/straw/C++` to the configure command.
`SCI_NATIVE=ON` enables instructions for the build machine. Omit it for a binary
intended for other CPUs. Release builds enable link-time optimization when the
compiler supports it. `-DBUILD_TESTING=OFF` removes the Python requirement for
test fixture generation. Install with `cmake --install build --prefix PREFIX`.

## Run

```sh
./build/sci -f /data/GM12878.hic -n results/GM12878 -r 100000 \
  --normalization GW_KR -g chromosome_sizes/hg19.chrom.sizes \
  -o both -s 25 -k 5
```

`GW_KR` is the default normalization, matching the original `hic2sci.sh`
converter. Choose the appropriate vector actually stored in your file, such as
`KR`, `SCALE`, `INTER_SCALE`, or `NONE`. There is no silent fallback to raw
counts. Normalization is raw count divided by the two bins' normalization
values; nonpositive/nonfinite normalization entries mask those bins. SCI keeps
only contacts whose **normalized value exceeds 1**, then weights them by its
natural logarithm. Switching normalization can substantially change the graph.

The optional `-g` restricts chromosomes and determines output names/order.
Chromosome names with and without `chr` are matched. Sizes must match the `.hic`
header. Without `-g`, every real chromosome in the header is used; the synthetic
`All` chromosome is always excluded. `--chromosomes 1,2,3` further restricts
the input. To reproduce the original human autosome graph, use the supplied
autosome sizes file. Input bin counts use ceiling division, and the last BED
interval ends at the chromosome length.

`.hic` versions supported by Straw are accepted. Versions 9 and 10 have
dedicated regression coverage, including v10 derived resolutions and v9
trans-only files with normalization vectors but no cis matrices. Older versions
use Straw's legacy normalization helper, which requires a cis matrix when
normalization is requested. Resolution must be advertised by the `.hic` header;
the port does not silently rebin a legacy file. Local files and Straw HTTP(S)
inputs use the same pipeline.

### Outputs

| File | Contents |
| --- | --- |
| `PREFIX_compartments.txt` | Legacy four-column table with a header |
| `PREFIX_compartments.bed` | Headerless chromosome/start/end/cluster BED |
| `PREFIX_summary.json` | Parameters, graph counts, memory sizes, stage times, inertia |

Bins with no retained trans edges are `NA`. Labels are integers starting at zero;
they are arbitrary cluster IDs, not biological A1/A2/B1/B2 names. Missing trans
matrices are skipped and counted in the summary. A graph without positive
weights is an error. Native stored matrix records are expected to be unique,
as in a standard `.hic` matrix; nonstandard duplicate records must first be
canonicalized. Straw aggregates v10 derived resolutions before the logarithm.

Optional `--dump-graph PATH` writes **one undirected edge** per line, using
genome-wide bin IDs before inactive bins are removed, and `log(count)` weights.
Optional `--embeddings PATH` writes a LINE-style text embedding with the same
genome-wide bin IDs. Both exports are off by default to avoid large text I/O.

### Parameters

| Flag | Default | Meaning |
| --- | --- | --- |
| `-o, --order` | `1` | `1`, `2`, or `both` |
| `-s, --samples` | `25` | Millions of positive edge samples **per order**; fractions accepted |
| `--sample-count` | — | Exact sample count, alternative to `-s` |
| `--dimensions` | `100` | Embedding dimensions per order |
| `--negative` | `5` | Negative samples per positive edge |
| `--learning-rate` | `0.025` | Initial learning rate, decaying linearly with a 0.0001 floor multiplier |
| `-k, --clusters` | `5` | K-means clusters |
| `--restarts` | `10` | Independent k-means++ starts, keeping the lowest inertia |
| `--max-iterations` | `300` | Maximum clustering iterations per start |
| `-t, --threads` | `1` | Training and clustering workers |
| `--seed` | `12345` | Random seed |

Single-thread runs are reproducible for a fixed build/input/seed. Parallel SGD
uses per-worker RNGs and ordered stripe locks around all touched rows, avoiding
undefined shared-memory races. Its update order is nondeterministic; additional
threads may lose performance to row contention, especially on small graphs.
One thread is the default deliberately. Increase threads after measuring your
workload. Compiler SIMD accelerates dimension loops without unsafe fast-math.

## Performance design

* **No dense contact matrices.** Only trans records are decoded, block by block;
  no cis reads or quadratic scans over missing contacts.
* **Persistent reader per file.** Pair sweeps reuse the open Straw reader.
  Normalization vectors are loaded once per selected chromosome and released
  after graph construction. V9 reads the vector index independently of cis data.
* **16 bytes per undirected edge.** Endpoints, alias index, and float sampling
  probability share a compact record. Weights are overwritten in place during
  alias construction. Random direction is equivalent to the original two
  directed edges. Fixed-size chunks avoid growing-vector copies.
* **Compact negative sampler.** An alias distribution over `degree^0.75`
  replaces LINE's fixed 100-million-integer negative table. The fixed
  30-million-integer vertex hash is also removed: no unconditional 520 MB
  allocation. Sampling takes O(1) expected time.
* **In-process float embeddings.** No intermediate serialization or process
  launches. First order allocates no context array. `both` trains directly into
  the final embedding's two halves and then normalizes concatenated rows.
* **Bounded k-means distances.** Hamerly bounds skip settled points. K-means++
  initialization and assignment avoid a bins-by-clusters distance matrix.
  The graph and negative sampler are freed before clustering.

For E retained undirected edges, N active bins, D dimensions per order, B total
bins, K clusters, and T clustering threads, principal allocations are:

| Phase | Storage (bytes, excluding Straw buffers/allocator overhead) |
| --- | --- |
| Graph ingestion | `16E + 8B + normalization vectors` |
| Negative alias construction | `16E + 4B + 20N` |
| Alias construction | `20E + 4B + 8N` |
| First-order training | `16E + 4ND + 8N + 4B` |
| Second-order training | `16E + 8ND + 8N + 4B` |
| Both-order training peak | `16E + 12ND + 8N + 4B` |
| K-means | `4ND' + O(N + TKD')`, where D' is D or 2D |

The graph rounds up to 65,536-edge chunks (less than 1 MiB slack). Alias
construction has one transient 4-byte-per-edge partition stack. Memory still
grows with retained contacts; this is an in-memory graph algorithm. Original
LINE alone kept about **64E bytes** of directed edge/sampler arrays, plus the
520 MB fixed tables and preprocessing matrices.

### Algorithm compatibility

The model remains LINE with weighted positive sampling, `degree^0.75` negative
sampling, SGD, and k-means++. `both` concatenates independently trained
first/second-order embeddings and L2-normalizes the result, as Python SCI does.
Single-order embeddings are not normalized.

Results are not bit-for-bit identical: RNGs, sampling-table precision, update
scheduling, and k-means initialization differ. Original LINE's sigmoid table
used integer division and accessed an uninitialized boundary entry; this port
uses a proper bounded float lookup. Zero-log-weight-only bins are now `NA`,
rather than receiving clusters from untrained random vectors. The extra empty
bin for exactly divisible lengths and overshooting last intervals are corrected.
Float weights retain more precision than the old six-decimal graph text.
Biological agreement on real data should be assessed separately from
graph-weight parity; cluster numbers themselves are arbitrary.

## Validation and benchmark

```sh
ctest --test-dir build --output-on-failure
python3 tests/benchmark.py build/sci --bins 512 --samples 25 --keep /tmp/sci-benchmark
```

Independent byte-level v9/v10 fixtures are checked against a dense Python graph
reference. Coverage includes normalization masking, all embedding modes,
direct versus concatenated embeddings, planted-community recovery, chromosome
aliases/transposition, clipped intervals, NA bins, missing matrices, v10
aggregation, deterministic seeds, threading, sampling distributions, SGD,
k-means, and invalid inputs. Only fixture generation uses Python (standard
library and native libzstd); scientific Python packages are unnecessary.

Measured in a Release/native arm64 build on this machine: 3 chromosomes × 512
bins, 786,432 undirected edges, first-order D=100, 5 negatives, 25 million
samples, K=5, 10 restarts, one thread:

| Measurement | Result |
| --- | --- |
| Wall time, including process startup and output | 5.63 s |
| Graph loading and alias construction | 0.023 s |
| LINE training | 5.58 s |
| K-means | 0.0053 s |
| Peak resident memory | 23.8 MiB |
| Graph allocation | 12 MiB |

Fixture generation is excluded. These are synthetic native measurements, not a
measured speedup over Python SCI or a whole-human-genome performance claim. The
original Python 2/GSL executable was not runnable in this environment. Re-run
the benchmark on your machine; summaries report actual stage timings per run.
