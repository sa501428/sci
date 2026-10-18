#include "core.h"
#include "straw.h"
#include "straw_v10.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>

namespace {
using Clock = std::chrono::steady_clock;
double seconds(Clock::time_point start) { return std::chrono::duration<double>(Clock::now() - start).count(); }
struct Arguments {
    sci::Options model;
    std::string file, prefix, norm = "GW_KR", order = "1", genome, chromosomes, graph, embeddings;
    int resolution = 0;
};
void help() {
    std::cout << R"(SCI C++: trans Hi-C graph -> LINE embedding -> k-means
Usage: sci -f input.hic -n output_prefix -r resolution [options]

  -f, --infile PATH        .hic file or HTTP(S) URL; no conversion needed
  -n, --name PREFIX        Output prefix (including an optional directory)
  -r, --resolution BP      Stored BP resolution (or V10 derived resolution)
      --normalization STR Normalization, default GW_KR; use NONE for raw counts
  -g, --genome_size PATH   Optional chromosome sizes, otherwise use .hic header
      --chromosomes LIST  Comma-separated subset; default all except All
  -o, --order 1|2|both     LINE order, default 1; both concatenates and L2 normalizes
  -s, --samples MILLIONS  Samples per order, default 25; fractional values allowed
      --sample-count N    Exact sample count per order (alternative to -s)
      --dimensions N      Dimensions per order, default 100
      --negative N        Negatives per positive sample, default 5
      --learning-rate X   Initial SGD learning rate, default 0.025
  -k, --clusters N        Number of clusters, default 5
      --restarts N        K-means++ restarts, default 10
      --max-iterations N  K-means iteration limit, default 300
  -t, --threads N         Training/clustering threads, default 1 (reproducible)
      --seed N            Random seed, default 12345
      --dump-graph PATH   Optional undirected graph: genome-bin IDs and log weight
      --embeddings PATH   Optional text embedding: genome-bin ID then features
  -h, --help              Show help

Writes PREFIX_compartments.txt (legacy table), PREFIX_compartments.bed,
and PREFIX_summary.json. Bins without positive log-weight trans contacts are NA.
BED intervals are clipped to chromosome ends. Cluster numbers are arbitrary.
Multi-thread SGD uses row locks; its update order is not reproducible.
)";
}
uint64_t integer(const std::string& text, const std::string& key) {
    if (text.empty() || text.find_first_not_of("0123456789") != std::string::npos)
        throw std::invalid_argument(key + " requires a nonnegative integer");
    size_t used = 0;
    auto result = std::stoull(text, &used);
    if (used != text.size()) throw std::invalid_argument("invalid " + key);
    return result;
}
int bounded(const std::string& text, const std::string& key, int minimum, int maximum) {
    uint64_t value = integer(text, key);
    if (value < uint64_t(minimum) || value > uint64_t(maximum))
        throw std::invalid_argument(key + " must be between " + std::to_string(minimum) + " and " + std::to_string(maximum));
    return int(value);
}
double real(const std::string& text, const std::string& key) {
    size_t used = 0; double value = std::stod(text, &used);
    if (used != text.size() || !std::isfinite(value) || value <= 0)
        throw std::invalid_argument(key + " requires a finite positive number");
    return value;
}
Arguments parse(int argc, char** argv) {
    Arguments a;
    bool samplesSet = false;
    for (int i = 1; i < argc; ++i) {
        std::string key = argv[i];
        if (key == "-h" || key == "--help") { help(); std::exit(0); }
        if (i + 1 == argc) throw std::invalid_argument("missing value for " + key);
        std::string value = argv[++i];
        if (key == "-f" || key == "--infile") a.file = value;
        else if (key == "-n" || key == "--name") a.prefix = value;
        else if (key == "-r" || key == "--resolution") a.resolution = bounded(value, key, 1, INT32_MAX);
        else if (key == "--normalization") a.norm = value;
        else if (key == "-g" || key == "--genome_size" || key == "--genome-size") a.genome = value;
        else if (key == "--chromosomes") a.chromosomes = value;
        else if (key == "-o" || key == "--order") a.order = value;
        else if (key == "-s" || key == "--samples" || key == "--sample-count") {
            if (samplesSet) throw std::invalid_argument("specify only one of --samples and --sample-count");
            samplesSet = true;
            if (key == "--sample-count") a.model.samples = integer(value, key);
            else {
                double count = real(value, key) * 1000000;
                if (count < 1 || count > 1e15) throw std::invalid_argument("sample count must be in [1, 10^15]");
                a.model.samples = uint64_t(std::round(count));
            }
            if (!a.model.samples || a.model.samples > 1000000000000000ULL)
                throw std::invalid_argument("sample count must be in [1, 10^15]");
        }
        else if (key == "--dimensions") a.model.dimensions = bounded(value, key, 1, 65536);
        else if (key == "--negative") a.model.negatives = bounded(value, key, 0, 4096);
        else if (key == "--learning-rate") {
            a.model.learningRate = float(real(value, key));
            if (!(a.model.learningRate > 0) || !std::isfinite(a.model.learningRate)) throw std::invalid_argument("learning rate out of float range");
        }
        else if (key == "-k" || key == "--clusters") a.model.clusters = bounded(value, key, 1, INT32_MAX);
        else if (key == "--restarts") a.model.restarts = bounded(value, key, 1, 10000);
        else if (key == "--max-iterations") a.model.iterations = bounded(value, key, 1, 1000000);
        else if (key == "-t" || key == "--threads") a.model.threads = bounded(value, key, 1, 1024);
        else if (key == "--seed") a.model.seed = integer(value, key);
        else if (key == "--dump-graph") a.graph = value;
        else if (key == "--embeddings") a.embeddings = value;
        else throw std::invalid_argument("unknown option " + key);
    }
    if (a.file.empty() || a.prefix.empty() || !a.resolution) throw std::invalid_argument("-f, -n and -r are required; use --help");
    if (a.order != "1" && a.order != "2" && a.order != "both") throw std::invalid_argument("--order must be 1, 2 or both");
    if (a.norm.empty()) throw std::invalid_argument("empty normalization");
    // Protect inputs and prevent optional exports from colliding with outputs.
    std::set<std::filesystem::path> paths;
    for (auto& p : std::vector<std::string>{a.file, a.genome, a.prefix + "_compartments.txt",
             a.prefix + "_compartments.bed", a.prefix + "_summary.json", a.graph, a.embeddings}) {
        if (p.empty() || p.find("://") != std::string::npos) continue;
        if (!paths.insert(std::filesystem::weakly_canonical(p)).second)
            throw std::invalid_argument("input and output paths must be distinct: " + p);
    }
    return a;
}
std::string canonical(std::string s) {
    if (s.size() > 3 && s.substr(0, 3) == "chr") s.erase(0, 3);
    return s;
}
struct Chromosome {
    std::string source, name;
    int64_t length;
    uint32_t offset, bins;
    std::vector<double> norm;
};
std::vector<Chromosome> chromosomes(const StrawFileInfo& info, const Arguments& a) {
    std::map<std::string, chromosome> available;
    for (auto& c : info.chromosomes) {
        std::string name = canonical(c.name);
        if (name == "All" || name == "ALL" || name == "all") continue;
        if (!available.emplace(name, c).second) throw std::runtime_error("ambiguous chromosome aliases in .hic header");
    }
    std::vector<std::pair<std::string, int64_t>> requested;
    if (!a.genome.empty()) {
        std::ifstream in(a.genome);
        if (!in) throw std::runtime_error("cannot open chromosome sizes: " + a.genome);
        std::string line;
        while (std::getline(in, line)) {
            if (line.empty() || line[0] == '#') continue;
            std::istringstream row(line); std::string name, size, extra;
            if (!(row >> name >> size) || row >> extra) throw std::runtime_error("invalid chromosome sizes row: " + line);
            uint64_t length = integer(size, "chromosome size");
            if (!length || length > INT64_MAX) throw std::runtime_error("invalid chromosome length");
            requested.emplace_back(name, int64_t(length));
        }
    } else for (auto& c : info.chromosomes) {
        if (available.count(canonical(c.name))) requested.emplace_back(c.name, c.length);
    }
    std::set<std::string> selected;
    if (!a.chromosomes.empty()) {
        std::istringstream list(a.chromosomes); std::string name;
        while (std::getline(list, name, ',')) {
            if (!available.count(canonical(name))) throw std::runtime_error("chromosome absent from .hic: " + name);
            if (!selected.insert(canonical(name)).second) throw std::runtime_error("duplicate chromosome: " + name);
        }
        if (selected.empty() || a.chromosomes.back() == ',') throw std::runtime_error("invalid chromosome list");
    }
    std::vector<Chromosome> result;
    std::set<std::string> seen;
    uint64_t offset = 0;
    for (auto& entry : requested) {
        const std::string key = canonical(entry.first);
        if (!selected.empty() && !selected.count(key)) continue;
        auto found = available.find(key);
        if (found == available.end()) throw std::runtime_error("chromosome absent from .hic: " + entry.first);
        if (!seen.insert(key).second) throw std::runtime_error("duplicate chromosome: " + entry.first);
        if (entry.second != found->second.length || entry.second <= 0)
            throw std::runtime_error("chromosome length differs from .hic: " + entry.first);
        uint64_t bins = uint64_t(entry.second) / a.resolution + (entry.second % a.resolution != 0);
        if (offset + bins >= UINT32_MAX) throw std::runtime_error("too many bins for 32-bit node IDs; increase resolution");
        result.push_back({found->second.name, entry.first, entry.second, uint32_t(offset), uint32_t(bins), {}});
        offset += bins;
    }
    for (auto& name : selected) if (!seen.count(name)) throw std::runtime_error("selected chromosome absent from sizes file: " + name);
    if (result.size() < 2) throw std::runtime_error("SCI needs at least two chromosomes for a trans graph");
    return result;
}
std::ofstream output(const std::string& path) {
    auto parent = std::filesystem::path(path).parent_path();
    if (!parent.empty()) std::filesystem::create_directories(parent);
    std::ofstream out;
    out.exceptions(std::ios::failbit | std::ios::badbit);
    out.open(path);
    return out;
}
void run(const Arguments& a) {
    const auto start = Clock::now();
    auto info = getFileInfo(a.file);
    if (std::find(info.bpResolutions.begin(), info.bpResolutions.end(), a.resolution) == info.bpResolutions.end())
        throw std::runtime_error("requested BP resolution is absent from .hic");
    if (a.norm != "NONE" && std::find(info.normalizations.begin(), info.normalizations.end(), a.norm) == info.normalizations.end())
        throw std::runtime_error("normalization " + a.norm + " is absent from .hic; choose --normalization explicitly");
    auto chroms = chromosomes(info, a);
    const size_t totalBins = size_t(chroms.back().offset) + chroms.back().bins;
    std::unique_ptr<StrawRawReader> legacy;
    std::unique_ptr<straw_v10::File> v10;
    if (info.version == 10) v10.reset(new straw_v10::File(a.file));
    else legacy.reset(new StrawRawReader(a.file));
    std::unique_ptr<StrawNormalizationVectors> storedNorms;
    if (info.version == 9 && a.norm != "NONE") storedNorms.reset(new StrawNormalizationVectors(a.file));
    for (auto& c : chroms) if (a.norm != "NONE") {
        if (v10) c.norm = v10->normalization(c.source, "BP", a.resolution, a.norm);
        else if (storedNorms) {
            const auto& entries = storedNorms->entries();
            for (size_t i = 0; i < entries.size(); ++i) {
                const auto& e = entries[i];
                if (e.chromosome != c.source || e.normalization != a.norm || e.unit != "BP" || e.resolution != a.resolution) continue;
                c.norm.reserve(size_t(e.count));
                storedNorms->stream(i, [&](uint32_t bits) {
                    float value; std::memcpy(&value, &bits, sizeof(value)); c.norm.push_back(value);
                });
                break;
            }
        }
        else if (!getNormalizationVectorForFile(a.file, c.source, a.resolution, a.norm, c.norm))
            throw std::runtime_error("normalization unavailable for " + c.source);
        if (c.norm.size() < c.bins) throw std::runtime_error("normalization vector too short for " + c.source);
    }
    storedNorms.reset();
    sci::Graph graph;
    std::vector<double> degrees(totalBins, 0);
    uint64_t records = 0, skipped = 0, missingPairs = 0;
    std::ofstream graphOut;
    if (!a.graph.empty()) { graphOut = output(a.graph); graphOut << std::setprecision(9); }
    for (size_t i = 0; i < chroms.size(); ++i) for (size_t j = i + 1; j < chroms.size(); ++j) {
        auto& x = chroms[i]; auto& y = chroms[j];
        auto consume = [&](int64_t bx, int64_t by, double count) {
            ++records;
            if (bx < 0 || by < 0 || uint64_t(bx) >= x.bins || uint64_t(by) >= y.bins)
                throw std::runtime_error("contact bin outside chromosome bounds");
            if (a.norm != "NONE") {
                double nx = x.norm[size_t(bx)], ny = y.norm[size_t(by)];
                if (!(nx > 0) || !(ny > 0) || !std::isfinite(nx) || !std::isfinite(ny)) { ++skipped; return; }
                count /= nx * ny;
            }
            // SCI writes log(count) for count >= 1. Count == 1 has zero
            // sampling probability; remove it rather than embedding random nodes.
            if (!(count > 1) || !std::isfinite(count)) { ++skipped; return; }
            float weight = float(std::log(count));
            const uint32_t u = x.offset + uint32_t(bx), v = y.offset + uint32_t(by);
            graph.add(u, v, weight); degrees[u] += weight; degrees[v] += weight;
            if (graphOut.is_open()) graphOut << u << '\t' << v << '\t' << weight << '\n';
        };
        bool found;
        if (v10) {
            found = v10->hasMatrix(x.source, y.source, "BP", a.resolution);
            if (found) v10->raw(x.source, y.source, "BP", a.resolution, 0, x.bins, 0, y.bins,
                [&](const straw_v10::Record& r) { consume(r.binX, r.binY, r.isScore ? double(r.score) : double(r.count)); });
        } else found = legacy->forEachBlock(x.source, y.source, a.resolution,
            [&](const std::vector<contactRecord>& block) {
                for (auto& r : block) consume(r.binX, r.binY, r.counts);
            });
        if (!found) ++missingPairs;
    }
    if (graphOut.is_open()) graphOut.close();
    legacy.reset(); v10.reset();
    for (auto& c : chroms) std::vector<double>().swap(c.norm);
    if (!graph.size()) throw std::runtime_error("no positive-weight trans edges (normalized counts must exceed 1)");
    std::vector<uint32_t> mapping(totalBins, UINT32_MAX);
    size_t nodes = 0; double weightSum = 0;
    for (size_t i = 0; i < totalBins; ++i) if (degrees[i] > 0) {
        mapping[i] = uint32_t(nodes++); weightSum += degrees[i];
    }
    if (size_t(a.model.clusters) > nodes) throw std::runtime_error("more clusters than active bins");
    std::vector<double> negativeWeights; negativeWeights.reserve(nodes);
    for (double degree : degrees) if (degree > 0) negativeWeights.push_back(std::pow(degree, 0.75));
    std::vector<double>().swap(degrees);
    for (uint32_t i = 0; i < graph.size(); ++i) {
        auto& e = graph[i]; e.u = mapping[e.u]; e.v = mapping[e.v];
    }
    sci::Alias negative(negativeWeights);
    std::vector<double>().swap(negativeWeights);
    graph.prepare();
    double loadSeconds = seconds(start);
    std::cerr << "Loaded " << graph.size() << " undirected edges, " << nodes << '/' << totalBins
              << " active bins in " << loadSeconds << " s; graph allocation " << graph.allocatedBytes() / 1048576.0 << " MiB\n";
    if (missingPairs) std::cerr << "Skipped " << missingPairs << " absent trans matrices\n";
    const size_t dim = size_t(a.model.dimensions) * (a.order == "both" ? 2 : 1);
    if (nodes > std::numeric_limits<size_t>::max() / dim) throw std::runtime_error("embedding allocation overflows");
    std::vector<float> features(nodes * dim);
    const auto training = Clock::now();
    if (a.order == "both") {
        sci::trainInto(graph, negative, nodes, a.model, 1, features, dim, 0);
        sci::trainInto(graph, negative, nodes, a.model, 2, features, dim, size_t(a.model.dimensions));
        sci::normalize(features, nodes, dim);
    } else sci::trainInto(graph, negative, nodes, a.model, a.order == "1" ? 1 : 2, features, dim, 0);
    double trainSeconds = seconds(training);
    std::cerr << "LINE training: " << trainSeconds << " s\n";
    const uint32_t edges = graph.size(); const size_t graphBytes = graph.allocatedBytes();
    graph = sci::Graph(); negative.probability.clear(); negative.probability.shrink_to_fit();
    negative.alias.clear(); negative.alias.shrink_to_fit();
    if (!a.embeddings.empty()) {
        auto out = output(a.embeddings);
        out << nodes << ' ' << dim << '\n' << std::setprecision(9);
        for (size_t i = 0; i < totalBins; ++i) if (mapping[i] != UINT32_MAX) {
            out << i;
            for (size_t d = 0; d < dim; ++d) out << ' ' << features[size_t(mapping[i]) * dim + d];
            out << '\n';
        }
        out.close();
    }
    const auto clustering = Clock::now();
    auto result = sci::kmeans(features, nodes, dim, a.model);
    double clusterSeconds = seconds(clustering);
    std::cerr << "K-means: " << clusterSeconds << " s; inertia " << result.inertia << '\n';
    auto table = output(a.prefix + "_compartments.txt"), bed = output(a.prefix + "_compartments.bed");
    table << "chr\tstart\tend\tcompartment\n";
    for (auto& c : chroms) for (uint32_t b = 0; b < c.bins; ++b) {
        int64_t begin = int64_t(b) * a.resolution, end = std::min(begin + a.resolution, c.length);
        uint32_t node = mapping[c.offset + b];
        std::string label = node == UINT32_MAX ? "NA" : std::to_string(result.labels[node]);
        table << c.name << '\t' << begin << '\t' << end << '\t' << label << '\n';
        bed << c.name << '\t' << begin << '\t' << end << '\t' << label << '\n';
    }
    table.close(); bed.close();
    auto summary = output(a.prefix + "_summary.json");
    summary << std::setprecision(12) << "{\n  \"hic_version\": " << info.version
            << ",\n  \"resolution\": " << a.resolution << ",\n  \"normalization\": ";
    // std::quoted escapes quotes and backslashes for normalization identifiers.
    for (unsigned char ch : a.norm) if (ch < 32) throw std::runtime_error("invalid normalization identifier");
    summary << std::quoted(a.norm) << ",\n  \"order\": " << std::quoted(a.order)
            << ",\n  \"seed\": " << a.model.seed << ",\n  \"threads\": " << a.model.threads
            << ",\n  \"samples_per_order\": " << a.model.samples << ",\n  \"dimensions\": " << dim
            << ",\n  \"negative_samples\": " << a.model.negatives << ",\n  \"learning_rate\": " << a.model.learningRate
            << ",\n  \"clusters\": " << a.model.clusters << ",\n  \"kmeans_restarts\": " << a.model.restarts
            << ",\n  \"kmeans_max_iterations\": " << a.model.iterations
            << ",\n  \"chromosomes\": " << chroms.size() << ",\n  \"total_bins\": " << totalBins
            << ",\n  \"active_bins\": " << nodes << ",\n  \"undirected_edges\": " << edges
            << ",\n  \"trans_records\": " << records << ",\n  \"filtered_records\": " << skipped
            << ",\n  \"missing_pairs\": " << missingPairs << ",\n  \"log_weight_sum\": " << weightSum / 2
            << ",\n  \"graph_allocated_bytes\": " << graphBytes
            << ",\n  \"embedding_bytes\": " << features.size() * sizeof(float)
            << ",\n  \"load_seconds\": " << loadSeconds << ",\n  \"train_seconds\": " << trainSeconds
            << ",\n  \"cluster_seconds\": " << clusterSeconds << ",\n  \"total_seconds\": " << seconds(start)
            << ",\n  \"inertia\": " << result.inertia << ",\n  \"best_kmeans_iterations\": " << result.iterations << "\n}\n";
    summary.close();
    std::cerr << "Wrote " << a.prefix << "_compartments.bed and run summary\n";
}
} // namespace
int main(int argc, char** argv) {
    try { run(parse(argc, argv)); return 0; }
    catch (const std::exception& e) { std::cerr << "sci: " << e.what() << '\n'; return 1; }
}
