#pragma once
#include <cstdint>
#include <memory>
#include <vector>

namespace sci {
// SplitMix64: per-worker streams, no global RNG state or GSL dependency.
struct Random {
    uint64_t state;
    uint64_t next();
    double uniform() { return (next() >> 11) * 0x1.0p-53; }
    uint32_t index(uint32_t n);
};
struct Edge { uint32_t u, v; float probability; uint32_t alias; };
static_assert(sizeof(Edge) == 16, "graph memory accounting");
// Fixed chunks avoid vector reallocation/copy peaks on large contact graphs.
class Graph {
    static constexpr unsigned shift = 16;
    static constexpr size_t chunk = size_t(1) << shift;
    std::vector<std::unique_ptr<Edge[]>> blocks;
    uint32_t count = 0;
public:
    void add(uint32_t u, uint32_t v, float weight);
    uint32_t size() const { return count; }
    size_t allocatedBytes() const { return blocks.size() * chunk * sizeof(Edge); }
    Edge& operator[](uint32_t i) { return blocks[i >> shift][i & (chunk - 1)]; }
    const Edge& operator[](uint32_t i) const { return blocks[i >> shift][i & (chunk - 1)]; }
    void prepare(); // replaces weights with alias probabilities
    const Edge& sample(Random& rng) const;
};
struct Alias {
    std::vector<float> probability;
    std::vector<uint32_t> alias;
    explicit Alias(const std::vector<double>& weights);
    uint32_t sample(Random& rng) const;
};
struct Options {
    int dimensions = 100, negatives = 5, threads = 1;
    uint64_t samples = 25000000, seed = 12345;
    float learningRate = 0.025f;
    int clusters = 5, restarts = 10, iterations = 300;
};
std::vector<float> train(const Graph&, const Alias&, size_t nodes, const Options&, int order);
void trainInto(const Graph&, const Alias&, size_t nodes, const Options&, int order,
               std::vector<float>& destination, size_t stride, size_t offset);
void normalize(std::vector<float>& data, size_t nodes, size_t dimensions);
struct Clustering { std::vector<uint32_t> labels; double inertia = 0; int iterations = 0; };
Clustering kmeans(const std::vector<float>& data, size_t nodes, size_t dimensions, const Options&);
} // namespace sci
