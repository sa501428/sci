#include "core.h"
#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <exception>
#include <limits>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <thread>

namespace sci {
uint64_t Random::next() {
    uint64_t z = (state += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}
uint32_t Random::index(uint32_t n) {
    if (!n) throw std::invalid_argument("cannot sample an empty distribution");
    uint64_t m = uint64_t(uint32_t(next())) * n;
    if (uint32_t(m) < n) {
        const uint32_t threshold = uint32_t(-n) % n;
        while (uint32_t(m) < threshold) m = uint64_t(uint32_t(next())) * n;
    }
    return uint32_t(m >> 32);
}
void Graph::add(uint32_t u, uint32_t v, float w) {
    if (count == UINT32_MAX) throw std::runtime_error("graph exceeds 2^32-1 undirected edges; increase resolution");
    if (!(w > 0) || !std::isfinite(w)) throw std::invalid_argument("graph weights must be finite and positive");
    if (!(count & (chunk - 1))) blocks.emplace_back(new Edge[chunk]);
    (*this)[count++] = {u, v, w, 0};
}
// One workspace serves both small and large stacks. The input weight slot is
// reused as the final probability; no extra weight/probability arrays survive.
template<class Probability, class SetAlias>
static void makeAlias(uint32_t n, Probability p, SetAlias setAlias) {
    double sum = 0;
    for (uint32_t i = 0; i < n; ++i) sum += p(i);
    if (!(sum > 0) || !std::isfinite(sum)) throw std::runtime_error("empty or invalid sampling distribution");
    std::vector<uint32_t> work(n);
    size_t small = 0, large = n;
    for (uint32_t i = 0; i < n; ++i) {
        p(i) = float(double(p(i)) * n / sum);
        if (p(i) < 1) work[small++] = i;
        else work[--large] = i;
    }
    while (small && large < n) {
        uint32_t s = work[--small], l = work[large++];
        setAlias(s, l);
        p(l) = float(double(p(l)) + p(s) - 1);
        if (p(l) < 1) work[small++] = l;
        else work[--large] = l;
    }
    while (small) p(work[--small]) = 1;
    while (large < n) p(work[large++]) = 1;
}
void Graph::prepare() {
    makeAlias(size(), [&](uint32_t i) -> float& { return (*this)[i].probability; },
              [&](uint32_t i, uint32_t j) { (*this)[i].alias = j; });
}
const Edge& Graph::sample(Random& r) const {
    uint32_t i = r.index(size());
    return (*this)[r.uniform() < (*this)[i].probability ? i : (*this)[i].alias];
}
Alias::Alias(const std::vector<double>& weights) : probability(weights.size()), alias(weights.size()) {
    if (weights.empty() || weights.size() > UINT32_MAX) throw std::invalid_argument("invalid node distribution size");
    // Scale before float conversion to avoid overflow/underflow in large counts.
    double maximum = *std::max_element(weights.begin(), weights.end());
    if (!(maximum > 0) || !std::isfinite(maximum)) throw std::invalid_argument("invalid negative sampling weights");
    for (size_t i = 0; i < weights.size(); ++i) {
        if (weights[i] < 0 || !std::isfinite(weights[i])) throw std::invalid_argument("invalid negative sampling weights");
        probability[i] = float(weights[i] / maximum);
    }
    makeAlias(uint32_t(weights.size()), [&](uint32_t i) -> float& { return probability[i]; },
              [&](uint32_t i, uint32_t j) { alias[i] = j; });
}
uint32_t Alias::sample(Random& r) const {
    uint32_t i = r.index(uint32_t(probability.size()));
    return r.uniform() < probability[i] ? i : alias[i];
}
template<class F> static void parallel(int threads, F fn) {
    if (threads == 1) { fn(0); return; }
    std::vector<std::thread> workers;
    std::exception_ptr failure;
    std::mutex failureMutex;
    try {
        for (int t = 0; t < threads; ++t) workers.emplace_back([&, t] {
            try { fn(t); }
            catch (...) { std::lock_guard<std::mutex> guard(failureMutex); if (!failure) failure = std::current_exception(); }
        });
    } catch (...) {
        for (auto& w : workers) w.join();
        throw;
    }
    for (auto& w : workers) w.join();
    if (failure) std::rethrow_exception(failure);
}
void trainInto(const Graph& graph, const Alias& negative, size_t n, const Options& o, int order,
               std::vector<float>& vertex, size_t stride, size_t offset) {
    if (!n || !graph.size() || o.dimensions < 1 || o.negatives < 0 || o.threads < 1 ||
        !o.samples || (order != 1 && order != 2)) throw std::invalid_argument("invalid LINE configuration");
    const size_t dim = size_t(o.dimensions);
    if (n > std::numeric_limits<size_t>::max() / dim) throw std::overflow_error("embedding too large");
    if (stride < dim || offset > stride - dim || vertex.size() / stride != n || vertex.size() % stride)
        throw std::invalid_argument("invalid embedding destination");
    std::vector<float> context;
    if (order == 2) context.resize(n * dim, 0);
    Random init{o.seed};
    for (size_t i = 0; i < n; ++i) for (size_t d = 0; d < dim; ++d)
        vertex[i * stride + offset + d] = float((init.uniform() - 0.5) / dim);
    std::array<float, 4097> sigmoid{};
    for (size_t i = 0; i < sigmoid.size(); ++i) {
        double x = 12.0 * i / 4096 - 6;
        sigmoid[i] = float(1 / (1 + std::exp(-x)));
    }
    // Ordered stripe locks protect every touched row, including reads. This
    // avoids the C++ data races of shared-array Hogwild implementations.
    constexpr uint32_t stripes = 4096;
    std::unique_ptr<std::mutex[]> locks;
    if (o.threads > 1) locks.reset(new std::mutex[stripes]);
    std::atomic<uint64_t> next{0};
    const int threads = int(std::min<uint64_t>(o.threads, (o.samples + 255) / 256));
    parallel(threads, [&](int t) {
        Random rng{o.seed + uint64_t(t + 1) * 0x9e3779b97f4a7c15ULL};
        std::vector<float> error(dim);
        std::vector<uint32_t> targets(size_t(o.negatives) + 1), touched(targets.size() + 1);
        for (;;) {
            uint64_t begin = next.fetch_add(256, std::memory_order_relaxed);
            if (begin >= o.samples) break;
            uint64_t end = std::min(begin + 256, o.samples);
            const float rho = o.learningRate * std::max(0.0001, 1.0 - double(begin) / o.samples);
            for (uint64_t step = begin; step < end; ++step) {
                const Edge& e = graph.sample(rng);
                const bool reverse = (rng.next() & 1) != 0;
                const uint32_t u = reverse ? e.v : e.u;
                targets[0] = reverse ? e.u : e.v;
                for (size_t j = 1; j < targets.size(); ++j) targets[j] = negative.sample(rng);
                size_t acquired = 0;
                if (locks) {
                    touched[0] = u % stripes;
                    for (size_t j = 0; j < targets.size(); ++j) touched[j + 1] = targets[j] % stripes;
                    std::sort(touched.begin(), touched.end());
                    acquired = size_t(std::unique(touched.begin(), touched.end()) - touched.begin());
                    for (size_t j = 0; j < acquired; ++j) locks[touched[j]].lock();
                }
                float* source = vertex.data() + size_t(u) * stride + offset;
                std::fill(error.begin(), error.end(), 0);
                bool diverged = false;
                for (size_t j = 0; j < targets.size(); ++j) {
                    float* target = order == 1 ? vertex.data() + size_t(targets[j]) * stride + offset
                                              : context.data() + size_t(targets[j]) * dim;
                    float dot = 0;
#if defined(__clang__)
#pragma clang loop vectorize(enable)
#endif
                    for (size_t d = 0; d < dim; ++d) dot += source[d] * target[d];
                    if (!std::isfinite(dot)) { diverged = true; break; }
                    float sig = dot <= -6 ? 0 : dot >= 6 ? 1 : sigmoid[size_t((dot + 6) * (4096.f / 12))];
                    float g = (float(j == 0) - sig) * rho;
                    // Fused pass is safe even when target aliases source: each
                    // component's old target is consumed before it is updated.
                    for (size_t d = 0; d < dim; ++d) {
                        error[d] += g * target[d];
                        target[d] += g * source[d];
                    }
                }
                for (size_t d = 0; d < dim; ++d) source[d] += error[d];
                for (size_t j = acquired; j > 0; --j) locks[touched[j - 1]].unlock();
                if (diverged) throw std::runtime_error("LINE diverged; reduce --learning-rate");
            }
        }
    });
    for (size_t i = 0; i < n; ++i) for (size_t d = 0; d < dim; ++d)
        if (!std::isfinite(vertex[i * stride + offset + d])) throw std::runtime_error("LINE diverged; reduce --learning-rate");
}
std::vector<float> train(const Graph& graph, const Alias& negative, size_t n, const Options& o, int order) {
    if (o.dimensions < 1 || n > std::numeric_limits<size_t>::max() / size_t(o.dimensions))
        throw std::invalid_argument("invalid embedding size");
    std::vector<float> result(n * size_t(o.dimensions));
    trainInto(graph, negative, n, o, order, result, size_t(o.dimensions), 0);
    return result;
}
void normalize(std::vector<float>& x, size_t n, size_t dim) {
    for (size_t i = 0; i < n; ++i) {
        double norm = 0;
        for (size_t d = 0; d < dim; ++d) norm += double(x[i * dim + d]) * x[i * dim + d];
        if (norm > 0) {
            float scale = float(1 / std::sqrt(norm));
            for (size_t d = 0; d < dim; ++d) x[i * dim + d] *= scale;
        }
    }
}
static double squared(const float* a, const float* b, size_t dim) {
    double sum = 0;
#if defined(__clang__)
#pragma clang loop vectorize(enable)
#endif
    for (size_t d = 0; d < dim; ++d) { double diff = double(a[d]) - b[d]; sum += diff * diff; }
    return sum;
}
Clustering kmeans(const std::vector<float>& x, size_t n, size_t dim, const Options& o) {
    if (!n || !dim || n > UINT32_MAX || x.size() / n != dim || x.size() % n ||
        o.clusters < 1 || size_t(o.clusters) > n || o.restarts < 1 || o.iterations < 1 || o.threads < 1)
        throw std::invalid_argument("invalid k-means configuration");
    for (float v : x) if (!std::isfinite(v)) throw std::invalid_argument("non-finite k-means input");
    size_t k = size_t(o.clusters);
    const int threads = int(std::min(n, size_t(o.threads)));
    Clustering best;
    best.inertia = std::numeric_limits<double>::infinity();
    for (int restart = 0; restart < o.restarts; ++restart) {
        Random rng{o.seed + uint64_t(restart) * 0x9e3779b97f4a7c15ULL};
        std::vector<float> centers(k * dim);
        std::vector<double> nearest(n, std::numeric_limits<double>::infinity());
        std::vector<uint32_t> chosen;
        uint32_t pick = rng.index(uint32_t(n));
        // D^2 k-means++ initialization, O(N) memory, no pairwise matrix.
        for (size_t c = 0; c < k; ++c) {
            chosen.push_back(pick);
            std::copy_n(x.data() + size_t(pick) * dim, dim, centers.data() + c * dim);
            double total = 0;
            for (size_t i = 0; i < n; ++i) {
                nearest[i] = std::min(nearest[i], squared(x.data() + i * dim, centers.data() + c * dim, dim));
                total += nearest[i];
            }
            if (total > 0) {
                double threshold = rng.uniform() * total;
                pick = uint32_t(n - 1);
                for (size_t i = 0; i < n; ++i) if ((threshold -= nearest[i]) < 0) { pick = uint32_t(i); break; }
            } else {
                pick = 0;
                while (std::find(chosen.begin(), chosen.end(), pick) != chosen.end() && size_t(pick) + 1 < n) ++pick;
            }
        }
        Clustering current;
        current.labels.assign(n, UINT32_MAX);
        // Hamerly bounds skip distance calculations for settled points using
        // only two bounds per row, rather than an N x K distance table.
        std::vector<double> upper(n, std::numeric_limits<double>::infinity()), lower(n, 0), separation(k), movement(k);
        std::vector<double> sums(size_t(threads) * k * dim);
        std::vector<size_t> counts(size_t(threads) * k);
        std::vector<size_t> changed(threads);
        for (int it = 0; it < o.iterations; ++it) {
            std::fill(separation.begin(), separation.end(), std::numeric_limits<double>::infinity());
            for (size_t a = 0; a < k; ++a) for (size_t b = a + 1; b < k; ++b) {
                double half = 0.5 * std::sqrt(squared(centers.data() + a * dim, centers.data() + b * dim, dim));
                separation[a] = std::min(separation[a], half); separation[b] = std::min(separation[b], half);
            }
            std::fill(sums.begin(), sums.end(), 0); std::fill(counts.begin(), counts.end(), 0);
            std::fill(changed.begin(), changed.end(), 0);
            parallel(threads, [&](int t) {
                for (size_t i = n * size_t(t) / threads; i < n * size_t(t + 1) / threads; ++i) {
                    uint32_t label = current.labels[i];
                    bool search = label == UINT32_MAX;
                    if (!search && upper[i] > std::max(lower[i], separation[label])) {
                        upper[i] = std::sqrt(squared(x.data() + i * dim, centers.data() + size_t(label) * dim, dim));
                        search = upper[i] > std::max(lower[i], separation[label]);
                    }
                    if (search) {
                        double first = std::numeric_limits<double>::infinity(), second = first;
                        uint32_t winner = 0;
                        for (size_t c = 0; c < k; ++c) {
                            double d = squared(x.data() + i * dim, centers.data() + c * dim, dim);
                            if (d < first) { second = first; first = d; winner = uint32_t(c); }
                            else if (d < second) second = d;
                        }
                        upper[i] = std::sqrt(first); lower[i] = std::sqrt(second);
                        if (label != winner) ++changed[t];
                        current.labels[i] = label = winner;
                    }
                    ++counts[size_t(t) * k + label];
                    double* sum = sums.data() + (size_t(t) * k + label) * dim;
                    for (size_t d = 0; d < dim; ++d) sum[d] += x[i * dim + d];
                }
            });
            bool reseeded = false;
            for (size_t c = 0; c < k; ++c) {
                size_t count = 0;
                for (int t = 0; t < threads; ++t) count += counts[size_t(t) * k + c];
                double move = 0;
                // Empty clusters receive a distant point, then bounds reset.
                size_t far = 0;
                if (!count) {
                    far = size_t(std::max_element(upper.begin(), upper.end()) - upper.begin());
                    upper[far] = -1; reseeded = true;
                }
                for (size_t d = 0; d < dim; ++d) {
                    double sum = 0;
                    for (int t = 0; t < threads; ++t) sum += sums[(size_t(t) * k + c) * dim + d];
                    float value = count ? float(sum / count) : x[far * dim + d];
                    double diff = double(value) - centers[c * dim + d]; move += diff * diff;
                    centers[c * dim + d] = value;
                }
                movement[c] = std::sqrt(move);
            }
            current.iterations = it + 1;
            if (std::accumulate(changed.begin(), changed.end(), size_t(0)) == 0 && !reseeded) break;
            double maximum = *std::max_element(movement.begin(), movement.end());
            for (size_t i = 0; i < n; ++i) {
                upper[i] = reseeded ? std::numeric_limits<double>::infinity() : upper[i] + movement[current.labels[i]];
                lower[i] = reseeded ? 0 : std::max(0.0, lower[i] - maximum);
            }
        }
        // Reassign against final centers even if the iteration limit was hit.
        current.inertia = 0;
        for (size_t i = 0; i < n; ++i) {
            double distance = std::numeric_limits<double>::infinity(); uint32_t label = 0;
            for (size_t c = 0; c < k; ++c) {
                double d = squared(x.data() + i * dim, centers.data() + c * dim, dim);
                if (d < distance) { distance = d; label = uint32_t(c); }
            }
            current.labels[i] = label; current.inertia += distance;
        }
        if (current.inertia < best.inertia) best = std::move(current);
    }
    return best;
}
} // namespace sci
