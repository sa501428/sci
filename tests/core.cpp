#include "core.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <numeric>
#include <stdexcept>

template<class F> void fails(F f) {
    bool caught = false;
    try { f(); } catch (const std::exception&) { caught = true; }
    assert(caught);
}
int main() {
    sci::Random rng{123};
    sci::Graph graph;
    graph.add(0, 1, 1); graph.add(1, 2, 3); graph.add(0, 2, 6);
    graph.prepare();
    size_t hits[3]{};
    for (int i = 0; i < 300000; ++i) {
        const auto& edge = graph.sample(rng);
        ++hits[edge.u == 0 && edge.v == 1 ? 0 : edge.u == 1 ? 1 : 2];
    }
    for (int i = 0; i < 3; ++i) assert(std::abs(double(hits[i])/300000 - (i == 0 ? .1 : i == 1 ? .3 : .6)) < .004);
    sci::Alias negative({0, 1, 3, 6});
    size_t nhits[4]{};
    for (int i = 0; i < 300000; ++i) ++nhits[negative.sample(rng)];
    assert(nhits[0] == 0);
    for (int i = 1; i < 4; ++i) assert(std::abs(double(nhits[i])/300000 - (i == 1 ? .1 : i == 2 ? .3 : .6)) < .004);
    fails([] { sci::Alias a({0, 0}); });
    fails([] { sci::Alias a({-1, 2}); });
    fails([] { sci::Graph g; g.prepare(); });
    fails([] { sci::Graph g; g.add(0, 1, 0); });
    // Cross a graph chunk boundary and test a highly skewed distribution.
    sci::Graph big;
    for (int i = 0; i < 70000; ++i) big.add(0, uint32_t(i+1), i == 69999 ? 1e8f : 1.f);
    big.prepare();
    size_t last = 0;
    for (int i = 0; i < 10000; ++i) last += big.sample(rng).v == 70000;
    assert(last > 9970);
    sci::Options o;
    o.samples = 2000; o.dimensions = 8; o.negatives = 2;
    sci::Alias neg({1, 1, 1});
    for (int order : {1, 2}) {
        auto a = sci::train(graph, neg, 3, o, order);
        assert(a == sci::train(graph, neg, 3, o, order));
        std::vector<float> combined(3*16, 0);
        sci::trainInto(graph, neg, 3, o, order, combined, 16, 8);
        for (size_t i = 0; i < 3; ++i) for (size_t d = 0; d < 8; ++d) assert(combined[i*16+8+d] == a[i*8+d]);
        for (float v : a) assert(std::isfinite(v));
        o.threads = 4;
        for (float v : sci::train(graph, neg, 3, o, order)) assert(std::isfinite(v));
        o.threads = 1;
    }
    // One update, no negatives: compare to an independent direct SGD formula.
    sci::Options one; one.samples = 1; one.dimensions = 3; one.negatives = 0;
    sci::Graph single; single.add(0, 1, 1); single.prepare();
    sci::Alias two({1, 1});
    sci::Random init{one.seed}; std::vector<float> expected(6);
    for (float& x : expected) x = float((init.uniform() - .5)/3);
    sci::Random choose{one.seed + 0x9e3779b97f4a7c15ULL};
    single.sample(choose); bool reverse = (choose.next() & 1) != 0;
    size_t u = reverse ? 3 : 0, v = reverse ? 0 : 3;
    float dot = 0;
    for (size_t d = 0; d < 3; ++d) dot += expected[u+d]*expected[v+d];
    double x = 12.0 * size_t((dot+6)*(4096.f/12)) / 4096 - 6;
    float g = (1 - float(1/(1+std::exp(-x)))) * one.learningRate;
    auto before = expected;
    for (size_t d = 0; d < 3; ++d) {
        expected[u+d] += g * before[v+d]; expected[v+d] += g * before[u+d];
    }
    auto actual = sci::train(single, two, 2, one, 1);
    for (size_t d = 0; d < 6; ++d) assert(std::abs(actual[d] - expected[d]) < 1e-7);
    std::vector<float> points{0,0, 0,2, 2,0, 100,100, 100,102, 102,100};
    o.clusters = 2; o.restarts = 4;
    auto c = sci::kmeans(points, 6, 2, o);
    assert(std::abs(c.inertia - 32./3) < 1e-4);
    assert(c.labels[0] == c.labels[1] && c.labels[1] == c.labels[2]);
    assert(c.labels[3] == c.labels[4] && c.labels[4] == c.labels[5] && c.labels[0] != c.labels[3]);
    o.threads = 3;
    assert(std::abs(sci::kmeans(points, 6, 2, o).inertia - c.inertia) < 1e-6);
    o.clusters = 1;
    assert(std::abs(sci::kmeans(points, 6, 2, o).inertia - 30010.6666667) < 1e-3);
    o.clusters = 3;
    auto identical = sci::kmeans(std::vector<float>(12, 5), 6, 2, o);
    assert(identical.inertia == 0);
    o.clusters = 7;
    fails([&] { sci::kmeans(points, 6, 2, o); });
    std::vector<float> norm{3, 4, 0, 0}; sci::normalize(norm, 2, 2);
    assert(std::abs(norm[0] - .6f) < 1e-6 && norm[2] == 0);
    std::cout << "alias distributions, SGD, strided embeddings, bounds and clustering passed\n";
}
