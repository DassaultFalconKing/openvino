// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
#include "../../src/graph/impls/ocl_v2/moe/lru_cache.hpp"
#include <stdexcept>
#include <iostream>

using ov::intel_gpu::ocl::moe::LRUCache;
static void require(bool value) {
    if (!value) throw std::runtime_error("LRU regression");
}
int main() {
    {
        LRUCache cache(2);
        cache.get_lru_item(0);
        cache.get_lru_item(1);
        cache.evict_one();
        const auto a = cache.get_lru_item(2);
        const auto b = cache.get_lru_item(1);
        require(a.first != b.first);
    }
    {
        LRUCache cache(2);
        cache.set_priorities({2, 0, 1});
        cache.get_lru_item(0);
        cache.set_filled(0);
        cache.get_lru_item(1);
        cache.set_filled(1);
        const auto item = cache.lookup(2);
        require(item.evicted && *item.evicted == 1);
        require(cache.lookup(0).hit);
    }
    {
        LRUCache cache(2);
        cache.set_priorities({0, 2, 0, 0});
        cache.get_lru_item(0);
        cache.get_lru_item(1);
        const std::vector<size_t> required{2, 3};
        const auto first = cache.lookup(2, required);
        const auto second = cache.lookup(3, required);
        require(first.slot != second.slot);
    }
    std::cout << "LRU regressions: 3 PASS\n";
}
