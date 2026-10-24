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
    {
        // N2: an in-use GPU reader slot cannot be evicted, even if it is LRU.
        LRUCache cache(2);
        const auto a = cache.lookup(0);
        const auto b = cache.lookup(1);
        cache.set_filled(a.slot);
        cache.set_filled(b.slot);
        cache.set_recycle_guard([&](size_t slot) { return slot != a.slot; });
        const auto c = cache.lookup(2);
        require(c.evicted && *c.evicted == 1 && c.slot == b.slot);
        require(cache.lookup(0).hit);
    }
    {
        // All leased slots -> fail before mutation, no stale hit, no eviction.
        LRUCache cache(2);
        auto a = cache.lookup(0);
        auto b = cache.lookup(1);
        cache.set_filled(a.slot);
        cache.set_filled(b.slot);
        cache.set_recycle_guard([](size_t) { return false; });
        bool rejected = false;
        try { (void)cache.lookup(2); } catch (const std::runtime_error&) { rejected = true; }
        require(rejected && cache.size() == 2 && cache.evictions() == 0);
        require(cache.lookup(0).hit && cache.lookup(1).hit);
        cache.set_recycle_guard({});
        require(cache.lookup(2).evicted.has_value());
    }
    std::cout << "LRU regressions: 5 PASS\n";
}
