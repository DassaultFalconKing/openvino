// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "lru_cache.hpp"

#include <stdexcept>

namespace ov::intel_gpu::ocl::moe {

LRUCache::LRUCache(size_t max_total_experts) : m_max_total_experts(max_total_experts), m_total_experts(0) {
    m_filled_list.resize(max_total_experts, false);
    m_free_slots.reserve(max_total_experts);
    for (size_t slot = max_total_experts; slot > 0; --slot) {
        m_free_slots.push_back(slot - 1);
    }
}

void LRUCache::set_priorities(std::vector<uint8_t> priorities) {
    for (const auto p : priorities) {
        if (p > 2) {
            throw std::invalid_argument("OTD resident priority must be 0, 1 or 2");
        }
    }
    std::lock_guard<std::mutex> lock(m_mutex);
    m_priorities = std::move(priorities);
}

uint8_t LRUCache::priority_of(size_t expert) const {
    return expert < m_priorities.size() ? m_priorities[expert] : uint8_t{0};
}

void LRUCache::move_to_end(std::list<Node>::iterator it) {
    if (std::next(it) == m_list.end()) {
        return;
    }
    m_list.splice(m_list.end(), m_list, it);
}

std::optional<LRUCache::Node> LRUCache::evict_one_unlocked() {
    if (m_list.empty()) {
        return std::nullopt;
    }

    // PGP tiers are a soft preference, not strict pins. If every resident
    // is protected, a demand miss still evicts the oldest of the least-high
    // tier. This avoids unbounded admission stalls under changing routing.
    auto victim = m_list.begin();
    for (auto it = std::next(m_list.begin()); it != m_list.end(); ++it) {
        if (priority_of(it->expert) < priority_of(victim->expert)) {
            victim = it;
            if (priority_of(victim->expert) == 0) break;
        }
    }
    const Node oldest = *victim;
    m_filled_list[oldest.lru_expert_no] = false;
    m_map.erase(Key{oldest.expert});
    m_list.erase(victim);
    --m_total_experts;
    ++m_evictions;
    return oldest;
}

void LRUCache::evict_one() {
    std::lock_guard<std::mutex> lock(m_mutex);
    if (auto evicted = evict_one_unlocked()) {
        m_free_slots.push_back(evicted->lru_expert_no);
    }
}

std::pair<size_t, bool> LRUCache::get_lru_item(size_t expert) {
    const auto result = lookup(expert);
    return {result.slot, result.hit};
}

LRUCache::Lookup LRUCache::lookup(size_t expert) {
    std::lock_guard<std::mutex> lock(m_mutex);

    Lookup result;
    Key key{expert};
    auto it = m_map.find(key);
    if (it == m_map.end()) {
        size_t to_filled_no = 0;
        if (m_free_slots.empty()) {
            const auto evicted = evict_one_unlocked();
            if (!evicted) {
                // Zero-capacity cache: nothing can be resident.
                return result;
            }
            result.evicted = evicted->expert;
            to_filled_no = evicted->lru_expert_no;
        } else {
            to_filled_no = m_free_slots.back();
            m_free_slots.pop_back();
        }
        m_list.push_back(Node{expert, to_filled_no});
        auto new_it = std::prev(m_list.end());
        m_map[key] = new_it;
        ++m_total_experts;
        result.slot = to_filled_no;
        result.hit = false;
        return result;
    }
    move_to_end(it->second);
    result.slot = it->second->lru_expert_no;
    result.hit = m_filled_list[it->second->lru_expert_no];
    return result;
}

}  // namespace ov::intel_gpu::ocl::moe
