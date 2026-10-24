// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include <functional>
#include <list>
#include <mutex>
#include <optional>
#include <unordered_map>
#include <utility>
#include <vector>
#include <cstdint>

namespace ov::intel_gpu::ocl::moe {

class LRUCache {
public:
    struct Lookup {
        size_t slot = 0;
        bool hit = false;
        // Expert whose slot was recycled to make room, if any.
        std::optional<size_t> evicted;
    };

    explicit LRUCache(size_t max_total_experts);

    std::pair<size_t, bool> get_lru_item(size_t expert);
    Lookup lookup(size_t expert, const std::vector<size_t>& required_experts = {});

    // Profile policy tiers: 0=cold/default LRU, 1=warm, 2=hot.
    // Advisory priorities cannot prohibit a demand miss: least-protected
    // resident wins; oldest wins ties. Empty vector restores exact LRU.
    void set_priorities(std::vector<uint8_t> priorities);

    // N2 integration point: veto a candidate slot BEFORE changing LRU mapping.
    // The callback MUST be nonblocking, noexcept in practice, and must not
    // re-enter this cache. Wire to terminal-consumer completion, not producer
    // copy completion. An empty guard preserves legacy synchronous behaviour.
    void set_recycle_guard(std::function<bool(size_t)> can_recycle_slot);
    void evict_one();

    size_t size() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_total_experts;
    }

    size_t evictions() const {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_evictions;
    }

    void set_filled(size_t lru_expert_no) {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (lru_expert_no >= m_filled_list.size()) {
            return;
        }
        m_filled_list[lru_expert_no] = true;
    }

    bool is_initialized() const {
        return m_initialized;
    }
    void set_initialized() {
        m_initialized = true;
    }

private:
    bool m_initialized = false;
    struct Key {
        size_t expert;
        bool operator==(const Key& other) const noexcept {
            return expert == other.expert;
        }
    };

    struct KeyHash {
        std::size_t operator()(const Key& k) const noexcept {
            return std::hash<size_t>()(k.expert);
        }
    };

    struct Node {
        size_t expert;
        size_t lru_expert_no;
    };

    size_t m_max_total_experts;
    size_t m_total_experts;
    size_t m_evictions = 0;

    std::list<Node> m_list;
    std::vector<bool> m_filled_list;
    // Unoccupied slots, lowest index at the back so fresh caches assign slots 0, 1, 2, ...
    std::vector<size_t> m_free_slots;
    std::vector<uint8_t> m_priorities;
    std::function<bool(size_t)> m_recycle_guard;
    std::unordered_map<Key, std::list<Node>::iterator, KeyHash> m_map;
    mutable std::mutex m_mutex;

    uint8_t priority_of(size_t expert) const;
    void move_to_end(std::list<Node>::iterator it);
    std::optional<Node> evict_one_unlocked(const std::vector<size_t>& required_experts = {});
};

}  // namespace ov::intel_gpu::ocl::moe
