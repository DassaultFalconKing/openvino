// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
#pragma once

// CIB2 N3/N4: single-LPDDR budget + non-executing prefetch suggestion queue.
// No NVMe reads / GPU allocation / background scheduling occurs here.
// A suggestion requires a real slot-generation lease AND admission on dispatch.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace ov::intel_gpu::ocl::moe_otd {

class UmaAdmissionLedger {
public:
    enum class Class { kv, weights, workspace, draft, staging, prefetch };
    struct ObservedFree {
        uint64_t system_bytes = 0, gpu_budget_bytes = 0, os_reserve_bytes = 0;
    };
    struct Ticket { uint64_t id; };
    struct Demand { Class memory_class; uint64_t bytes; bool gpu_visible; };
    explicit UmaAdmissionLedger(ObservedFree available) : free_(available) {}
    // Refresh only after all outstanding provisional reservations retire.
    // Runtime reconciliation with WDDM/OS accounting needs a separate adapter.
    void reset(ObservedFree available) {
        std::lock_guard<std::mutex> lock(mu_);
        if (!active_.empty()) throw std::logic_error("live UMA reservations");
        free_ = available;
    }
    // A fresh OS/WDDM sample is allowed while reservations exist. This
    // conservative accounting may double-count already-committed allocations,
    // but cannot admit more than either observed free constraint permits.
    void observe(ObservedFree available) {
        std::lock_guard<std::mutex> lock(mu_);
        free_ = available;
    }
    std::optional<Ticket> reserve(Class memory_class, uint64_t bytes, bool gpu_visible) {
        if (!bytes) return std::nullopt;
        std::lock_guard<std::mutex> lock(mu_);
        const uint64_t system_budget = free_.system_bytes > free_.os_reserve_bytes ?
            free_.system_bytes - free_.os_reserve_bytes : 0;
        if (bytes > system_budget || system_used_ > system_budget - bytes) return std::nullopt;
        // GPU and CPU budgets overlap physically; never add them together.
        if (gpu_visible && (bytes > free_.gpu_budget_bytes ||
                           gpu_used_ > free_.gpu_budget_bytes - bytes)) return std::nullopt;
        if (next_id_ == UINT64_MAX) throw std::overflow_error("UMA ticket counter");
        auto id = ++next_id_;
        active_.emplace(id, Reservation{memory_class, bytes, gpu_visible});
        system_used_ += bytes;
        if (gpu_visible) gpu_used_ += bytes;
        return Ticket{id};
    }
    // All-or-nothing admission for a model + KV + workspace + staging cohort.
    // Neither failed admission nor integer overflow mutates the ledger.
    std::optional<std::vector<Ticket>> reserve_batch(const std::vector<Demand>& demands) {
        if (demands.empty()) return std::nullopt;
        std::lock_guard<std::mutex> lock(mu_);
        uint64_t system_add = 0, gpu_add = 0;
        for (const auto& demand : demands) {
            if (!demand.bytes || demand.bytes > UINT64_MAX - system_add)
                return std::nullopt;
            system_add += demand.bytes;
            if (demand.gpu_visible) {
                if (demand.bytes > UINT64_MAX - gpu_add) return std::nullopt;
                gpu_add += demand.bytes;
            }
        }
        const uint64_t system_budget = free_.system_bytes > free_.os_reserve_bytes ?
            free_.system_bytes - free_.os_reserve_bytes : 0;
        if (system_add > system_budget || system_used_ > system_budget - system_add ||
            gpu_add > free_.gpu_budget_bytes || gpu_used_ > free_.gpu_budget_bytes - gpu_add ||
            demands.size() > UINT64_MAX - next_id_)
            return std::nullopt;
        std::vector<Ticket> tickets;
        tickets.reserve(demands.size());
        for (const auto& demand : demands) {
            const auto id = ++next_id_;
            active_.emplace(id, Reservation{demand.memory_class, demand.bytes, demand.gpu_visible});
            tickets.push_back({id});
        }
        system_used_ += system_add;
        gpu_used_ += gpu_add;
        return tickets;
    }
    void release(Ticket t) {
        std::lock_guard<std::mutex> lock(mu_);
        auto it = active_.find(t.id);
        if (it == active_.end()) throw std::logic_error("stale UMA ticket");
        system_used_ -= it->second.bytes;
        if (it->second.gpu_visible) gpu_used_ -= it->second.bytes;
        active_.erase(it);
    }
    uint64_t gpu_reserved() const {
        std::lock_guard<std::mutex> lock(mu_);
        return gpu_used_;
    }
    uint64_t system_reserved() const {
        std::lock_guard<std::mutex> lock(mu_);
        return system_used_;
    }
private:
    struct Reservation { Class cls; uint64_t bytes; bool gpu_visible; };
    mutable std::mutex mu_;
    ObservedFree free_;
    uint64_t next_id_ = 0, system_used_ = 0, gpu_used_ = 0;
    std::map<uint64_t, Reservation> active_;
};

// Shared process-wide registry keyed by the *executing* GPU LUID, not
// model/layer. The same ledger must be used by model/KV/workspace/OTD owners.
// It does not automatically intercept GPU allocator calls: until those owners
// register their actual allocations, N3 native runtime admission remains OFF.
class GlobalUmaAdmission {
public:
    static std::shared_ptr<UmaAdmissionLedger> for_adapter(
        const std::string& luid_hex, UmaAdmissionLedger::ObservedFree initial) {
        if (luid_hex.size() != 16 ||
            !std::all_of(luid_hex.begin(), luid_hex.end(), [](char c) {
                return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
            })) throw std::invalid_argument("exact 16-hex GPU LUID required");
        static std::mutex registry_mutex;
        static std::map<std::string, std::weak_ptr<UmaAdmissionLedger>> registry;
        std::lock_guard<std::mutex> lock(registry_mutex);
        if (auto existing = registry[luid_hex].lock()) return existing;
        auto ledger = std::make_shared<UmaAdmissionLedger>(initial);
        registry[luid_hex] = ledger;
        return ledger;
    }
};

struct PredictedExpert {
    uint32_t layer, expert;
    double probability, measured_miss_stall_us, load_cost_us;
    double possible_overlap_us, eviction_penalty_us;
    uint64_t staging_bytes;
    bool resident, already_loading;
};
struct PrearmSuggestion {
    uint32_t layer, expert;
    double expected_net_saved_us;
    uint64_t staging_bytes;
};

// "Arm" is planning only until GPU completion, reader lifetime and
// cross-layer admission are integrated with the kernel's terminal event.
// Demand always wins; an already-resident or loading candidate is skipped.
inline std::vector<PrearmSuggestion> plan_prefetch(
    const std::vector<PredictedExpert>& predictions,
    size_t available_slots, uint64_t available_staging_bytes) {
    std::vector<PrearmSuggestion> candidates;
    if (!available_slots || !available_staging_bytes) return candidates;
    for (const auto& p : predictions) {
        if (!std::isfinite(p.probability) || !std::isfinite(p.measured_miss_stall_us) ||
            !std::isfinite(p.load_cost_us) || !std::isfinite(p.possible_overlap_us) ||
            !std::isfinite(p.eviction_penalty_us) || p.probability < 0 ||
            p.probability > 1 || p.measured_miss_stall_us < 0 || p.load_cost_us < 0 ||
            p.possible_overlap_us < 0 || p.eviction_penalty_us < 0 ||
            !p.staging_bytes || p.resident || p.already_loading) continue;
        const double hidden_cost = std::max(0.0, p.load_cost_us - p.possible_overlap_us);
        const double gain = p.probability * p.measured_miss_stall_us -
                            hidden_cost - p.eviction_penalty_us;
        if (std::isfinite(gain) && gain > 0)
            candidates.push_back({p.layer, p.expert, gain, p.staging_bytes});
    }
    std::sort(candidates.begin(), candidates.end(), [](const auto& a, const auto& b) {
        if (a.expected_net_saved_us != b.expected_net_saved_us)
            return a.expected_net_saved_us > b.expected_net_saved_us;
        return std::pair<uint32_t,uint32_t>(a.layer, a.expert) <
               std::pair<uint32_t,uint32_t>(b.layer, b.expert);
    });
    std::vector<PrearmSuggestion> selected;
    for (const auto& p : candidates) {
        if (selected.size() >= available_slots) break;
        if (p.staging_bytes > available_staging_bytes) continue;
        if (std::any_of(selected.begin(), selected.end(), [&](const auto& item) {
            return item.layer == p.layer && item.expert == p.expert;
        })) continue;
        selected.push_back(p);
        available_staging_bytes -= p.staging_bytes;
    }
    return selected;
}
} // namespace ov::intel_gpu::ocl::moe_otd
