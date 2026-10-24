// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
#pragma once

// CIB2 N2: conservative GPU slot lifetime ledger. This is an integration
// primitive, not yet wired to moe_3gemm_swiglu_opt's terminal GPU events.
// No event -> no release: never interpret a missing event as completed.
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <map>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace ov::intel_gpu::ocl::moe_otd {
using GpuCompletion = std::function<bool()>; // nonblocking, strong event ownership

class GpuSlotLeaseLedger {
public:
    struct Ticket { size_t slot; uint64_t generation; uint64_t id; };
    enum class State { empty, writing, copying, resident };
    explicit GpuSlotLeaseLedger(size_t count) : slots_(count) {
        if (!count) throw std::invalid_argument("zero GPU cache slots");
    }
    std::optional<Ticket> begin_write(size_t slot) {
        std::lock_guard<std::mutex> lock(mu_);
        auto& s = at(slot);
        collect(s);
        if (s.readers || !s.pending_readers.empty() ||
            s.state == State::writing || s.state == State::copying) return std::nullopt;
        if (s.generation == UINT64_MAX || sequence_ == UINT64_MAX)
            throw std::overflow_error("GPU lease counter overflow");
        s.state = State::writing;
        s.writer_id = ++sequence_;
        return Ticket{slot, ++s.generation, s.writer_id};
    }
    void submit_write(Ticket t, GpuCompletion completion) {
        if (!completion) throw std::invalid_argument("GPU copy event required");
        std::lock_guard<std::mutex> lock(mu_);
        auto& s = at(t.slot);
        if (s.state != State::writing || s.generation != t.generation ||
            s.writer_id != t.id) throw std::logic_error("stale GPU writer ticket");
        s.copy_done = std::move(completion);
        s.state = State::copying;
    }
    void cancel_unsubmitted_write(Ticket t) {
        std::lock_guard<std::mutex> lock(mu_);
        auto& s = at(t.slot);
        if (s.state != State::writing || s.generation != t.generation ||
            s.writer_id != t.id) throw std::logic_error("invalid GPU write cancellation");
        s.state = State::empty;
    }
    std::optional<Ticket> begin_read(size_t slot, uint64_t expected_generation) {
        std::lock_guard<std::mutex> lock(mu_);
        auto& s = at(slot);
        collect(s);
        if (s.state != State::resident || s.generation != expected_generation)
            return std::nullopt;
        if (sequence_ == UINT64_MAX) throw std::overflow_error("GPU reader counter overflow");
        auto id = ++sequence_;
        active_.emplace(id, std::pair<size_t, uint64_t>(slot, expected_generation));
        ++s.readers;
        return Ticket{slot, expected_generation, id};
    }
    void retire_read(Ticket t, GpuCompletion completion) {
        if (!completion) throw std::invalid_argument("GPU consumer event required");
        std::lock_guard<std::mutex> lock(mu_);
        auto it = active_.find(t.id);
        if (it == active_.end() || it->second.first != t.slot ||
            it->second.second != t.generation) throw std::logic_error("unknown GPU reader ticket");
        auto& s = at(t.slot);
        if (!s.readers || s.generation != t.generation)
            throw std::logic_error("GPU reader generation mismatch");
        s.pending_readers.push_back(std::move(completion));
        --s.readers;
        active_.erase(it);
    }
    bool can_evict(size_t slot) {
        std::lock_guard<std::mutex> lock(mu_);
        auto& s = at(slot);
        collect(s);
        return !s.readers && s.pending_readers.empty() &&
               s.state != State::writing && s.state != State::copying;
    }
    bool resident(size_t slot, uint64_t expected_generation) {
        std::lock_guard<std::mutex> lock(mu_);
        auto& s = at(slot);
        collect(s);
        return s.state == State::resident && s.generation == expected_generation;
    }
    uint64_t generation(size_t slot) const {
        std::lock_guard<std::mutex> lock(mu_);
        return at(slot).generation;
    }
private:
    struct Slot {
        uint64_t generation = 0, writer_id = 0;
        size_t readers = 0;
        State state = State::empty;
        GpuCompletion copy_done;
        std::vector<GpuCompletion> pending_readers;
    };
    static bool complete(const GpuCompletion& f) noexcept {
        if (!f) return false;
        try { return f(); } catch (...) { return false; }
    }
    static void collect(Slot& s) {
        if (s.state == State::copying && complete(s.copy_done)) {
            s.state = State::resident;
            s.copy_done = {};
        }
        auto& q = s.pending_readers;
        q.erase(std::remove_if(q.begin(), q.end(), [](const auto& x) {
            return complete(x);
        }), q.end());
    }
    Slot& at(size_t id) {
        if (id >= slots_.size()) throw std::out_of_range("GPU slot out of range");
        return slots_[id];
    }
    const Slot& at(size_t id) const {
        if (id >= slots_.size()) throw std::out_of_range("GPU slot out of range");
        return slots_[id];
    }
    mutable std::mutex mu_;
    std::vector<Slot> slots_;
    std::map<uint64_t, std::pair<size_t, uint64_t>> active_;
    uint64_t sequence_ = 0;
};
} // namespace ov::intel_gpu::ocl::moe_otd
