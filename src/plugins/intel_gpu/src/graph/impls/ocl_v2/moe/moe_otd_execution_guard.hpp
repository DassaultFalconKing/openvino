// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <exception>
#include <functional>
#include <mutex>
#include <stdexcept>
#include <utility>

namespace ov::intel_gpu::ocl::moe_otd {
// Conservative fallback until terminal event ownership is implemented for every
// dispatch path. Shared by provider clones. A failed GPU drain poisons the gate;
// no subsequent request may recycle its potentially live resident arena.
class SynchronousExecutionGate {
public:
    class Scope {
    public:
        Scope(SynchronousExecutionGate& gate, std::function<void()> drain, bool enabled = true)
            : gate_(gate), lock_(gate.mutex_, std::defer_lock), drain_(std::move(drain)), enabled_(enabled) {
            if (!enabled_) return;
            lock_.lock();
            if (gate_.poisoned_) throw std::logic_error("OTD execution arena poisoned by failed GPU drain");
            if (!drain_) throw std::invalid_argument("OTD execution requires a GPU drain callback");
        }
        Scope(const Scope&) = delete;
        Scope& operator=(const Scope&) = delete;
        ~Scope() noexcept(false) {
            if (!enabled_) return;
            try {
                drain_();
            } catch (...) {
                gate_.poisoned_ = true;
                if (std::uncaught_exceptions() == 0) throw;
            }
        }
    private:
        SynchronousExecutionGate& gate_;
        std::unique_lock<std::mutex> lock_;
        std::function<void()> drain_;
        bool enabled_;
    };
    Scope enter(std::function<void()> drain, bool enabled = true) { return Scope(*this, std::move(drain), enabled); }
private:
    std::mutex mutex_;
    bool poisoned_ = false;
};
}  // namespace ov::intel_gpu::ocl::moe_otd
