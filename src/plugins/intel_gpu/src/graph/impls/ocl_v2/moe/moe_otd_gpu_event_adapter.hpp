// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "intel_gpu/runtime/event.hpp"
#include "moe_otd_slot_lease_v2.hpp"

#include <stdexcept>
#include <utility>

namespace ov::intel_gpu::ocl::moe_otd {

// Keep the actual GPU event alive for as long as the generation lease exists.
// Never substitute a null event with a completed signal. A missing terminal
// event must retain the synchronous drain/fail-stop fallback.
inline GpuCompletion own_gpu_completion(cldnn::event::ptr terminal_event) {
    if (!terminal_event)
        throw std::invalid_argument("N2 requires a real GPU terminal completion event");
    return [event = std::move(terminal_event)]() {
        return event->is_set();
    };
}

}  // namespace ov::intel_gpu::ocl::moe_otd
