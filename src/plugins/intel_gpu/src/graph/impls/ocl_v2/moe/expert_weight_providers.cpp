// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "expert_weight_providers.hpp"

#include <iostream>
#include <unordered_map>
#include <unordered_set>
#include <utility>

// moe_otd_runtime.hpp is not self-contained: it relies on cldnn graph/runtime
// types (typed_primitive_inst, data_type_traits) being visible at include time.
// primitive_ocl_base.hpp is the established prelude that brings them in.
#include "impls/ocl_v2/primitive_ocl_base.hpp"
#include "moe_otd_runtime.hpp"

namespace ov::intel_gpu::ocl::moe {

std::optional<ExpertSlotLease> ResidentExpertWeightProvider::try_acquire_simultaneous(const std::vector<uint32_t>& experts, cldnn::stream& /*stream*/) {
    // Resident: identity mapping, always succeeds.
    return std::vector<size_t>(experts.begin(), experts.end());
}

size_t ResidentExpertWeightProvider::acquire_one(uint32_t expert, cldnn::stream& /*stream*/) {
    return expert;
}

OffloadExpertWeightProvider::OffloadExpertWeightProvider(size_t capacity,
                                                         const cldnn::MOECompressed::Config& config,
                                                         std::vector<size_t> weight_bin_offsets,
                                                         std::filesystem::path weights_path,
                                                         const std::string& trace_layer_name)
    : _capacity(capacity),
      _config(config),
      _weight_bin_offsets(std::move(weight_bin_offsets)),
      _weights_path(std::move(weights_path)),
      _cache(std::make_shared<LRUCache>(capacity)),
      _weight_reader(_weights_path) {
    if (auto* trace = moe_otd::get_trace_writer()) {
        _trace_layer = trace->register_layer(trace_layer_name, _weights_path.generic_string(), _capacity, _config.num_expert, _config.top_k);
    }
}

void OffloadExpertWeightProvider::configure_pack(moe_otd::PackMode mode, const std::filesystem::path& dir, const moe_otd::RoutedTensorLayouts& layouts) {
    _pack.reset();
    _pack_reader.reset();
    _pack_name.clear();
    if (mode == moe_otd::PackMode::off) {
        return;
    }
    OPENVINO_ASSERT(!dir.empty(), "MoE OTD pack mode requires GPU_MOE_OTD_PACK_DIR");

    const auto spec = moe_otd::make_pack_spec(_config, _weight_bin_offsets, layouts, _weight_reader);
    const auto path = dir / spec.file_name();
    if (mode == moe_otd::PackMode::build) {
        const auto built = moe_otd::build_layer_pack(dir, spec, _config, layouts, _weight_reader);
        // Layers compile on parallel threads; one write per line keeps messages from interleaving.
        std::cerr << ("MOE OTD pack " + std::string(built.reused_existing ? "reused " : "built ") + built.path.filename().string() +
                      " bytes=" + std::to_string(built.bytes_written) + "\n");
    }

    std::string reason;
    auto pack = moe_otd::PackLayout::open(path, spec, &reason);
    if (!pack) {
        OPENVINO_ASSERT(mode != moe_otd::PackMode::require, "MoE OTD pack ", path.filename().string(), " is unusable: ", reason);
        std::cerr << ("MOE OTD pack " + path.filename().string() + " unusable (" + reason + "), streaming from weights file\n");
        return;
    }
    _pack = std::move(pack);
    _pack_reader = std::make_unique<moe_otd::ParallelWeightReader>(path);
    _pack_name = path.filename().string();
}

void OffloadExpertWeightProvider::bind(cldnn::moe_weights& resident) {
    _resident = &resident;
    _cache->set_initialized();
    _bound = true;
    GPU_DEBUG_TRACE << "MOE OTD bind: resident_slots=" << _capacity << " total_experts=" << _config.num_expert
                    << " shared_expert=" << (_config.num_shared_expert > 0 ? 1 : 0) << std::endl;
}

void OffloadExpertWeightProvider::fill_routed_weight_views(cldnn::moe_weights& /*weights*/, RoutedWeightViews& views) {
    // Offloaded: point at the LRU resident buffers (not the original weight inputs).
    OPENVINO_ASSERT(_resident != nullptr, "OffloadExpertWeightProvider: resident buffers not bound");
    views.weight[0] = _resident->gate_w;
    views.scale[0] = _resident->gate_s;
    views.zp[0] = _resident->gate_z;
    views.weight[1] = _resident->up_w;
    views.scale[1] = _resident->up_s;
    views.zp[1] = _resident->up_z;
    views.weight[2] = _resident->down_w;
    views.scale[2] = _resident->down_s;
    views.zp[2] = _resident->down_z;
}

LRUCache::Lookup OffloadExpertWeightProvider::acquire_slot(uint32_t expert, cldnn::stream& stream, uint64_t trace_call) {
    auto* perf = moe_otd::get_perf_counters();
    auto* trace = moe_otd::get_trace_writer();

    const auto item = _cache->lookup(expert);
    moe_otd::ExpertLoadStats load;

    if (item.hit) {
        if (perf) {
            perf->gpu_hits.fetch_add(1, std::memory_order_relaxed);
        }
    } else {
        if (perf) {
            perf->gpu_misses.fetch_add(1, std::memory_order_relaxed);
            if (item.evicted) {
                perf->evictions.fetch_add(1, std::memory_order_relaxed);
            }
        }
        OPENVINO_ASSERT(_resident != nullptr, "OffloadExpertWeightProvider: resident buffers not bound");
        if (_pack) {
            // The slot stays unfilled on failure, so a later lookup of this expert is a miss, never a stale hit.
            try {
                moe_otd::fill_weights_from_pack(stream, *_pack, *_pack_reader, *_resident, expert, item.slot, _pack_staging, trace ? &load : nullptr);
            } catch (const moe_otd::PackIntegrityError& e) {
                OPENVINO_THROW("MoE OTD pack ", _pack_name, " rejected expert ", expert, " for slot ", item.slot, ": ", e.what(),
                               "; resident device memory was not modified");
            }
        } else {
            moe_otd::fill_weights_memory(stream, _config, _weight_bin_offsets, _weight_reader, *_resident, {expert}, {item.slot}, trace ? &load : nullptr);
        }
        _cache->set_filled(item.slot);
    }

    if (trace) {
        trace->expert(trace_call, _trace_layer, expert, item.slot, item.hit, item.evicted, load);
    }
    return item;
}

std::optional<ExpertSlotLease> OffloadExpertWeightProvider::try_acquire_simultaneous(const std::vector<uint32_t>& experts, cldnn::stream& stream) {
    std::vector<uint32_t> unique_experts;
    {
        std::unordered_set<uint32_t> seen;
        seen.reserve(experts.size());
        for (uint32_t expert : experts) {
            if (seen.insert(expert).second) {
                unique_experts.push_back(expert);
            }
        }
    }

    auto* trace = moe_otd::get_trace_writer();
    uint64_t trace_call = 0;
    if (trace) {
        trace_call = trace->begin_call(_trace_layer, moe_otd::AcquireKind::simultaneous, experts.size(), unique_experts);
    }

    // Reject before touching the cache: loading part of an oversized set would stream experts
    // and evict resident ones that the caller's per-expert fallback then has to reload.
    if (unique_experts.size() > _capacity) {
        if (trace) {
            trace->end_call(trace_call, true);
        }
        return std::nullopt;
    }

    std::unordered_map<uint32_t, size_t> expert_to_slot;
    expert_to_slot.reserve(unique_experts.size());
    for (uint32_t expert : unique_experts) {
        expert_to_slot.emplace(expert, acquire_slot(expert, stream, trace_call).slot);
    }

    std::vector<size_t> slots(experts.size());
    for (size_t i = 0; i < experts.size(); i++) {
        slots[i] = expert_to_slot.at(experts[i]);
    }

    if (trace) {
        trace->end_call(trace_call, false);
    }
    return slots;
}

size_t OffloadExpertWeightProvider::acquire_one(uint32_t expert, cldnn::stream& stream) {
    auto* trace = moe_otd::get_trace_writer();
    uint64_t trace_call = 0;
    if (trace) {
        trace_call = trace->begin_call(_trace_layer, moe_otd::AcquireKind::one, 1, {expert});
    }
    const auto slot = acquire_slot(expert, stream, trace_call).slot;
    if (trace) {
        trace->end_call(trace_call, false);
    }
    return slot;
}

}  // namespace ov::intel_gpu::ocl::moe
