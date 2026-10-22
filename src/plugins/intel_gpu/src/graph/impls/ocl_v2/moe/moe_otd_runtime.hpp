// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <atomic>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include "intel_gpu/primitives/moe_3gemm_fused_compressed.hpp"
#include "intel_gpu/runtime/stream.hpp"
#include "lru_cache.hpp"
#include "moe_otd_trace.hpp"
#include "openvino/util/parallel_io.hpp"

namespace ov::intel_gpu::ocl::moe_otd {

// Plain-value copy of OtdPerfCounters, suitable for before/after deltas around a measured run.
struct OtdPerfSnapshot {
    uint64_t gpu_hits = 0;
    uint64_t gpu_misses = 0;
    uint64_t evictions = 0;
    uint64_t disk_io_ns = 0;
    uint64_t disk_read_bytes = 0;
    uint64_t disk_reads = 0;
    uint64_t transpose_ns = 0;
    uint64_t gpu_copy_ns = 0;
    uint64_t tensor_load_count = 0;
    uint64_t expert_loads = 0;
    uint64_t overflow_partial_loads = 0;
    uint64_t batched_fallbacks = 0;
    uint64_t grouped_fallbacks = 0;
    uint64_t created_onednn_kernels = 0;
    uint64_t disk_read_p50_us = 0;
    uint64_t disk_read_p95_us = 0;
    uint64_t disk_read_p99_us = 0;
};

// Lightweight perf counters for OTD profiling.
// Enabled by setting MOE_OTD_PERF_LOG=1 environment variable.
// Counters are process-global and printed on process exit; they are not per-request metrics.
struct OtdPerfCounters {
    std::atomic<uint64_t> gpu_hits{0};
    std::atomic<uint64_t> gpu_misses{0};
    std::atomic<uint64_t> evictions{0};
    std::atomic<uint64_t> disk_io_ns{0};
    std::atomic<uint64_t> disk_read_bytes{0};
    std::atomic<uint64_t> disk_reads{0};
    std::atomic<uint64_t> transpose_ns{0};
    std::atomic<uint64_t> gpu_copy_ns{0};
    std::atomic<uint64_t> tensor_load_count{0};
    std::atomic<uint64_t> expert_loads{0};
    // Experts streamed by a simultaneous acquisition that then overflowed capacity. They stay
    // cached, but may be evicted again by the caller's per-expert fallback before use.
    std::atomic<uint64_t> overflow_partial_loads{0};
    std::atomic<uint64_t> batched_fallbacks{0};
    std::atomic<uint64_t> grouped_fallbacks{0};
    std::atomic<uint64_t> created_onednn_kernels{0};
    LatencyHistogram disk_read_latency;

    void record_load(const ExpertLoadStats& load);
    OtdPerfSnapshot snapshot() const;
    void dump() const;
};

OtdPerfCounters* get_perf_counters();

class ParallelWeightReader {
public:
    explicit ParallelWeightReader(const std::filesystem::path& weights_path) : _weights_path(weights_path) {}

    ~ParallelWeightReader() {
        if (_opened) {
            ov::util::close_file_handle(_shared_handle);
        }
    }

    ParallelWeightReader(const ParallelWeightReader&) = delete;
    ParallelWeightReader& operator=(const ParallelWeightReader&) = delete;

    const std::filesystem::path& path() const {
        return _weights_path;
    }

    size_t file_size() {
        ensure_open();
        return _file_size;
    }

    void read(char* dst, size_t size, size_t file_offset) {
        ensure_open();
        if (!ov::util::positional_read(_shared_handle, dst, size, file_offset)) {
            OPENVINO_THROW("Failed to read enough bytes from OTD weight file");
        }
    }

private:
    void ensure_open() {
        if (!_opened) {
            std::streamoff file_size = 0;
            ov::util::get_file_handle_and_size(_weights_path, 0, _shared_handle, file_size);
            _file_size = static_cast<size_t>(file_size);
            _opened = true;
        }
    }

    std::filesystem::path _weights_path;
    ov::FileHandle _shared_handle{};
    size_t _file_size = 0;
    bool _opened = false;
};

void maybe_transpose_scale_zp(const cldnn::MOECompressed::Config& config,
                              size_t offset_pos,
                              const cldnn::layout& layout,
                              std::vector<uint8_t>& payload,
                              size_t per_expert_size);

void fill_weights_memory(cldnn::stream& exec_stream,
                         const cldnn::MOECompressed::Config& config,
                         const std::vector<size_t>& weight_bin_offsets,
                         ParallelWeightReader& weight_reader,
                         cldnn::moe_weights& wei_mem,
                         const std::vector<uint32_t>& experts_list,
                         const std::vector<size_t>& lru_experts,
                         ExpertLoadStats* stats = nullptr);

}  // namespace ov::intel_gpu::ocl::moe_otd