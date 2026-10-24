// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "moe_otd_uma_prefetch_v2.hpp"
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <dxgi1_4.h>
#include <wrl/client.h>
#pragma comment(lib, "dxgi.lib")

namespace ov::intel_gpu::ocl::moe_otd {
struct WindowsUmaSnapshot {
    uint64_t system_total_bytes = 0;
    uint64_t system_available_bytes = 0;
    uint64_t wddm_budget_bytes = 0;
    uint64_t wddm_process_usage_bytes = 0;
    UmaAdmissionLedger::ObservedFree admission_limits(uint64_t os_reserve_bytes) const {
        return {system_available_bytes,
                wddm_budget_bytes > wddm_process_usage_bytes ? wddm_budget_bytes - wddm_process_usage_bytes : 0,
                os_reserve_bytes};
    }
};

// Caller must pass the actual executing GPU's LUID, never the first or largest
// adapter. WDDM remaining budget and physical RAM are overlapping constraints.
// This is a probe adapter, not model-wide allocation reconciliation. Its result
// must not reset an admission ledger that still owns pending reservations.
inline WindowsUmaSnapshot sample_windows_uma(LUID executing_adapter, UINT node = 0) {
    MEMORYSTATUSEX memory{};
    memory.dwLength = sizeof(memory);
    if (!GlobalMemoryStatusEx(&memory)) throw std::runtime_error("OTD UMA: physical RAM probe failed");
    Microsoft::WRL::ComPtr<IDXGIFactory1> factory;
    if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory))))
        throw std::runtime_error("OTD UMA: DXGI factory creation failed");
    for (UINT index = 0;; ++index) {
        Microsoft::WRL::ComPtr<IDXGIAdapter1> adapter;
        const auto status = factory->EnumAdapters1(index, &adapter);
        if (status == DXGI_ERROR_NOT_FOUND) break;
        if (FAILED(status)) throw std::runtime_error("OTD UMA: adapter enumeration failed");
        DXGI_ADAPTER_DESC1 desc{};
        if (FAILED(adapter->GetDesc1(&desc))) throw std::runtime_error("OTD UMA: adapter identity probe failed");
        if (desc.AdapterLuid.HighPart != executing_adapter.HighPart || desc.AdapterLuid.LowPart != executing_adapter.LowPart) continue;
        Microsoft::WRL::ComPtr<IDXGIAdapter3> adapter3;
        if (FAILED(adapter.As(&adapter3))) throw std::runtime_error("OTD UMA: WDDM budget interface unavailable");
        DXGI_QUERY_VIDEO_MEMORY_INFO info{};
        if (FAILED(adapter3->QueryVideoMemoryInfo(node, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &info)))
            throw std::runtime_error("OTD UMA: WDDM budget query failed");
        return {memory.ullTotalPhys, memory.ullAvailPhys, info.Budget, info.CurrentUsage};
    }
    throw std::runtime_error("OTD UMA: executing GPU LUID not found");
}
}  // namespace ov::intel_gpu::ocl::moe_otd
#endif
