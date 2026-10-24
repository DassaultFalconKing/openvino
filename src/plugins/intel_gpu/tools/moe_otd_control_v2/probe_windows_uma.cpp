// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
#include "../../src/graph/impls/ocl_v2/moe/moe_otd_windows_uma.hpp"
#include <iostream>

int main() {
#ifdef _WIN32
    using Microsoft::WRL::ComPtr;
    ComPtr<IDXGIFactory1> factory;
    if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory)))) return 1;
    unsigned found = 0;
    for (UINT index = 0;; ++index) {
        ComPtr<IDXGIAdapter1> adapter;
        const auto status = factory->EnumAdapters1(index, &adapter);
        if (status == DXGI_ERROR_NOT_FOUND) break;
        if (FAILED(status)) return 2;
        DXGI_ADAPTER_DESC1 desc{};
        if (FAILED(adapter->GetDesc1(&desc))) return 3;
        if (desc.VendorId != 0x8086 || (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE)) continue;
        const auto snapshot = ov::intel_gpu::ocl::moe_otd::sample_windows_uma(desc.AdapterLuid);
        if (!snapshot.system_total_bytes || !snapshot.wddm_budget_bytes) return 4;
        const auto free = snapshot.admission_limits(0);
        if (free.system_bytes > snapshot.system_total_bytes || free.gpu_budget_bytes > snapshot.wddm_budget_bytes) return 5;
        std::cout << "{\"luid_high\":" << desc.AdapterLuid.HighPart << ",\"luid_low\":" << desc.AdapterLuid.LowPart
                  << ",\"device_id\":" << desc.DeviceId << ",\"system_total_bytes\":" << snapshot.system_total_bytes
                  << ",\"system_available_bytes\":" << snapshot.system_available_bytes
                  << ",\"wddm_budget_bytes\":" << snapshot.wddm_budget_bytes
                  << ",\"wddm_process_usage_bytes\":" << snapshot.wddm_process_usage_bytes
                  << ",\"wddm_remaining_bytes\":" << free.gpu_budget_bytes << "}\n";
        ++found;
    }
    return found ? 0 : 6;
#else
    return 77;
#endif
}
