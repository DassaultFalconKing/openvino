// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
// Independent C++17 N2/N3/N4 host test; not a native GPU acceptance test.
#include "../../src/graph/impls/ocl_v2/moe/moe_otd_slot_lease_v2.hpp"
#include "../../src/graph/impls/ocl_v2/moe/moe_otd_uma_prefetch_v2.hpp"
#include <atomic>
#include <cassert>
#include <iostream>
#include <stdexcept>

using namespace ov::intel_gpu::ocl::moe_otd;
int main() {
    GpuSlotLeaseLedger leases(2);
    const auto w = leases.begin_write(0);
    assert(w);
    std::atomic<bool> copied{false};
    leases.submit_write(*w, [&] { return copied.load(); });
    assert(!leases.resident(0, w->generation));
    assert(!leases.can_evict(0));
    assert(!leases.begin_read(0, w->generation));  // GPU copy still in flight
    copied.store(true);
    assert(leases.resident(0, w->generation));
    const auto r = leases.begin_read(0, w->generation);
    assert(r);
    assert(!leases.begin_read(0, w->generation + 1));  // stale generation
    assert(!leases.begin_write(0));  // active GPU reader prevents overwrite
    std::atomic<bool> used{false};
    leases.retire_read(*r, [&] { return used.load(); });
    assert(!leases.can_evict(0));
    assert(!leases.begin_write(0));
    used.store(true);
    assert(leases.can_evict(0));
    const auto w2 = leases.begin_write(0);
    assert(w2 && w2->generation > w->generation);
    assert(!leases.resident(0, w->generation));
    leases.cancel_unsubmitted_write(*w2);
    assert(!leases.resident(0, w2->generation));
    bool bad = false;
    try { leases.cancel_unsubmitted_write(*w2); }
    catch (const std::logic_error&) { bad = true; }
    assert(bad);
    const auto w3 = leases.begin_write(0);
    assert(w3);
    leases.submit_write(*w3, []() -> bool { throw std::runtime_error("GPU event failure"); });
    assert(!leases.resident(0, w3->generation));
    assert(!leases.can_evict(0));

    UmaAdmissionLedger budget({1000, 650, 100});
    auto kv = budget.reserve(UmaAdmissionLedger::Class::kv, 400, true);
    assert(kv);
    assert(!budget.reserve(UmaAdmissionLedger::Class::weights, 300, true));
    auto staging = budget.reserve(UmaAdmissionLedger::Class::staging, 200, false);
    assert(staging);
    assert(budget.system_reserved() == 600);
    bad = false;
    try { budget.reset({2000, 2000, 100}); }
    catch (const std::logic_error&) { bad = true; }
    assert(bad);
    budget.release(*staging);
    budget.release(*kv);
    // N3: same exact-LUID ledger is shared across model/layer owners.
    auto global_a = GlobalUmaAdmission::for_adapter("00000000000000a1", {1000, 600, 100});
    auto global_b = GlobalUmaAdmission::for_adapter("00000000000000a1", {9000, 9000, 0});
    auto other_gpu = GlobalUmaAdmission::for_adapter("00000000000000a2", {1000, 600, 100});
    assert(global_a == global_b && global_a != other_gpu);
    const std::vector<UmaAdmissionLedger::Demand> cohort = {
        {UmaAdmissionLedger::Class::weights, 250, true},
        {UmaAdmissionLedger::Class::kv, 200, true},
        {UmaAdmissionLedger::Class::staging, 100, false}
    };
    const auto admitted = global_a->reserve_batch(cohort);
    assert(admitted && admitted->size() == 3);
    assert(global_b->system_reserved() == 550 && global_b->gpu_reserved() == 450);
    assert(!global_b->reserve(UmaAdmissionLedger::Class::workspace, 200, true));
    assert(!global_b->reserve_batch({{UmaAdmissionLedger::Class::workspace, 100, true},
                                     {UmaAdmissionLedger::Class::prefetch, 200, true}}));
    assert(global_b->system_reserved() == 550 && global_b->gpu_reserved() == 450);
    // A fresh low WDDM budget must tighten admission, never grant stale capacity.
    global_b->observe({500, 460, 100});
    assert(!global_a->reserve(UmaAdmissionLedger::Class::workspace, 1, true));
    for (auto ticket : *admitted) global_a->release(ticket);
    global_a->reset({1000, 600, 100});
    assert(global_b->system_reserved() == 0 && global_b->gpu_reserved() == 0);
    bad = false;
    try { GlobalUmaAdmission::for_adapter("wrong-luid", {1000, 600, 100}); }
    catch (const std::invalid_argument&) { bad = true; }
    assert(bad);

    bad = false;
    try { budget.release(*kv); }
    catch (const std::logic_error&) { bad = true; }
    assert(bad);

    const std::vector<PredictedExpert> input = {
        {0, 4, 0.95, 800, 500, 450, 50, 128, false, false},
        {1, 8, 0.90, 600, 500, 400, 30, 256, false, false},
        {0, 3, 0.05, 500, 900, 100, 0, 128, false, false},
        {0, 4, 0.95, 800, 500, 450, 50, 128, false, false},
        {1, 2, 1.0, 999, 100, 95, 0, 1, true, false},
        {2, 1, 1.0, 1000, 50, 50, 0, 1, false, true}
    };
    const auto p = plan_prefetch(input, 2, 384);
    assert(p.size() == 2);
    assert(p[0].layer == 0 && p[0].expert == 4);
    assert(p[1].layer == 1 && p[1].expert == 8);
    assert(plan_prefetch(input, 2, 127).empty());
    assert(plan_prefetch(input, 0, 384).empty());
    std::cout << "CIB2 N2/N3/N4 host model: PASS\\n";
}
