# CIB2 N2 / N3 / N4: GPU residency safety primitives

Source-only controls on the **target OpenVINO 2026.4 dev branch**, NOT
the 2026.5 donor. Files live under `src/plugins/intel_gpu/src/graph/impls/ocl_v2/moe/`.

- `moe_otd_slot_lease_v2.hpp`: generation-tagged GPU reader leases;
  a slot remains unavailable until an owned nonblocking GPU completion
  callback reports completion. No event => no release. No stale hit.
- `moe_otd_uma_prefetch_v2.hpp`: admission for overlapping CPU/WDDM
  budget limits (never sum them) plus bounded, deterministic next-hot
  predictions from **measured** statistics.
- `test_control.cpp`: a standalone host regression harness without
  OpenVINO runtime/GPU dependencies.

## Host invocation

The controls, actual native pack codec/publication tests, and LRU regressions
can also be built independently of the full GPU runtime:

```bat
cmake -S src/plugins/intel_gpu/tools/moe_otd_control_v2 -B build/otd-controls -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build/otd-controls --parallel 2
ctest --test-dir build/otd-controls -V
```

This requires the pinned `thirdparty/gtest/gtest` submodule. It runs host code,
not GPU kernels. On Windows 2026-10-24, MSVC 14.44.35207 built all three
targets with `/std:c++17 /W4 /WX`: control harness PASS, three LRU scenarios
PASS, and 24 pack codec/publication tests PASS. The independent Python
verifier passed 14 tests. Full plugin and `ov_gpu_unit_tests` remain BUILDING.

The initial N1 port now connects per-layer pack specs, placement/residency
sidecars, checked packed reads and tracing to the 2026.4 provider and graph
factory, including compiled-cache loading. Only the source donor's runtime,
provider and LRU pieces were ported; the target's GEMM kernels and dispatch
paths remain unchanged. The simultaneous working set is protected from
priority-based self-eviction during acquisition. This protection is not a
GPU event lease. N2/N3/N4 native runtime gates remain open; do not enable
speculative loading or accept packed GPU serving from these host results.

### Conservative N2 execution fallback

`moe_otd_execution_guard.hpp` serializes executions sharing an offloaded
provider. The graph drains both oneDNN and OCL before releasing that lock,
including early returns and exception/cancellation unwinding. The native
adapter independently attempts both drains. If either fails, it emits
`OTD_FATAL_UNVERIFIED_GPU_COMPLETION` and exits the process with code 86,
without resource teardown. Poison alone cannot retain the request and engine
owners during failed-drain unwind. Context-level quarantine/recovery is not
implemented; the fail-stop policy deliberately sacrifices all active requests
on an unverified GPU completion. Successful drains preserve cancellation. Scratch
storage is declared before the guard so the drain precedes scratch destruction.
The r0 path bypasses this gate. Resident arena memory owners are retained in
the provider, and rebinding the cache to a different arena fails closed.

The host execution-guard test exercises four concurrent request threads (400
serialized sections), cancellation, drain failure, poisoned reuse, failure
while unwinding, and r0 bypass. This is a synchronous fallback, **not a
completed per-slot GPU event lease integration**. Device-reset teardown,
actual consumer-event ownership, clone arena eligibility and live concurrency
must still be validated before serving. Prefetch remains disabled.

Four additional child-process fault-injection cases verify independent drain
attempts, exit code 86 after either/both failures, and successful completion.
The generic host gate still tests poison/unwind semantics; the actual native
adapter uses the stronger fail-stop policy above.

### N3 Windows probe adapter

`moe_otd_windows_uma.hpp` samples GlobalMemoryStatusEx and the exact supplied
adapter LUID's DXGI/WDDM budget/current process usage. Probe failures throw;
there is no guessed adapter or additive CPU+GPU free-memory budget. The
`otd_windows_uma_probe` target enumerates Intel adapters for inventory only.
It was compiled with MSVC C++17 /W4 /WX and run on the local Arc 140V. The
probe reports overlapping physical availability and WDDM remaining budget
separately. This is **not** global committed/pending reservation accounting:
the provider, KV cache and workspace allocations are not yet enrolled in one
cross-layer controller. Multiple LUID observations must never be summed.

Windows Developer Command Prompt:

```bat
cl /nologo /std:c++17 /EHsc /W4 /WX /O2 test_control.cpp /Fe:control_test.exe
control_test.exe
```

Linux/GCC:

```sh
g++ -std=c++17 -pthread -Wall -Wextra -Werror -O2 test_control.cpp -o control_test
./control_test
```

This branch's exact source has not been compiled by the cloud agent.
An independent local C++17 prototype was compiled with GCC 14.2 and
its synthetic N2/N3/N4 host assertions passed. DO NOT attach that
result as an execution receipt for this GitHub commit.

## Required live adapter (not yet implemented)

The existing `ExpertSlotLease` is a vector of slot indices, **not**
a GPU event lifetime lease. The existing `moe_3gemm_swiglu_opt` uses
synchronous barriers at multiple call sites.

The native engineer must connect:

1. GPU provider owns `GpuSlotLeaseLedger` per resident arena;
   `begin_write` BEFORE selecting an evicted physical slot;
   reject reusing any pinned/currently-copying slot.
2. A checked packed read and device copy produce a real
   `cldnn::event::ptr`. Call `submit_write(ticket, [event] {
   return event->is_set(); })` with **strong event ownership**.
   Do not advertise a resident hit until that event completes.
3. Each grouped/single expert consumer calls `begin_read`, owns
   the exact generation and attaches its terminal GPU **consumer**
   event via `retire_read`. The producer-copy event is NOT enough.
4. If a model execution is cancelled or errors, NEVER discard a
   pending consumer lease unless the device event reports complete
   or a full stream/device drain is proven. Stale generations fail.
5. A global UMA controller must sample Windows actual available
   memory and WDDM local GPU budget, reserve OS/KV/workspace safety,
   and enforce per-model/per-layer admissions. This offline ledger
   only models new incremental reservations, not actual resident
   allocations or asynchronous reconciliation.
6. Prefetch suggestions must pass (2)-(5) before any async I/O.
   Demand outranks prefetch; pending suggestions may be dropped
   on memory pressure. Prove benefit with profile-on/off A/B.

## Mandatory native gates

- N1: port Pack V1/V2 reader, writer and tests from donor source PR #9
  onto exact 2026.4 release target; compile GPU plugin and OVMS as one
  exact binary tuple, no 2026.5 DLLs.
- N2: cancellation, multi-reader, grouped top-k and async copy adversarial
  tests; event identity and lease lifetime across cloned networks.
- N3: actual OS/WDDM memory sampling, committed/released budget tracking,
  OOM rollback, and multi-layer allocation pressure.
- N4: real routing profiling -> bounded prearm -> device copy -> terminal
  event -> same-logits parity; disabled-mode LRU performance baseline.

No FULL V2 GPU acceptance, physical read benefit, or production PASS
from header-only host simulation. Path A `usm_device` only. Path B OFF.


## CIB2 N2/N3 safety incremental patch (2026-10-24)

Review branch `dev/cib2-n2n3-safety-ledger-20261024`, based on the **2026.4** target PR #10.

- `LRUCache::set_recycle_guard` is a callback-driven **veto before eviction**.
  Wire only to an owner that knows each slot's actual *terminal consumer*
  GPU event, across all queues. A producer/copy event is insufficient.
  If none is available, keep the current synchronous drain path and do not
  arm async prefetch. No callback keeps the old synchronous LRU behaviour.
- Guard invoked under LRU mutex; callback must not re-enter LRU, wait on
  cache-owned locks, or throw. Veto all slots => demand fails closed **without**
  changing the expert mapping, and local code must handle backpressure.
- `UmaAdmissionLedger::reserve_batch` is a single-lock, all-or-none CPU
  staging + GPU-visible memory reservation. Both classes contribute to
  shared system RAM; GPU-visible allocations also count against **overlapping**
  exact-adapter WDDM budget. The reserved amounts here represent *pending*
  allocations only, not verified process-wide committed usage.
- Host regression assertions: `otd_lru_test` protects active slot, tests
  all-guarded stop and progress after release; `otd_control_test` checks
  atomic combined admission, rejection, and ticket release.
- Limitations: real GPU event ownership, global OS/WDDM committed-vs-pending
  reconciliation, PGP dispatcher, Pack V2 shards, and live GPU correctness
  remain NOT_DONE/NOT_RUN. Path A only; Path B/prefetch OFF.

For host controls (no Arc GPU acceptance implied):

```powershell
cmake -S src/plugins/intel_gpu/tools/moe_otd_control_v2 -B C:\temp\cib2-n2n3-test
cmake --build C:\temp\cib2-n2n3-test --config Release
ctest --test-dir C:\temp\cib2-n2n3-test -C Release --output-on-failure
```
