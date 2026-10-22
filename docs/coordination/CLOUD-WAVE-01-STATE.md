# CLOUD-WAVE-01-STATE — 2026-10-22

**Evidence class:** CLOUD CODE AUTHORING ONLY; `ACCEPTED=NO`; operator-controlled GPU testing remains paused.

## Source locks

- OpenVINO GPU base: `DassaultFalconKing/openvino` / `research/ov-gpu-otd-cohort-b5bfb148` @ `479219245f6f10b670367ad35af16513d72d2091`.
- OVMS integration: `DassaultFalconKing/gemmamonster_model_server_OVMS` / `integration/qwm01-otd-pack-20261022` @ `746dc833dc3bcccf64fabdb469feda9260582bae`; unchanged.
- C6 golden: `c6-head-mainline-20261020` @ `ba6ec792c3121850fea167fa7d7e190b25fcdc76`; untouched.

## CLD task state

| task_id | repo | base_sha | branch | code_commit_sha | PR_URL | status | handoff_path |
|---|---|---|---|---|---|---|---|
| CLD-01 | DassaultFalconKing/openvino | 479219245f6f10b670367ad35af16513d72d2091 | feature/otd-pack-require-integrity-cloud-20261022 | 2ed011fcf54dacc05061c36e1b27136dee09528b | https://github.com/DassaultFalconKing/openvino/pull/1 | IMPLEMENTED_NOT_RUN; draft | docs/coordination/handoffs/CLD-01-HANDOFF.md |
| CLD-02 | DassaultFalconKing/openvino | same pinned baseline, separate future branch | NOT_CREATED | NOT_CREATED | NOT_CREATED | NOT_STARTED | NOT_WRITTEN |
| CLD-03 | DassaultFalconKing/openvino | same pinned baseline, separate future branch | NOT_CREATED | NOT_CREATED | NOT_CREATED | NOT_STARTED | NOT_WRITTEN |

### CLD-01 changed files

- `src/plugins/intel_gpu/src/graph/impls/ocl_v2/moe/moe_otd_pack.hpp`
- `src/plugins/intel_gpu/src/graph/impls/ocl_v2/moe/moe_otd_pack.cpp`
- `src/plugins/intel_gpu/src/graph/impls/ocl_v2/moe/moe_otd_runtime.cpp`
- `src/plugins/intel_gpu/tests/unit/test_cases/moe_offload_pack_test.cpp`
- `docs/coordination/handoffs/CLD-01-HANDOFF.md` (documentation)
- `docs/coordination/CLOUD-WAVE-01-STATE.md` (this document)

**Tests actually run:** no compiler, OpenVINO unit binary or GPU environment was available. Source anchor assertions and GitHub tree/ref checks were performed while publishing. **Not** a `STATIC_CHECK_PASS` compiler result.

**Tests not run:** Windows and Linux C++ builds, `ov_gpu_unit_tests --gtest_filter=moe_otd_pack.*:moe_expert_weight_provider.*`, true GPU mutation/require mode, Arc 140V, NVMe performance, MTP K3, tool loops and physical NVMe measurement.

**Known risks:** FNV is not authentication; per-miss O(payload) CPU checksum; open→read file replacement race not solved by CLD-01; checksum time not represented in current read/copy counters; provider `use` policy for runtime-corrupt (already-open) file requires explicit local decision; C++ compiler/ABI has not been checked.

## Existing vs gaps

| Already existing | Correctness gap treated in CLD-01 |
|---|---|
| Native OpenVINO GPU OTD and 4096-aligned pack v1 | Mandatory-pack metadata canonicality/overflow hardening |
| Pack checksum table and standalone `verify_pack()` | Per-expert checksum **at runtime read boundary** |
| Resident LRU, one-read/miss staging | Reject damaged staging before any device copy |
| Pack unit tests and C++ runtime | Targeted negative tests of the actual shared read helper |

## Next queue: independent local work

1. Reviewer 1: strict C++ lifetime, failure and security audit before touching hardware.
2. Reviewer 2: pin branch SHA and build OpenVINO plugin + focused unit target.
3. Reviewer 3: run all positive/negative/placement/checksum tests, add provider `require/use` seam tests.
4. Reviewer 4: **only with operator's authorization**, Arc 140V correctness, physical NVMe and throughput baselines.
5. Only then decide whether to revise CLD-01; proceed to separate CLD-02 atomic publication and CLD-03 trace accounting PRs. Never merge or promote without evidence.

No scheduler, hot/cold expert policy, speculative layer-ahead prefetch, CPU lane, quantization, U4 or C6 runtime changes are authorized.
