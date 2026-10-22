# CLD-01-HANDOFF — Expert Pack Integrity (Cloud Wave 01)

**Status:** `IMPLEMENTED_NOT_RUN`; Draft PR https://github.com/DassaultFalconKing/openvino/pull/1. This cloud-only delivery is **not** GPU acceptance, OpenVINO build certification or permission to merge.

## 1. Source locks / identity

- Repository: `DassaultFalconKing/openvino`; measured original branch `research/ov-gpu-otd-cohort-b5bfb148`.
- Original pinned parent: `479219245f6f10b670367ad35af16513d72d2091`.
- Implementation branch: `feature/otd-pack-require-integrity-cloud-20261022`; code commit: `2ed011fcf54dacc05061c36e1b27136dee09528b`.
- OVMS fork `DassaultFalconKing/gemmamonster_model_server_OVMS` integration `integration/qwm01-otd-pack-20261022` @ `746dc833dc3bcccf64fabdb469feda9260582bae` (not modified).
- Golden `c6-head-mainline-20261020` @ `ba6ec792c3121850fea167fa7d7e190b25fcdc76` (not modified).
- Other GPU observability branch `research/ov-gpu-otd-observability-2026` not mixed in.
- Source changes: `src/plugins/intel_gpu/src/graph/impls/ocl_v2/moe/moe_otd_pack.hpp`, `.cpp`, `moe_otd_runtime.hpp`, `moe_otd_runtime.cpp`, `expert_weight_providers.hpp`, `expert_weight_providers.cpp`; unit fixtures `src/plugins/intel_gpu/tests/unit/test_cases/moe_offload_pack_test.cpp`, `moe_offload_weight_provider_test.cpp`.
- Commits: `2ed011fc` (initial integrity patch), `8e1bb766` (docs), then a follow-up hardening commit (pre-copy validation, `PackIntegrityError`, verify counters, provider-level tests); read the exact follow-up SHA from `git log` on the PR branch.
- Final documentation commit/head: inspect PR branch and `git rev-parse HEAD` (the handoff itself must not self-reference a future commit).

## 2. What changed / algorithm / non-goals

- Existing v1 `PackLayout::open` now checks the entire packed header against `PackLayout::plan(expected)`, including canonical offsets, tensor spans, stride, file size, plus checked uint64 size arithmetic.
- `open` reads an **expert-ID-indexed** 64-bit FNV checksum table into immutable layout metadata. The v1 on-disk format is unchanged; even when physical records are permuted, `checksums[expert_id]` remains the key.
- `read_verified_record` performs **one** positional read directly into the reusable staging vector, hashes exact unpadded payload bytes, throws on mismatch and only then returns. `fill_weights_from_pack` invokes this helper **before** the first GPU `copy_from`.
- The former `verify_pack` utility reuses the common record checksum method; it is not the runtime safety test.
- No replacement of packer, router, LRU, execution policy, GPU memory placement, quant, MTP, U4, parser, XGrammar or unrelated runtime logic.
- `PackMode::require`: missing/invalid metadata fails at configure; payload/checksum corruption on miss raises an exception, with no weight-file fallback or successful slot fill.
- `PackMode::use`: existing startup fallback for an absent/unopenable pack is unchanged. A pack that opens but later fails a read intentionally throws rather than silently switching sources. Confirm product policy locally before integrating.

### Follow-up hardening commit (on top of `8e1bb766`)

- **No partial slot mutation.** `fill_weights_from_pack` previously asserted each tensor's resident size *inside* the copy loop, so a mismatch on tensor `t` could leave tensors `0..t-1` of the slot overwritten with the new expert while the rest still held the evicted one. All size and slot-range checks now run before the read; `copy_from` runs only after every check and the checksum have passed.
- **`PackIntegrityError`** (derives from `std::runtime_error`) is thrown by `read_verified_record` on mismatch, so integrity rejections are distinguishable from I/O errors. The message carries expert ID, physical record position and payload size only.
- **Provider diagnostic.** `OffloadExpertWeightProvider::acquire_slot` rethrows it as `ov::Exception` with the content-keyed pack *file name* (no directory, no machine path), expert and slot, and states that resident memory was not modified. The LRU slot stays `filled=false`, so a retry re-reads and fails again; it is never served as a stale hit.
- **Checksum cost is now counted** under `MOE_OTD_PERF_LOG`: new `pack_verify_ns`, `pack_verified_records`, `pack_verify_failures` in `OtdPerfCounters`/`OtdPerfSnapshot`, printed as a third `[OTD_PERF]` line. `ExpertLoadStats` and the `MOE_OTD_TRACE` format are unchanged (CLD-03 owns trace).
- **Provider-level tests** in `moe_offload_weight_provider_test.cpp` drive the real `configure_pack -> acquire_one -> fill_weights_from_pack` path with a tiny 4-expert layer (u8 weights, f16 scales, u8 zero points) and real device buffers pre-filled with a sentinel.

### Per-miss checksum cost (estimate, not measured)

FNV-1a-64 is a serial xor+multiply chain, roughly 4 cycles/byte, i.e. about 0.8 ns/byte (~1.2 GB/s) on one ~4.8 GHz core. For a record of `P` payload bytes the added CPU time per miss is about `0.8 ns * P`: ~1.7 ms for a 2 MiB expert, against ~0.3-0.4 ms to read the same 2 MiB sequentially from a fast NVMe drive. **On a miss-heavy workload the checksum can cost several times the disk read.** It runs on the calling thread, adds no heap allocation (it hashes the existing staging buffer) and no extra I/O. Startup cost is O(num_expert) for the checksum table (8 bytes per expert). Reviewer 4 must measure `avg_pack_verify_us` against `avg_disk_io_us`. If the cost is unacceptable, options needing a separate decision are: a v2 pack with a hardware-accelerated checksum (CRC32C) or a vectorized hash, or verify-once-per-process (weaker: does not catch bytes that change between reads). Neither is implemented here.

## 3. Threat model / invariants / limitations

1. Detect accidental bit flips and metadata corruption, including re-sealed non-canonical header offsets and wrong placement index. FNV-1a-64 **is not authenticated / cryptographically collision-resistant**. A malicious actor able to rewrite content and checksums together is outside the integrity guarantee.
2. The in-memory checksum table is read at configure, not all payload data. Required mode can accept a corrupt record at startup but *must reject it on its first miss before the GPU is modified*. Startup O(experts), per-miss O(payload) checksum.
3. Exactly one file-reader invocation per miss, no new disk payload read, no new model tier or hot-path heap allocation other than possible existing staging resize.
4. `LRUCache::lookup` reserves an unfilled slot before read; `set_filled` is only reached after successful verification and GPU copying. On exception do not resume using stale slot. Independently review repeated failure/retry semantics, including concurrent requests.
5. No lifetime pin for the pack between `open` and `ParallelWeightReader`; concurrent replacement/mutation, reader handle semantics and file swap are separate CLD-02 questions.
6. FNV CPU time is excluded from the read and GPU-copy timings and reported separately as `pack_verify_ns` (only with `MOE_OTD_PERF_LOG`). It is still not in `MOE_OTD_TRACE`. Compare wall time and CPU use; never call the checksum free.
9. Thread safety is unchanged: the provider's single `_pack_staging` buffer already implies that one provider (one MoE layer) is not filled concurrently. Perf counters are relaxed atomics.
7. Planner assumes the source model specification is trusted; very large sparse files combined with attacker-controlled spec can still cause memory pressure. Header-controlled count is checked against expected source identity before allocation.
8. Size and offsets are bounded by `std::streamoff` representability. Original v1 records and tensor bytes remain byte-compatible.

## 4. Local build/unit verification

### Evidence actually collected by the author

`STATIC_CHECK_PASS` (syntax only, MSVC 14.44.35207 `/Zs`, Windows host, no objects, no link, no GPU): each changed TU was compiled with the exact flags of an existing cohort build's compile database, with the source path swapped to this branch. The include trace confirmed the branch's `moe_otd_pack.hpp` was used.

| TU | exit |
|---|---|
| `moe_otd_pack.cpp` | 0 |
| `moe_otd_runtime.cpp` | 0 |
| `expert_weight_providers.cpp` | 0 |
| `tests/unit/test_cases/moe_offload_pack_test.cpp` | 0 |
| `tests/unit/test_cases/moe_offload_weight_provider_test.cpp` | 0 |

The two test TUs used an approximated include set (gtest, gmock, `tests/unit`, graph include dirs), because unit tests were not configured in that build. Their only warnings (C4244 in `test_utils.h:613` and `xutility`) also appear on the unmodified base file. This is **not** `BUILD PASS`: no linking, template instantiation in other TUs, GCC/Clang or Linux checks. `UNIT` tests: `NOT_RUN`.

### Remaining local verification (NOT RUN by the author)

Check out this **specific PR branch**; verify `git status --short` clean, expected source locks and `git diff 479219245f6f10b670367ad35af16513d72d2091...HEAD --stat`.

### Windows 11, Arc 140V development host, PowerShell (no machine reboot)

```powershell
git fetch origin feature/otd-pack-require-integrity-cloud-20261022
git switch --detach origin/feature/otd-pack-require-integrity-cloud-20261022
git status --short
git rev-parse HEAD
cmake -S . -B build-cld01 -G Ninja -DENABLE_TESTS=ON -DENABLE_INTEL_GPU=ON -DGPU_RT_TYPE=OCL
if ($LASTEXITCODE -ne 0) { throw "cmake configure failed" }
cmake --build build-cld01 --target ov_gpu_unit_tests --parallel 8
if ($LASTEXITCODE -ne 0) { throw "ov_gpu_unit_tests build failed" }
$exe = Get-ChildItem build-cld01 -Recurse -Filter ov_gpu_unit_tests.exe | Select-Object -First 1 -ExpandProperty FullName
if (-not $exe) { throw "missing ov_gpu_unit_tests.exe" }
& $exe '--gtest_filter=moe_otd_pack.*:moe_expert_weight_provider.*'
if ($LASTEXITCODE -ne 0) { throw "focused unit tests failed" }
```

**Note:** CMake options are a suggested fresh OCL/Ninja recipe, **not validated against this machine**. Reuse the exact working local OpenVINO/OVMS build configuration if it requires additional environment or dependencies. This repo's `src/plugins/intel_gpu/tests/unit/CMakeLists.txt` names the actual target `ov_gpu_unit_tests`.

Capture compiler identity, CMake cache/build settings, full test log, exit code, source commit SHA, `Get-FileHash` of the tested GPU-plugin DLL. Verify runtime actually loads that DLL, not a previously installed plugin. Do not modify global drivers, model weights, C6, or production paths.

### Linux/P720 source and CPU-only fixture parity

```bash
git fetch origin feature/otd-pack-require-integrity-cloud-20261022
git switch --detach origin/feature/otd-pack-require-integrity-cloud-20261022
git status --short && git rev-parse HEAD
cmake -S . -B build-cld01 -G Ninja -DENABLE_TESTS=ON -DENABLE_INTEL_GPU=ON -DGPU_RT_TYPE=OCL
cmake --build build-cld01 --target ov_gpu_unit_tests -j 8
EXE="$(find build-cld01 -type f -name ov_gpu_unit_tests -perm -111 | head -n 1)"
test -n "$EXE" && "$EXE" --gtest_filter='moe_otd_pack.*:moe_expert_weight_provider.*'
```

Linux/P720 can check C++ and filesystem portability only; **do not** claim equivalence to Intel Arc GPU inference. Check C++17 compiler, case-sensitive paths, failure propagation, file descriptor behavior and concurrent access.

## 5. Unit/negative test queue

- Baseline tests: `parse_mode`, `build_open_and_records_round_trip`, `corrupt_record_is_detected_and_not_reused`, `corrupt_header_or_truncation_fails_open_and_triggers_rebuild`, placement permutation and missing/cross-model pack.
- Added tests: `serving_read_validates_logical_expert_before_gpu_copy` (shared read helper, first/last expert, rearranged placement, one reader call); `serving_read_rejects_corrupt_checksum_and_truncated_record`; `resealed_noncanonical_offsets_and_sizes_are_rejected`; `invalid_placement_table_and_overflowed_plan_fail_closed`.
- Added provider tests (`moe_expert_weight_provider.*`, need a GPU device for `get_test_stream()` and device buffers):
  - `pack_require_fails_closed_on_unusable_pack_and_use_falls_back`: missing pack, corrupted header, wrong directory under `require` throw `ov::Exception`; missing pack under `use` falls back (`uses_pack()==false`).
  - `pack_require_serves_verified_record_into_slot`: a valid record lands byte-exact in its slot; every other byte of every resident buffer keeps the sentinel.
  - `pack_require_rejects_corrupt_record_without_touching_device`: a flipped byte in the **last** tensor (`down_z`) of expert 1 throws, all nine resident buffers remain entirely sentinel, the slot is not marked filled, a retry throws again, and expert 0 still serves.
  - `pack_use_mode_does_not_silently_serve_corrupt_open_pack`: documents the chosen `use` policy (no silent source switch once a pack is open).
- Still desired locally: parallel inference error propagation, cross-model checkpoint source key through the provider, and a runtime trace of a failure with zero successful slot fill.
- Repeat on Windows and Linux; record `PASS/FAIL/SKIP` per case, never infer from source inspection.

## 6. GPU correctness / performance after operator authorizes compute

1. Pin model/IR/quant, pack SHA, exact source, driver, GenAI/OVMS/GPU-plugin DLL SHA, batch context and U4 setting.
2. Validate `GPU_MOE_OTD_PACK_MODE=off` vs `require` on the same model and deterministic fixed prompts; inspect numerical output/toolcall drift. Do not change router or top-k.
3. Compare unpacked and packed `OFFLOAD_RATIO=40` and `75`, including 55/384 existing smoke reference and separate 2048-token input MTP K3 **only when approved**; warmup and measured repetitions.
4. Record p50/p95/p99 TPOT, TTFT, committed-token throughput, allocations, UMA pressure, GPU residency, OS swap, page cache, wall time, CPU checksum overhead and plugin logical reads/bytes.
5. **Physical NVMe read bytes must come from OS device-level instrumentation**, not the plugin's logical `disk_read_bytes`. Current physical attribution / speculative acceptance are `NOT_EXPOSED`.
6. Stop on performance regression beyond review budget, GPU resource errors, changing output semantics, or integrity/fallback violations. Do not promote to integration without human decision.

## 7. Independent reviewer assignments / local-agent bootstrap

**Reviewer 1 — C++ safety:** inspect `PackLayout::plan/open` overflow and canonicality, FNV coverage, pointer/size/placement invariants, stream bounds, pack reader and LRU exception safety; propose minimal fixes directly to PR.

**Reviewer 2 — build/ABI:** independently compile Windows MSVC and Linux GCC/Clang in correct OpenVINO configuration, check includes, C++ dialect, symbols, GPU DLL identity and avoid rebuilding unrelated C6.

**Reviewer 3 — unit negatives:** execute focused GTest suite including new shared read boundary, concurrency and failure-injection additions where safe; add missing provider-mode tests. Record receipts.

**Reviewer 4 — Arc performance:** only when explicitly authorized, measure r40/r75 and compare to original pinned baseline, wall-time checksum overhead, OS physical NVMe bytes, p95/p99, 32-GiB UMA pressure and output correctness.

**LOCAL-AGENT-BOOTSTRAP:** Your assignment is one reviewer role above. Fetch only Draft PR #1 and this file; independently read the diff and original SHA instead of trusting the cloud author's conclusions. Run only the commands for your role and hardware. Report exact commit, command, exit code, files/lines, reproducible negative fixture and any objection. Do not inherit an author's verdict, auto-merge, mutate C6, or start GPU benchmarks while the operator's compute pause remains.

## 8. STOP, rollback and release gate

- Stop immediately on integrity violation, silent fallback under `require`, incomplete GPU writes, race-induced hit of stale slot, compiler failure or large throughput regression.
- Roll back at source level by building pinned parent `479219245f6f10b670367ad35af16513d72d2091`; for A/B disable packs with `GPU_MOE_OTD_PACK_MODE=off` and restart a test service in the controlled environment only. Never silently downgrade a `require` production service.
- Source PR is **DRAFT**, not integrated. Order: C++ review → exact build → targeted unit and corruption tests → GPU correctness → authorized benchmarks → integration decision.
