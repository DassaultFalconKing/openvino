# CLOUD-WAVE-01-STATE — 2026-10-22

**Evidence class:** CLOUD CODE AUTHORING with Windows host-only checks (MSVC 14.44 `/Zs` syntax checks, a standalone gtest harness without GPU, Python `unittest`). `ACCEPTED=NO`. No GPU, inference, benchmark or Linux run took place; operator-controlled GPU testing remains paused. No PR is merged; all three are Drafts.

## Source locks

- OpenVINO GPU base: `DassaultFalconKing/openvino` / `research/ov-gpu-otd-cohort-b5bfb148` @ `479219245f6f10b670367ad35af16513d72d2091` (remote ref re-checked after the last push).
- OVMS integration: `DassaultFalconKing/gemmamonster_model_server_OVMS` / `integration/qwm01-otd-pack-20261022` @ `746dc833dc3bcccf64fabdb469feda9260582bae`; unchanged.
- C6 golden: `c6-head-mainline-20261020` @ `ba6ec792c3121850fea167fa7d7e190b25fcdc76`; untouched.

## Status vocabulary

`STATIC_CHECK_PASS` = MSVC `/Zs` with the cohort build's exact compile flags, exit 0. `UNIT_PASS (standalone harness)` = the TU under test, its test file and gtest from `thirdparty/gtest`, built with MSVC and run on Windows, exit 0; it is **not** `ov_gpu_unit_tests`. `NOT_RUN` = possible here but not done. `BLOCKED_ENV` = no environment (Linux toolchain, GPU under operator pause).

## CLD-01 — pack integrity at the runtime read boundary

| field | value |
|---|---|
| task_id | CLD-01 |
| repo | DassaultFalconKing/openvino |
| base_sha | 479219245f6f10b670367ad35af16513d72d2091 |
| branch | feature/otd-pack-require-integrity-cloud-20261022 |
| commit_sha | code `2ed011fcf54dacc05061c36e1b27136dee09528b`, `48748f92` (follow-up); head is the commit that carries this file (see PR) |
| PR_URL | https://github.com/DassaultFalconKing/openvino/pull/1 (Draft) |
| changed_files | `moe_otd_pack.hpp/.cpp`, `moe_otd_runtime.hpp/.cpp`, `expert_weight_providers.hpp/.cpp`, `tests/unit/test_cases/moe_offload_pack_test.cpp`, `tests/unit/test_cases/moe_offload_weight_provider_test.cpp`, `docs/coordination/handoffs/CLD-01-HANDOFF.md`, this file |
| tests_run | `STATIC_CHECK_PASS` on 5 TUs (`moe_otd_pack.cpp`, `moe_otd_runtime.cpp`, `expert_weight_providers.cpp`, `moe_3gemm_swiglu_opt.cpp`, pack test TU); `moe_otd_pack.*` `UNIT_PASS (standalone harness)` 15/15, exit 0 |
| tests_not_run | provider tests `moe_expert_weight_provider.*` including the 4 new `require`/`use` GPU seam tests (need a GPU device: `BLOCKED_ENV`); `ov_gpu_unit_tests` build (`NOT_RUN`); Linux (`BLOCKED_ENV`); Arc 140V, NVMe and throughput (`BLOCKED_ENV`) |
| known_risks | FNV-1a is corruption detection, not authentication; per-miss O(payload) CPU checksum, estimated ~1.7 ms per 2 MiB expert against a ~0.3-0.4 ms read (unmeasured); identity between the configured pack and the lazily opened reader is not pinned; `use` mode with an already-open corrupt pack fails the request (it does not fall back), which needs an operator decision |
| handoff_path | docs/coordination/handoffs/CLD-01-HANDOFF.md |

## CLD-02 — atomic pack publication

| field | value |
|---|---|
| task_id | CLD-02 |
| repo | DassaultFalconKing/openvino |
| base_sha | 479219245f6f10b670367ad35af16513d72d2091 |
| branch | feature/otd-pack-atomic-publish-cloud-20261022 |
| commit_sha | code `65d4afca`, head `dbcf08c0ae90cd611172de18bb5c41f50127225e` |
| PR_URL | https://github.com/DassaultFalconKing/openvino/pull/2 (Draft) |
| changed_files | `moe_otd_pack.cpp`, `tests/unit/test_cases/moe_offload_pack_publish_test.cpp` (new), `tests/unit/sources.cmake` (one line), `docs/coordination/handoffs/CLD-02-HANDOFF.md` |
| tests_run | red run of the new suite on the base: 4 of 5 FAIL (the defects reproduce); green `moe_otd_pack_publish.*` `UNIT_PASS (standalone harness)` 5/5, exit 0, and `--gtest_repeat=50` exit 0; existing `moe_otd_pack.*` 11/11, exit 0; `STATIC_CHECK_PASS` for `moe_otd_pack.cpp` and the test TU |
| tests_not_run | Linux POSIX branch (`fsync`, `rename(2)`) `BLOCKED_ENV`; `ov_gpu_unit_tests` `NOT_RUN`; build-mode wall time `BLOCKED_ENV` |
| known_risks | Windows: replacement is refused while a reader holds the pack, and a concurrent open can briefly fail with `cannot open` (seen 2 times in one stress run, no retry added); stale `*.tmp-*` after a crash; temp-name check is check-then-open; reader identity pinning is a follow-up that must be based on CLD-01 |
| handoff_path | docs/coordination/handoffs/CLD-02-HANDOFF.md |

## CLD-03 — OTD trace integrity and offline accounting

| field | value |
|---|---|
| task_id | CLD-03 |
| repo | DassaultFalconKing/openvino |
| base_sha | 479219245f6f10b670367ad35af16513d72d2091 |
| branch | feature/otd-trace-accounting-cloud-20261022 |
| commit_sha | code `a973b0069aeafc52481240eb35534386c0781474`, head `4e8fa732151e1169aa58faa9a5732f2b021d5b16` |
| PR_URL | https://github.com/DassaultFalconKing/openvino/pull/3 (Draft) |
| changed_files | `moe_otd_trace.hpp/.cpp`, `tests/unit/test_cases/moe_offload_trace_test.cpp`, `tools/moe_otd_sim/otd_sim.py`, `tools/moe_otd_sim/test_otd_sim.py`, `tools/moe_otd_sim/README.md`, `docs/coordination/handoffs/CLD-03-HANDOFF.md` |
| tests_run | `python -m unittest test_otd_sim` `UNIT_PASS` 33/33, exit 0 (Python 3.14); red run against the base `otd_sim.py`: 14 of 16 new tests fail; base and branch give identical pre-existing output on a deterministic synthetic trace (exact LRU replay unchanged); `moe_otd_trace.*` `UNIT_PASS (standalone harness)` 13/13, exit 0; output of the real writer (close and destructor, LF and CRLF) verifies with 0 mismatches; `STATIC_CHECK_PASS` for `moe_otd_trace.cpp`, `expert_weight_providers.cpp` and the test TU |
| tests_not_run | Linux and older Python (`BLOCKED_ENV`/`NOT_RUN`); `ov_gpu_unit_tests` `NOT_RUN`; a traced real OTD run (`BLOCKED_ENV`, operator pause) |
| known_risks | an incomplete call lacks the miss that threw, so replay after it can diverge (reported as mismatches; fixing it needs the call site, a follow-up based on CLD-01); traces with an unterminated final line, accepted before, now need `--salvage`; physical NVMe bytes stay `NOT_EXPOSED` without a sidecar |
| handoff_path | docs/coordination/handoffs/CLD-03-HANDOFF.md |

## Combined trial merge (local only, never pushed)

CLD-01 head `07770ec1`, then CLD-02 head `dbcf08c0`, then CLD-03 head `4e8fa732`, merged onto the base: **no conflicts** (trial merge commit `fce99b55`, tree `4315dff7`, then deleted). On the merged tree:

- `STATIC_CHECK_PASS` for `moe_otd_trace.cpp`, `moe_otd_pack.cpp`, `moe_otd_runtime.cpp`, `expert_weight_providers.cpp` and `moe_3gemm_swiglu_opt.cpp`.
- `UNIT_PASS (standalone harness)`: `moe_otd_pack.*` 15/15, `moe_otd_pack_publish.*` 5/5 and `moe_otd_trace.*` 13/13.
- Python 33/33.

All exit codes were 0. Recommended merge order: CLD-01, then CLD-02 (rebase; textual overlap only in `moe_otd_pack.cpp`); CLD-03 is independent.

## Not done in this wave (by mandate)

No CLD-04 scheduler PR and no scheduler ADR was written in this session. No cache scheduler, layer-ahead prefetch, cross-slot single-flight, hot/cold policy, CPU expert lane, quantization, KV U4, parser/XGrammar, OVMS or C6 changes.

## Local QA sequence (in this order; each step gates the next)

1. **Review:** Reviewers 1 per handoff (C++ lifetime/failure/filesystem audit) for CLD-01, CLD-02, CLD-03, pinning the PR head SHAs above.
2. **Build:** Windows MSVC and Linux GCC/Clang build of the GPU plugin and `ov_gpu_unit_tests` from each branch and from the combined merge.
3. **Unit:** `--gtest_filter=moe_otd_pack.*:moe_otd_pack_publish.*:moe_otd_trace.*`, `--gtest_repeat=200` for `moe_otd_pack_publish.*`, `python -m unittest test_otd_sim`; repeat each handoff's red runs.
4. **GPU correctness** (operator authorization required): `moe_expert_weight_provider.*` on Arc 140V, including the CLD-01 `require`/`use` seam tests; corrupt-pack and held-open-pack manual checks; one traced short OTD run that must verify with `complete=true`.
5. **Benchmarks:** cost of the CLD-01 per-miss checksum (`pack_verify_ns` against read time), CLD-02 build-mode wall time, CLD-03 traced-run overhead, and physical NVMe bytes from an OS counter recorded in an `otd_run_meta v1` sidecar.
6. **Tuning:** only after the benchmarks, for example checksum strategy or the Windows `cannot open` retry decision, each as a separate reviewed change.
7. **Integration:** OVMS integration and any promotion only after steps 1-6 pass with receipts; never by automatic merge.
