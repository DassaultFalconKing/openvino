# P0-B Phases 4–5 — representative GEMM + short OVMS A/B — 2026-10-22

**Gate verdict:** `B_SAFE_FOR_MODEL_AB` (kept)  
**OVMS model slice:** `MODEL_PASS`  
**Not:** `B_PERF_REJECTED` · `B_BLOCKED` · merge of PR #8 / #6

`OTD_RESIDENT_USM_HOST` remains **default off** in code; path B enabled only via env on staged runtime.

Full local evidence: `C:\gm\orchestration\p0b-inflight-20261022\phase4-5\`.

## Pins

| Item | Value |
|---|---|
| Branch | `scratch/uma-02-host-harness-20261022` |
| Draft PR | https://github.com/DassaultFalconKing/openvino/pull/8 |
| Device | Intel(R) Arc(TM) 140V GPU (16GB) |
| `ov_gpu_unit_tests.exe` SHA256 | `511D941D077BF2AA07BA3893A4B2C2A6264DDA069E1477C51FB5546151A31FA3` |
| Scratch `openvino_intel_gpu_plugin.dll` SHA256 | `5207901E9096C86B4D55F2AE531F0B23C16C815100F6C6D75C1487C0861A3828` |
| Candidate GPU plugin (untouched) | `1093B69F6314898238FFEC06FB46631759526E89C90E3E3FD2FD29C122E8058E` |
| Staged `openvino.dll` (= candidate) | `13A3C5A91BD6AA45D29271E08613056F09F0F7DD067250C2BAC6B02C96597C8F` |
| Candidate `runtime\otd` | **UNTOUCHED** |
| Staged runtime | `C:\gm\otd-research\runtime\otd-p0b-phase45-20261022` |
| Fresh pack | `C:\gm\otd-research\packs\ornith-35b-ov-r20` — **40** `.pack` files (wiped + rebuilt; do not use prior partial/stale r20) |

## Phase 4 — ornith-scale GEMM/GEMV A/B

Filter: `moe_otd_gemm_ab.*` · `MOE_OTD_PERF_LOG=1` · flag default off except path-B cold env.

Record geometry matches live ornith pack header: **1 695 744 B (~1.617 MiB)** / expert.

### Cold miss (capacity=1, 8 iters)

| Path | cold p50 | cold p95 | copy p50 | pre-write / fence calls |
|---|---:|---:|---:|---:|
| A `usm_device` | 3 187 600 ns | 3 340 100 ns | 612 100 ns | 0 / 0 |
| B `usm_host` + flag | 2 514 900 ns | 2 675 400 ns | 93 000 ns | 8 / 8 |
| B/A | **0.79** | | **0.15** | |

### Quantized MoE GEMM M ∈ {1,2,4,8,11} (randomized order, i4 weights)

| M | A p50 ns | B p50 ns | B/A |
|---:|---:|---:|---:|
| 1 | 46500 | 53800 | 1.157 |
| 2 | 100700 | 52000 | 0.516 |
| 4 | 51900 | 50700 | 0.977 |
| 8 | 49800 | 49100 | 0.986 |
| 11 | 44300 | 49700 | 1.122 |
| **mean** | | | **0.871** |
| **worst** | | | **1.157** |

Steady B does **not** erase cold-copy savings → **not** `B_PERF_REJECTED`.

## Phase 5 — short OVMS C1/C2/C4 @ OFFLOAD_RATIO=20

Operator interrupt: wiped stale `ornith-35b-ov-r20` and rebuilt fresh on staged runtime before serve. Pack build reported **40** files (`ovms-build-r20-pathA-20261022-221512`). Serve A then B; kill `ovms` after each. max_tokens=48.

| Path | ready | warmup TTFT s | C1 | C2 | C4 | fatals |
|---|---|---:|---|---|---|---|
| A (flag `0`) | yes (~42 s) | ~8.95 | 1/1 (~0.59 s) | 2/2 (~1.15–1.43 s) | 4/4 (~1.77–1.92 s) | none |
| B (`OTD_RESIDENT_USM_HOST=1`) | yes (~283 s) | ~13.79 | 1/1 (~0.58 s) | 2/2 (~1.11–1.39 s) | 4/4 (~1.65–1.80 s) | none |

`MODEL_PASS`. No `GPU_CONTEXT_FATAL` / OOM / race symptoms on stop-scan.

## Governance

- No C6 / OVMS / driver source changes.
- No merge of PR #8 / #6; no soak; no production enablement.
