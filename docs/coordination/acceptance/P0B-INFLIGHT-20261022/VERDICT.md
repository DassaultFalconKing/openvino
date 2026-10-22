# P0-B verdict — 2026-10-22

**Verdict:** `B_SAFE_FOR_MODEL_AB`

Path-B in-flight ownership + post-write visibility gates **PASS** on Arc 140V. Ready for **authorized** representative GEMM / OVMS model A/B. Do **not** merge PR #8 / #6. `OTD_RESIDENT_USM_HOST` remains default off.

## Pins

| Item | Value |
|---|---|
| Branch | `scratch/uma-02-host-harness-20261022` |
| Draft PR | https://github.com/DassaultFalconKing/openvino/pull/8 |
| Code SHA (this verdict) | `13d56464c2af9e40b5bd27cbabe99d58784bfc6a` |
| Fix commit | `ed246cdf0252185fae6f16c6391afa63ec36404c` |
| `ov_gpu_unit_tests.exe` SHA256 | `E35E432C3879E903534FE74D41B83155AB04AD7AA974957D4B919024E2C1A8F7` |
| Device | Intel(R) Arc(TM) 140V GPU |

## Root cause (resolved)

Post-scatter `flush`+`finish` alone does **not** prove prior GPU readers of a recycled `usm_host` slot completed before CPU memcpy overwrite. Fix: pre-write `finish()` on fill stream (+ registered secondary reader streams) **before** scatter; keep post-write fence before `set_filled`.

## Phase 3 results

| ID | Result |
|---|---|
| P0B-O1 same-queue overlap | **PASS** |
| P0B-O2 two-queue overlap | **PASS** |
| P0B-V1 visibility before `set_filled` | **PASS** (with fence); without post-fence NEO still matched host dump on this host — fence retained |
| P0B-L1 no false hit + A/B reuse parity | **PASS** |
| `moe_otd_uma02.*` regression | **9 PASS / 1 SKIP** (U2-G02 needs `MOE_OTD_PERF_LOG=1`) |
| Sweet-spot `OffloadSweetSpots*:moe_otd_sweetspot.*` | **10/10 PASS** |

## Phases 4–5

**Not entered this session** — plan requires Phase 3 PASS **and** operator confirmation for real-size GEMM then OVMS C1/C2/C4. Status: `NOT_RUN` (authorized next).

## Evidence

- Repo docs: this directory  
- Local: `C:\gm\orchestration\p0b-inflight-20261022\evidence\`
