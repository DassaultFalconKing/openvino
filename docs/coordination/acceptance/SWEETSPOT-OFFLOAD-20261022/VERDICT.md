# Sweet-spot OFFLOAD_RATIO pack suitability — 2026-10-22

**Verdict:** `SWEETSPOT_PASS` (host unit; N=40 scaled MoE layer)  
**Branch:** `scratch/uma-02-host-harness-20261022`  
**Filter:** `OffloadSweetSpots*:moe_otd_sweetspot.*`  
**Result:** **10/10 PASSED** (234 ms)

## Ratios under test

Production reference: Qwen / C6 `memory_safe` uses **OFFLOAD_RATIO=20**. Nearby spots **25** and **30** (historical functional pack benches also used 25).

| OFFLOAD_RATIO | resident slots (N=40) | upload_bytes (SS01 gate_w) |
|---------------|-----------------------|----------------------------|
| 20            | 32                    | 1048576                    |
| 25            | 30                    | 983040                     |
| 30            | 28                    | 917504                     |

Formula (plugin lock): `max(1, N * (100 - R) / 100)` — same as `moe_offload_constant.cpp`.

## Cases

| ID | Claim |
|----|--------|
| SS01 | `try_prepare_partial_upload` upload_shape[0] matches formula at 20/25/30 |
| SS02 | pack build → require → working set ≤ capacity → simultaneous lease + byte-correct hits |
| SS03 | capacity+1 simultaneous reject; sequential fill + eviction still serves (expert_loads) |
| SS04 | ratio=20 path A (`usm_device`) vs path B (`usm_host`) pack fill byte parity |

## Evidence

- Log: `C:\gm\orchestration\sweetspot-offload-20261022\evidence\sweetspot-gtest.txt`
- Exit: `0`
- Binary SHA256: `ED14876C4C36A7316F0ABA25B609CD72CE82EE728A958A427FCF3C76EE729ACE`
- Source: `moe_otd_sweetspot_offload_test.cpp`

## Scope / non-claims

- Not full-model TTFT / OVMS soak; scaled pack fixture only.
- No golden C6 / production OVMS / driver change.
- Path B remains scratch (`OTD_RESIDENT_USM_HOST` default off).
