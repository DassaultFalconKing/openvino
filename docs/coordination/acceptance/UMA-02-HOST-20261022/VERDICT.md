# UMA-02 HOST verdict — 2026-10-22

**Verdict:** `UMA-02 HOST_PASS`  
**GPU:** `UMA-02 GPU_BLOCKED` (`COMPUTE_GO` absent — G01–G03 NOT_RUN)  
**Linux:** `LINUX_DEFERRED_NO_MACHINE`  
**Production enablement:** **NO** — flag `OTD_RESIDENT_USM_HOST` default off; do not merge to mainline without operator decision.

## Identity

| Item | Value |
|---|---|
| Branch | `scratch/uma-02-host-harness-20261022` |
| HEAD (code+docs) | `74c8ea0d60692f1ac22436a311e63866c1f064b8` |
| Draft PR | https://github.com/DassaultFalconKing/openvino/pull/8 |
| Base (Wave-01) | `74003278a9c8925fcda7bc32ceb27320e59bbb1d` |
| Cohort baseline | `479219245f6f10b670367ad35af16513d72d2091` |
| Design pack | UMA-01 `docs/.../UMA-01-DESIGN-20261022/` / OVMS Draft PR #22 |
| Binary SHA256 | see local `C:\gm\orchestration\uma-02-windows-20261022\evidence\` |

## Host matrix (required for HOST_PASS)

| ID | Result |
|---|---|
| U2-P01 single-expert parity | **PASS** |
| U2-P02 multi-slot no crosstalk | **PASS** |
| U2-P03 evict-then-refill parity | **PASS** |
| U2-N01 checksum fail-closed | **PASS** |
| U2-N02 no scatter on verify fail | **PASS** |
| U2-L01 no false hit after fail | **PASS** |
| U2-L02 hit only after recover | **PASS** |
| U2-A01 usm_host under flag=1 | **PASS** |
| U2-A02 preferred alloc with flag unset | **PASS** |
| U2-A03 fence before publish | **PASS** |
| U2-G01–G03 | **BLOCKED_COMPUTE_GO** |

Command: `ov_gpu_unit_tests.exe --gtest_filter=moe_otd_uma02.*` → **7/7 PASS**  
Regression: `moe_expert_weight_provider.pack_*` → **4/4 PASS** (flag unset).

## What landed (scratch only)

1. Env `OTD_RESIDENT_USM_HOST=1` → routed partial-upload residents as `usm_host` (else preferred/`usm_device`).
2. `fill_weights_from_pack` path B: verify → CPU scatter into `usm_host` → `stream.flush()`+`finish()` → return (caller `set_filled`).
3. Host gtests in `moe_otd_uma02_test.cpp`.

## Non-claims

- Not production OTD enablement.
- Not Arc kernel-read / GEMM bandwidth proof (UMA-03 / COMPUTE_GO).
- Fence choice is flush+finish; NEO kernel visibility still needs GPU_LIVE confirmation.
