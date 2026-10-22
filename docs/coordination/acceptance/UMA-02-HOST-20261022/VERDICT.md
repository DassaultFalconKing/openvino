# UMA-02 HOST verdict — 2026-10-22 (reconciled)

**Verdict:** `UMA-02 HOST_PASS`  
**GPU short matrix:** `UMA-02 GPU_KERNEL_PASS` (G01–G03) — see [GPU-LIVE.md](./GPU-LIVE.md) and `../UMA-03-CYCLE-20261022/`  
**P0-B in-flight ownership:** see `../P0B-INFLIGHT-20261022/` (supersedes “ready for model A/B” until that gate)  
**Linux:** `LINUX_DEFERRED_NO_MACHINE`  
**Production enablement:** **NO** — flag `OTD_RESIDENT_USM_HOST` default off; do not merge to mainline without operator decision.

## Identity

| Item | Value |
|---|---|
| Branch | `scratch/uma-02-host-harness-20261022` |
| HEAD at GPU_LIVE/sweetspot close | `d82f5f573dfb2b30a4cbf35b1cb8f75e4066c621` |
| Draft PR | https://github.com/DassaultFalconKing/openvino/pull/8 |
| Base (Wave-01) | `74003278a9c8925fcda7bc32ceb27320e59bbb1d` |
| Cohort baseline | `479219245f6f10b670367ad35af16513d72d2091` |
| Binary SHA256 (pre-P0B) | `ED14876C4C36A7316F0ABA25B609CD72CE82EE728A958A427FCF3C76EE729ACE` |

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

## GPU_LIVE (short; COMPUTE_GO)

| ID | Result |
|---|---|
| U2-G01 kernel digests A==B | **PASS** (`GPU_KERNEL_PASS`) |
| U2-G02 cold-miss counters | **PASS** (`GPU_KERNEL_PASS`) |
| U2-G03 simultaneous + eviction | **PASS** (`GPU_KERNEL_PASS`) |

**Reconcile note:** An earlier revision of this file marked G01–G03 `BLOCKED_COMPUTE_GO` / `NOT_RUN`, and Draft PR #8 body still said “GPU_LIVE BLOCKED”. Those statuses are **stale**. Live evidence: [GPU-LIVE.md](./GPU-LIVE.md), UMA-03 VERDICT, sweetspot VERDICT.

## What landed (scratch only)

1. Env `OTD_RESIDENT_USM_HOST=1` → routed partial-upload residents as `usm_host` (else preferred/`usm_device`).
2. `fill_weights_from_pack` path B: verify → (P0-B) pre-write drain → CPU scatter → `stream.flush()`+`finish()` → return (caller `set_filled`).
3. Host + GPU_LIVE gtests in `moe_otd_uma02_test.cpp`; P0-B overlap/visibility in `moe_otd_p0b_inflight_test.cpp`.

## Non-claims

- Not production OTD enablement.
- Not real MoE GEMM / OVMS model A/B until P0-B verdict `B_SAFE_FOR_MODEL_AB` + operator OK.
- Fence choice is flush+finish; NEO kernel visibility covered by U2-G01 and P0B-V1.
