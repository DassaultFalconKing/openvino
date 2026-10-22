# Verdict — UMA-02 G03 + UMA-03 cycle

**Device:** Intel(R) Arc(TM) 140V GPU  
**Branch:** `scratch/uma-02-host-harness-20261022`  
**Draft PR:** https://github.com/DassaultFalconKing/openvino/pull/8  
**Linux:** `LINUX_DEFERRED_NO_MACHINE`  
**Production enablement:** **NO** (`OTD_RESIDENT_USM_HOST` default off)

## Gates

| Gate | Result |
|---|---|
| UMA-02 HOST (P/N/L/A) | **PASS** (prior) |
| U2-G01 kernel digest A==B | **PASS** (prior) |
| U2-G02 cold-miss counters | **PASS** (prior) |
| U2-G03 simultaneous + eviction | **PASS** |
| U3-B01 steady digest proxy | **PASS** (measured; no production win claim) |
| U3-B02 cold-miss timing A vs B | **PASS** (measured; fixture-scale only) |
| Full `moe_otd_uma02.*` | **10/10 PASS** |
| `moe_otd_uma03.*` | **2/2 PASS** |

## Recommendation

**`UMA_B_PROTOTYPE_MEASURED`** — path B remains a scratch candidate. Fixture-scale numbers suggest cold-fill path B can be cheaper than path A `copy_from` on this host, while steady digest bandwidth is roughly comparable. **Do not** enable in OVMS/production without a real MoE GEMM / TTFT study on target models.

**Do not merge** without operator decision.
