# UMA-03 bandwidth / cold-miss — Arc 140V

**Status:** **PASS** (measurements recorded)  
**Tests:** `moe_otd_uma03.U3_B01_*`, `moe_otd_uma03.U3_B02_*`  
**Proxy:** schedule-independent FNV digest kernel over 2048 B tensor0 (not MoE GEMM)

## Representative run (this host)

### U3-B01 steady kernel read (40 iters after warmup)

| Path | p50_ns | p95_ns | approx GB/s (proxy) |
|---|---:|---:|---:|
| A `usm_device` | ~55100 | ~86100 | ~0.037 |
| B `usm_host` | ~50300 | ~71500 | ~0.041 |
| ratio host/device p50 | **~0.91** | | |

On this fixture, path B is **not slower** than path A for the digest proxy (noise-dominated at 2 KiB).

### U3-B02 cold miss fill (12 iters, capacity=1)

| Path | cold_p50_ns | copy_p50_ns |
|---|---:|---:|
| A `usm_device` + `copy_from` | ~179300 | ~85900 |
| B `usm_host` + CPU scatter | ~85200 | ~13200 |
| ratio B/A | **~0.48** cold | **~0.15** copy |

Path B cold-fill looks cheaper on this synthetic pack. **Do not** extrapolate to OVMS TTFT / full expert tensors.

## Explicit warnings

- Fixture experts / 2 KiB windows — not production weight sizes.
- Digest ≠ GEMM / SwiGLU bandwidth.
- No claim that path B wins end-to-end inference.
- Flag remains default **off**.
