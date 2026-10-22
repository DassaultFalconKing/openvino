# Decode 4× regression — investigation 2026-10-23

**Symptom:** staged / OTD / `runtime\base` GPU plugin+OVMS ≈ **8.2 tok/s** vs stock 2026.4 ≈ **32 tok/s** on Qwen3.6-35B-A3B-int4 MoE decode (same harness, same model, GPU).

**Stock identity:** `2026.4.0-22930-61afcb26271-releases/2026/4` (`C:\llm\ovms`)  
**Head identity:** `2026.5.0-5-591ab0b517a-codex/c6-head-20261019-openvino` (`C:\gm\otd-research\runtime\base`)

## Measured

| Arm | OFFLOAD_RATIO | parsers | decode tok/s |
|---|---|---|---|
| stock 2026.4 | 20 | qwen3coder+qwen3 | 31.8 / 32.2 / 33.0 |
| base-head | 20 | same | 8.12 / 8.28 / 8.28 |
| base-head | **0** | same | **8.39 / 8.36** |
| base-head | 0 | **none** | **8.48** |

Warmup TTFT: stock ~13.5 s, head ~18 s (≈1.3×). Decode is the 4× cliff.

## Falsified (not the micro-root)

1. **OTD hooks always-on** — `on_before_batched_gemv` `stream.finish()` + slot remap only run when `lru_expert_num > 0`. `OFFLOAD_RATIO=0` forces `ResidentExpertWeightProvider` and the 4× remains.
2. **Partial-upload `usm_device`** — `moe_offload_constant.cpp` `get_preferred_memory_allocation_type()` (device) vs 2026.4 lockable-preferred (`usm_shared`). Only applies when `0 < ratio < 100`. Dead at ratio=0.
3. **Build flags** — both Release `/O2 /Ob2`, `ENABLE_ONEDNN_FOR_GPU=ON`, `GPU_RT_TYPE=OCL`, debug caps off. Not a debug/`finish_after_enqueue` build.
4. **OVMS tool/reasoning parsers** — same 8.5 tok/s without them.
5. **GEMV kernel source** — `moe_3gemm_swiglu_mlp.cl` decode kernels unchanged except a comment. `mlp_reduce` LWS `gcd(2048,1024)=1024` = stock `min(1024)`.
6. **`supports_replay()=false`** — record/replay default off; LLM is dynamic + KV variables. Not used.
7. **Cross-plugin DLL swap** — 2026.4 plugin into 2026.5 core fails (error 127). Isolation is three-way bench, not ABI mix.

## What *is* true

- Decode is supposed to take `exec_batched_gemv` when `token_num <= moe_batched_gemv_threshold` (default 32). Source of that function is the same in 61afcb and 591ab0 except braces + gcd on reduce.
- Head compiled-model cache has an extra **1.22 GB** `.blob` (stock top blobs 486+426 MB only). Same `cl_cache`/`onednn.cl_cache` *counts* (63/26), so the extra mass is serialized graph / extra impls, not a new kernel family count.
- 2026.4→head GPU plugin is a large delta: `primitive_inst.cpp` +1434, `network.cpp` +695, `ocl_stream.cpp` +395, `transformations_pipeline.cpp` +255, `sdpa_gen_micro.cpp` +680, plus new MLIR/ZE paths (built with `GPU_RT_TYPE=OCL`, so ZE is not the default engine).

## Open micro-root (not yet caught with a single-line proof)

The 4× is **inside the 2026.5 `openvino_intel_gpu_plugin` execute path**, not OTD and not flags.

## MOE_PATH_LOG probe (2026-10-23 00:40)

Release-safe `fprintf` in `moe_3gemm_swiglu_opt_impl::execute()` after `get_seq_len`. Staged DLL into `runtime\base`. Same harness, `OFFLOAD_RATIO=0`, `MOE_PATH_LOG=1`, 32 tok / 1 repeat.

**Result: 8.99 tok/s. Decode is on batched GEMV. `get_seq_len` is not the bug.**

| phase | n | token_num | thr | path | shape |
|---|---|---|---|---|---|
| prefill (warmup + measured) | 0–4, 1280 | 55 | 32 | **grouped** | `55,1,2048` |
| decode (every 40th call) | 40…1240, 1320…2520 | **1** | 32 | **gemv** | `1,1,2048` |

No decode call with `token_num>32`. No `micro` / `onednn` on this model. `rank=3` dynamic layout is real tokens, not padded.

**Falsified:** “decode takes prefill grouped/onednn because `get_seq_len` overcounts.” Do not patch `get_seq_len`.

**Still open (ranked):**

1. **GEMV body is slower in 2026.5** even on the same `exec_batched_gemv` source (weight bind / expert gather / extra events / `bind_weights_on_first_exec` every step). Kernel `.cl` text was already same-as-stock; the *callers* around GEMV were not timed.
2. **Non-MoE on the same step** (PA GENERATE / GDN / extra 1.22 GB blob). MoE path is correct, so a 4× can still sit outside MoE.
3. **Host sync in `primitive_inst` / `ocl_stream`** every primitive. Util sampler still `n=0`.

## Evidence paths

- Harness: `C:\gm\orchestration\decode-regression-20261022\ab-run.ps1`
- Benches: `runs\stock-2026.4-20261022-235643`, `runs\base-head-20261022-235332`, `runs\base-head-r0-20261023-001850`, `runs\head-util-r0-20261023-002334`, **`runs\head-moe-path-20261023-004014`**
- Path log: `runs\head-moe-path-20261023-004014\stderr.log`
- Sources compared: `61afcb26271` vs `591ab0b5` (`moe-2026.4-to-head.diff`, `swiglu_opt.*.cpp`)
- Probe (env-gated, leave in): `moe_3gemm_swiglu_opt.cpp` `MOE_PATH_LOG`
