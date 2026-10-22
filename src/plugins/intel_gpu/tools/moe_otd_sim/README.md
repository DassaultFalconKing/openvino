# MoE OTD trace replay

Offline tools for the Intel GPU MoE offload-to-disk (OTD) expert cache. They read traces
written by the plugin and never touch the runtime.

## Recording a trace

Set `MOE_OTD_TRACE=<file>` before the process compiles an OTD-enabled model
(`OFFLOAD_RATIO` 1..99). Every expert-slot acquisition is recorded per MoE layer:

```text
# moe_otd_trace v1
L,<layer>,<layer_key>,<model_key>,<capacity>,<num_expert>,<top_k>
C,<call>,<t_ns>,<layer>,<S|O>,<routed>,<ok|overflow>,<unique experts in first-occurrence order>
E,<call>,<t_ns>,<layer>,<expert>,<slot>,<hit>,<evicted|-1>,<bytes>,<reads>,<read_ns>,<transform_ns>,<copy_ns>
```

`layer_key` and `model_key` are FNV-1a hashes of the primitive name and weights path, so a
target model and a separately compiled drafter are distinguishable without writing paths.
No prompt or token data is recorded. `MOE_OTD_PERF_LOG=1` independently enables the
process-exit counter dump (now including bytes read, evictions, overflow partial loads and
read-latency p50/p95/p99 bucket bounds).

Not exposed by the plugin, and therefore not in the trace: router top-k weights, request or
sequence identity, and draft/verify/accepted/rejected positions. Supply committed tokens from
the serving benchmark with `--committed-tokens`; never derive them from the trace.

## Commands

```text
python otd_sim.py verify   <trace>    # exact LRU replay must reproduce every recorded slot/hit/eviction
python otd_sim.py summary  <trace>    # recorded read / transform / copy time, loads, bytes
python otd_sim.py simulate <trace> [--policy lru|clock|2q|opt]... \
       [--capacity N... | --capacity-fraction F...] [--precheck] [--committed-tokens N]
python synth_trace.py <out.trace> [--sessions 4 --spec-window 3 --capacity 16 ...] [--meta meta.json]
python -m unittest test_otd_sim
```

`simulate` replays per layer. When a simulated capacity lets an overflowing lease succeed,
the recorded per-expert fallback calls are skipped; when it makes a lease overflow, the
fallback is synthesized in sorted expert order. `--precheck` models rejecting an
overflowing lease before any expert is loaded. `opt` (Belady) is an offline lower bound on
misses, not an implementable policy. Results are split by call phase (1-token decode,
2-8 token windows such as speculative verification, larger prefill batches, fallback) and
by model key.
