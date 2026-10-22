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
No prompt or token data is recorded. An overflowing `S` lease is rejected before the cache is
touched, so it has no `E` records; the per-expert fallback follows as `O` calls.
`MOE_OTD_PERF_LOG=1` independently enables the process-exit counter dump (now including bytes
read, evictions and read-latency p50/p95/p99 bucket bounds).

Not exposed by the plugin, and therefore not in the trace: router top-k weights, request or
sequence identity, and draft/verify/accepted/rejected positions. Supply committed tokens from
the serving benchmark with `--committed-tokens` or a run sidecar; never derive them from the trace.

### Integrity extensions

Version 1 readers skip `#` lines, so the writer adds integrity records as comments:

```text
# incomplete C,...   # a call still open when the writer closed (its load threw), status "incomplete"
# incomplete E,...   # the experts of that call whose load completed before the failure
# end,calls=<n>,experts=<n>,incomplete_calls=<n>,incomplete_experts=<n>
```

The end record is written when the writer is closed or destroyed (normal process exit). A trace
without one was cut short (crash, kill, disk full) or predates the extension; it still parses
with a warning and `trace_integrity.complete=false`. If the end-record counts do not match the
records read, records were lost and parsing fails. An incomplete call lacks the miss that threw:
that lookup had already evicted and reassigned a slot, so replay of later calls on the same layer
may diverge, and `verify` reports it as mismatches rather than hiding it.

Parsing is strict: wrong field counts, duplicate layer/call ids, values outside uint64, slots or
experts out of range, hits that evict, `E` records on overflowing calls or the wrong layer, and
records after the end record all fail with the line number. A final line without a terminator is
treated as truncated even when it looks valid; `--salvage` discards only that line and reports it.
Malformed records before the end are never skipped. Exit code 2 means the trace or sidecar was
rejected; `verify` exits 1 on replay mismatches.

### Logical and physical bytes

`bytes`/`reads` in `E` records are the expert payload the runtime asked for (logical). They are
reported as `logical_expert_bytes`/`logical_reads`. Physical NVMe traffic (page cache hits,
read-ahead, filesystem amplification) is not visible to the plugin: `physical_nvme_read_bytes`
is `NOT_EXPOSED` unless a run sidecar supplies it with its counter source. Simulated policy
results always report it as `NOT_EXPOSED`; a sidecar value is only echoed under `run_meta`,
because a different cache policy would change physical traffic in ways the trace cannot predict.

### Run sidecar (`otd_run_meta v1`)

`--run-meta meta.json` attaches facts the trace cannot contain. Every field except `schema` is
optional; absent fields are reported as `NOT_EXPOSED`, unknown fields are rejected.

```json
{"schema": "otd_run_meta v1", "trace_sha256": "<sha256 of the trace file bytes>",
 "openvino_sha": "...", "genai_sha": "...", "ovms_sha": "...", "runtime": "...", "quant": "...",
 "target_model": "...", "drafter_model": "...", "workload": "C1|C2|C4", "kv_cache_precision": "...",
 "context_tokens": 0, "proposed_tokens": 0, "accepted_tokens": 0, "committed_tokens": 0,
 "parser_result": "pass|fail", "tool_loop_result": "pass|fail",
 "physical_nvme_read_bytes": 0, "physical_read_counter_source": "..."}
```

A `trace_sha256` that does not match the trace, `accepted_tokens > proposed_tokens`, or physical
bytes without a source are errors. `committed_tokens` from the sidecar feeds the per-token rates;
a different `--committed-tokens` value is an error. The `synth_trace.py --meta` file is generator
metadata, not a run sidecar. Synthetic traces carry no end record, so they report
`trace_integrity.complete=false`.

## Commands

```text
python otd_sim.py verify   <trace> [--salvage] [--format json|csv]
python otd_sim.py summary  <trace> [--salvage] [--format json|csv] [--run-meta meta.json]
python otd_sim.py simulate <trace> [--policy lru|clock|2q|opt]... \
       [--capacity N... | --capacity-fraction F...] [--legacy-partial-lease] [--committed-tokens N] \
       [--salvage] [--format json|csv] [--run-meta meta.json]
python synth_trace.py <out.trace> [--sessions 4 --spec-window 3 --capacity 16 ...] [--meta meta.json]
python -m unittest test_otd_sim
```

`verify` and `summary` print `trace_integrity` (end record, incomplete calls, salvaged line,
SHA-256 of the trace file, warnings). CSV output has one `result,metric,value` row per metric,
with nested keys joined by dots.

`simulate` replays per layer. When a simulated capacity lets an overflowing lease succeed,
the recorded per-expert fallback calls are skipped; when it makes a lease overflow, the
fallback is synthesized in sorted expert order. `--legacy-partial-lease` models the upstream
runtime, which streams experts of an overflowing lease (evicting resident ones) before rejecting it. `opt` (Belady) is an offline lower bound on
misses, not an implementable policy. Results are split by call phase (1-token decode,
2-8 token windows such as speculative verification, larger prefill batches, fallback) and
by model key.
