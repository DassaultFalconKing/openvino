# P0-B root cause — path-B in-flight slot overwrite

**Authority:** OVMS Draft PR #23 + OpenVINO Draft PR #8 coordinator review  
**Code:** `src/plugins/intel_gpu/src/graph/impls/ocl_v2/moe/moe_otd_runtime.cpp`  
**Status:** identified; fixed on scratch branch (pre-write drain + retained post-write fence)

## Race

Path B did:

1. `scatter_verified_record_to_residents` — unqueued `std::memcpy` into an existing `usm_host` slot  
2. `fence_path_b_publish` — `flush`+`finish` on the fill stream  
3. Caller `set_filled` in `expert_weight_providers.cpp`

A wait **after** an unqueued host write does **not** prove that prior GPU kernels still reading a recycled slot completed **before** the overwrite. Path A’s queued `copy_from(..., wait=true)` has different ordering. `U2-G03` / sweetspot do not cover overlapping GEMM or two queues.

```text
LRU miss / evict slot S
  Prior GPU kernel still reading S  --RACE-->  CPU memcpy scatter(E_new → S)
                                                    |
                                                    v
                                              flush+finish
                                                    |
                                                    v
                                              set_filled publish
```

## Fix (scratch, flag default off)

1. **Pre-write ownership:** `finish()` on the fill stream (and any registered secondary reader streams) **before** memcpy scatter.  
2. **Post-write visibility:** keep `flush`+`finish` so a GPU launched after scatter sees new bytes **before** `set_filled`.

## Non-claims

- Not production enablement (`OTD_RESIDENT_USM_HOST` default off).  
- Not C6 / OVMS / driver change.  
- Not merge of PR #8 / #6.
