# SOURCE-IDENTITY

## Baseline
- Repo: `DassaultFalconKing/openvino`
- Base branch: `research/ov-gpu-otd-cohort-b5bfb148`
- Pinned baseline SHA: `479219245f6f10b670367ad35af16513d72d2091`

## Remote PR HEADs (verified 2026-10-22)
| PR | Tip SHA | Match pin |
|----|---------|-----------|
| #1 | `0ba38e441dc9327f473c1b4d36a2bef1aaeeac98` | MATCH |
| #4 | `20cf30f7082b490d70f0f4c6779aa05aef2ed662` | MATCH |
| #2 | `dbcf08c0ae90cd611172de18bb5c41f50127225e` | MATCH |
| #3 | `4e8fa732151e1169aa58faa9a5732f2b021d5b16` | MATCH |
| #5 | `2646ae60223d2ba4a7f36b49813641bff4399995` | MATCH |

## Integration merges (order #1, #4, #2, #3, #5) then docs sync
```
99ee2c2c docs(gpu/otd): sync integration to pinned PR#1 tip docs
94fa8284 Merge PR #5 trace evidence gate
6f5cee48 Merge PR #3 trace accounting
aab6bbdb Merge PR #2 atomic publication
f46ff045 Merge PR #4 reader identity pin
b81a82d8 Merge PR #1 expert pack integrity
2646ae60 test(gpu/otd): gate clean, legacy and incomplete traces for policy evidence
ebb7c805 fix(gpu/otd): label incomplete traces ineligible for cache policy acceptance
20cf30f7 docs(gpu/otd): CLD-01B same-handle reader pin handoff
359445cd fix(gpu/otd): pin serving pack handle to validated metadata snapshot
f288a5b9 docs(gpu/otd): Cloud Wave 01 state for CLD-01/02/03 with receipts and QA order
4e8fa732 [docs] CLD-03 handoff: trace integrity and offline accounting
a973b006 [GPU] MoE OTD: trace integrity records and strict offline accounting (CLD-03)
dbcf08c0 docs(gpu/otd): CLD-02 handoff for pack publication
65d4afca fix(gpu/otd): publish packs with a single rename and fail closed on refused replacement
07770ec1 docs(gpu/otd): CLD-01 handoff for follow-up hardening and static-check receipts
48748f92 fix(gpu/otd): validate slot before pack read, typed integrity error, verify counters
8e1bb766 docs(gpu/otd): CLD-01 local handoff and cloud wave state
2ed011fc fix(gpu/otd): verify pack expert payload before resident GPU copy
```

## Drift note
First merge of PR#1 used tip `f288a5b9` (ancestor of pin). Docs-only `0ba38e44` cherry-picked so integration matches remote PR#1 HEAD. No code delta between those tips.

## Forbidden surfaces
Golden C6, production OVMS, main models, Intel drivers, global Windows config: untouched.
