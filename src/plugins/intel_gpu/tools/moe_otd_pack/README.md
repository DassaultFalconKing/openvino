# Offline OpenVINO MoE OTD pack v1 verifier

Independent, **read-only** Python inspector for the existing 4096-aligned pack v1 binary format. No OpenVINO Python wheel, model weights, GPU, NumPy or other runtime dependencies. The tool is intended for Windows operator scratch packs and the CLD-01/02/01B negative-test acceptance track. It does not modify the OpenVINO plugin and cannot authorize production integration by itself.

## Usage (Windows, PowerShell)

```powershell
cd src/plugins/intel_gpu/tools/moe_otd_pack
python -m unittest -v test_verify_pack
python verify_pack.py 'C:\scratch\otd-packs\otd-0123456789abcdef.pack' --json --expect-key 0x0123456789abcdef --expect-experts 6
if ($LASTEXITCODE -ne 0) { throw 'pack verifier detected an invalid or damaged pack' }
```

Use the **real expected** key/expert count for your model, not the illustrative ones above. The file must retain its content-keyed basename. Do not publish pack paths, model weights, or machine-specific failure logs. Run against an **immutable/quiescent** file: coordinate with builders so no concurrent in-place modification occurs. A rename after opening on Windows may fail because the tool holds an open handle.

Exit codes: `0` valid geometry and all logical-expert checksums match, `1` header/table valid but one or more expert payloads corrupt, `2` malformed metadata, wrong expected identity, file I/O error or unstable metadata. `--json` prints a stable `otd_pack_verify_v1` record with `checked_experts`, `corrupt_experts`, `layer_key`, `record_stride`, `payload_bytes_per_expert` and `physical_nvme_read_bytes=NOT_EXPOSED`. `trailing_bytes` is reported but not fatal because the native `PackLayout::open` permits extra trailing data.

The verifier explicitly follows the on-disk [pack-v1 struct and packer](../../src/graph/impls/ocl_v2/moe/moe_otd_pack.hpp): packed **little-endian** 216-byte header, header FNV-1a checksum with its checksum field cleared, 64-byte tensor alignment, 4096-byte record alignment, logical-expert placement permutation, and logical-expert-indexed FNV-1a checksums over **unpadded** record payload bytes. For Windows Intel/AMD x86, little-endian matches the native writer. Fixed-size 1-MiB reads make memory independent of pack size. `--max-experts` (default 65536) limits metadata allocations from untrusted headers. This is a v1 compatibility diagnostic, not a cryptographic authenticity verifier: coordinated mutation of payload and FNV table cannot be detected, and neither model provenance nor OS-physical NVMe bytes are inferred.

## Tests and isolation

```powershell
python -m unittest -v test_verify_pack
```

Synthetic test records exercise a six-expert pack with nontrivial placement, all nine tensor fields, first/last expert corruption, checksum-table corruption, resealed noncanonical header, malformed/duplicate placement, truncation, wrong model key/file name, alignment padding, >2MiB payload streaming, trailing data, and CLI exit statuses. The synthetic source fixture is not a substitute for comparing the verifier against the real C++ `build_pack()` output on Windows.

## Native differential acceptance

1. Build a pack with the **combined Windows OpenVINO integration** in an isolated scratch directory.
2. Run this verifier with the known model key and expert count; compare its `status` and failed logical expert IDs with the native `verify_pack()` and `PackLayout::open()` responses.
3. Copy a pack in the scratch directory, corrupt one payload byte, one placement entry, one checksum, then reseal an illegal header offset; repeat the C++/Python comparison. Never patch the real production pack.
4. For concurrent publication, do not run this tool on a path undergoing active replacement and call an incidental open failure evidence of corruption. The CLD-02 race tests own publication semantics; this tool is intentionally read-only, single-handle, and snapshot-limited.
5. Keep original C6, OVMS production and operator GPU compute pause untouched. Only integrate after native build, targeted negative units, provider tests and controlled Arc acceptance are passed by local agents.

**Source task:** Cloud Wave A4 negative pack integrity / A2 trustworthy offline evidence. Not CLD-04 scheduler and not an inference-path optimization.
