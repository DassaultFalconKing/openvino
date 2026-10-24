# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Read-only, bounded-memory offline verifier for native OpenVINO GPU OTD pack v1.

This is a second implementation of the v1 *disk contract*, not an inference reader.
It does not prove trusted model provenance or protect against adversarial hash rewrites.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import stat
import struct
import sys
from typing import BinaryIO

MAGIC = b"OTDPACK1"
VERSION = 1
ALIGNMENT = 4096
TENSOR_ALIGNMENT = 64
TENSORS = 9
UINT64_MAX = (1 << 64) - 1
FNV_OFFSET = 14695981039346656037
FNV_PRIME = 1099511628211
HEADER = struct.Struct("<8sII" + "Q" * (6 + TENSORS * 2 + 1))
U32 = struct.Struct("<I")
U64 = struct.Struct("<Q")
READ_CHUNK = 1 << 20
DEFAULT_MAX_EXPERTS = 65536


class PackError(ValueError):
    """Rejected malformed metadata, inconsistent layout or an unreadable pack."""


def fnv_update(hash_value: int, data: bytes) -> int:
    for byte in data:
        hash_value = ((hash_value ^ byte) * FNV_PRIME) & UINT64_MAX
    return hash_value


def align_up(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def exact_read(fh: BinaryIO, nbytes: int, offset: int, what: str) -> bytes:
    fh.seek(offset)
    data = fh.read(nbytes)
    if len(data) != nbytes:
        raise PackError(f"truncated {what}")
    return data


def canonical_layout(tensor_sizes: list[int], n: int) -> tuple[list[int], int, int, int, int]:
    offsets: list[int] = []
    cursor = 0
    payload = 0
    for size in tensor_sizes:
        offsets.append(cursor)
        if size:
            payload = cursor + size
        cursor = align_up(cursor + size, TENSOR_ALIGNMENT)
    if cursor == 0:
        raise PackError("all nine tensors are empty")
    stride = align_up(cursor, ALIGNMENT)
    checksum_offset = ALIGNMENT + align_up(n * 4, 8)
    records_offset = align_up(checksum_offset + n * 8, ALIGNMENT)
    file_end = records_offset + n * stride
    if file_end > (1 << 63) - 1:
        raise PackError("canonical file layout exceeds C++ streamoff")
    return offsets, payload, stride, checksum_offset, records_offset


def inspect_pack(path: Path, *, expect_key: int | None = None,
                 expect_experts: int | None = None, max_experts: int = DEFAULT_MAX_EXPERTS) -> dict:
    """Check canonical v1 geometry and hash *every* logical-expert payload.

    Uses a single open read handle and 1-MiB chunks, without loading payloads into RAM.
    Call only on immutable/quiescent packs: concurrent in-place mutation is not snapshot-safe.
    """
    if max_experts <= 0 or max_experts > 0xFFFFFFFF:
        raise PackError("max_experts must be in [1, UINT32_MAX]")
    path = Path(path)
    with path.open("rb") as fh:
        st = os.fstat(fh.fileno())
        if not stat.S_ISREG(st.st_mode):
            raise PackError("not a regular file")
        if st.st_size < HEADER.size:
            raise PackError("truncated header")
        raw_header = exact_read(fh, HEADER.size, 0, "header")
        fields = HEADER.unpack(raw_header)
        magic, version, header_bytes = fields[:3]
        key, n, stride, records_offset, placement_offset, checksum_offset = fields[3:9]
        sizes = list(fields[9:18])
        offsets = list(fields[18:27])
        checksum = fields[27]

        if magic != MAGIC or version != VERSION or header_bytes != ALIGNMENT:
            raise PackError("unsupported pack magic/version/header size")
        if fnv_update(FNV_OFFSET, raw_header[:-8] + b"\0" * 8) != checksum:
            raise PackError("header checksum mismatch")
        if n == 0 or n > 0xFFFFFFFF or n > max_experts:
            raise PackError("expert count exceeds configured safety bound")
        if expect_key is not None and key != expect_key:
            raise PackError("layer key differs from expected trusted key")
        if expect_experts is not None and n != expect_experts:
            raise PackError("expert count differs from expected")
        if path.name != f"otd-{key:016x}.pack":
            raise PackError("filename does not match embedded layer key")

        canonical_offsets, payload, canonical_stride, canonical_checksums, canonical_records = canonical_layout(sizes, n)
        if (offsets != canonical_offsets or stride != canonical_stride or
                placement_offset != ALIGNMENT or checksum_offset != canonical_checksums or
                records_offset != canonical_records):
            raise PackError("non-canonical header layout")
        end = records_offset + n * stride
        if st.st_size < end:
            raise PackError("truncated expert records")

        placement_raw = exact_read(fh, n * U32.size, placement_offset, "placement table")
        placement = list(struct.unpack("<" + "I" * n, placement_raw))
        if sorted(placement) != list(range(n)):
            raise PackError("placement table is not a permutation")
        checksum_raw = exact_read(fh, n * U64.size, checksum_offset, "checksum table")
        checksums = struct.unpack("<" + "Q" * n, checksum_raw)

        failures: list[int] = []
        for expert in range(n):
            offset = records_offset + placement[expert] * stride
            fh.seek(offset)
            digest = FNV_OFFSET
            remaining = payload
            while remaining:
                part = fh.read(min(remaining, READ_CHUNK))
                if not part:
                    raise PackError("record became unreadable during verification")
                digest = fnv_update(digest, part)
                remaining -= len(part)
            if digest != checksums[expert]:
                failures.append(expert)

        # Detect obvious concurrent in-place metadata mutation. This is not a snapshot
        # isolation guarantee; the caller must keep builders out of the directory.
        if (os.fstat(fh.fileno()).st_size != st.st_size or
                exact_read(fh, HEADER.size, 0, "final header") != raw_header or
                exact_read(fh, n * U32.size, placement_offset, "final placement") != placement_raw or
                exact_read(fh, n * U64.size, checksum_offset, "final checksums") != checksum_raw):
            raise PackError("file changed during inspection; retry only on a quiescent pack")

    return {
        "schema": "otd_pack_verify_v1",
        "file": path.name,
        "status": "CORRUPT" if failures else "OK",
        "layer_key": f"{key:016x}",
        "num_experts": n,
        "payload_bytes_per_expert": payload,
        "record_stride": stride,
        "expected_file_bytes": end,
        "actual_file_bytes": st.st_size,
        "trailing_bytes": st.st_size - end,
        "checked_experts": n,
        "corrupt_experts": failures,
        "corrupt_count": len(failures),
        "physical_nvme_read_bytes": "NOT_EXPOSED",
    }


def _parse_key(text: str) -> int:
    value = int(text, 0)
    if not 0 <= value <= UINT64_MAX:
        raise argparse.ArgumentTypeError("key must fit uint64")
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pack", type=Path, help="single immutable pack file to validate, never modified")
    parser.add_argument("--expect-key", type=_parse_key, help="trusted model/layer key, decimal or 0x-hex")
    parser.add_argument("--expect-experts", type=int, help="trusted expert count")
    parser.add_argument("--max-experts", type=int, default=DEFAULT_MAX_EXPERTS)
    parser.add_argument("--json", action="store_true", help="emit machine-readable result")
    args = parser.parse_args(argv)
    try:
        result = inspect_pack(args.pack, expect_key=args.expect_key, expect_experts=args.expect_experts,
                              max_experts=args.max_experts)
    except (PackError, OSError, OverflowError) as exc:
        result = {"schema": "otd_pack_verify_v1", "file": args.pack.name, "status": "INVALID",
                  "error": str(exc), "physical_nvme_read_bytes": "NOT_EXPOSED"}
        code = 2
    else:
        code = 1 if result["corrupt_count"] else 0
    if args.json:
        print(json.dumps(result, sort_keys=True))
    else:
        print(f"{result['status']}: {result['file']}" +
              (f" - {result['error']}" if "error" in result else
               f" - {result['checked_experts']} checked, {result['corrupt_count']} corrupt"))
    return code


if __name__ == "__main__":
    sys.exit(main())
