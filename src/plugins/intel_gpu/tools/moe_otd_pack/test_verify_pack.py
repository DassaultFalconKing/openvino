# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Deterministic disk-format regression fixtures, no GPU or model weights."""
import io
import json
import os
from pathlib import Path
import struct
import tempfile
import unittest
from contextlib import redirect_stdout

from verify_pack import (ALIGNMENT, FNV_OFFSET, HEADER, MAGIC, VERSION, PackError,
                         align_up, canonical_layout, fnv_update, inspect_pack, main)

KEY = 0x12345ABCDEF98765
SIZES = [33, 17, 35, 3, 5, 7, 2, 1, 0]
N = 6


def fixture(root: Path, placement=None, sizes=None) -> Path:
    sizes = SIZES if sizes is None else sizes
    placement = list(reversed(range(N))) if placement is None else placement
    offsets, payload, stride, ck_offset, records_offset = canonical_layout(sizes, N)
    header_fields = [MAGIC, VERSION, ALIGNMENT, KEY, N, stride, records_offset, ALIGNMENT, ck_offset,
                     *sizes, *offsets, 0]
    header = HEADER.pack(*header_fields)
    header_fields[-1] = fnv_update(FNV_OFFSET, header)
    header = HEADER.pack(*header_fields)
    output = root / f"otd-{KEY:016x}.pack"
    checksums = [0] * N
    records = [bytearray(stride) for _ in range(N)]
    for expert in range(N):
        record = records[placement[expert]]
        for tensor, size in enumerate(sizes):
            record[offsets[tensor]:offsets[tensor] + size] = bytes(
                (expert * 31 + tensor * 17 + i) % 256 for i in range(size))
        checksums[expert] = fnv_update(FNV_OFFSET, record[:payload])
    with output.open("wb") as fh:
        fh.write(bytearray(records_offset))
        for record in records:
            fh.write(record)
        fh.seek(0)
        fh.write(header)
        fh.seek(ALIGNMENT)
        fh.write(struct.pack("<" + "I" * N, *placement))
        fh.seek(ck_offset)
        fh.write(struct.pack("<" + "Q" * N, *checksums))
    return output


def tamper(path: Path, offset: int, data: bytes):
    with path.open("r+b") as fh:
        fh.seek(offset)
        fh.write(data)


def reseal(path: Path, update):
    with path.open("r+b") as fh:
        h = list(HEADER.unpack(fh.read(HEADER.size)))
        update(h)
        h[-1] = 0
        h[-1] = fnv_update(FNV_OFFSET, HEADER.pack(*h))
        fh.seek(0)
        fh.write(HEADER.pack(*h))


class PackVerificationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.path = fixture(self.root)

    def test_valid_permuted_placement(self):
        a = inspect_pack(self.path, expect_key=KEY, expect_experts=N)
        self.assertEqual((a["status"], a["checked_experts"], a["corrupt_experts"]), ("OK", N, []))
        self.assertEqual(a["record_stride"] % ALIGNMENT, 0)
        self.assertEqual(a["physical_nvme_read_bytes"], "NOT_EXPOSED")

    def test_first_and_last_expert_bitflips(self):
        h = HEADER.unpack(self.path.read_bytes()[:HEADER.size])
        placement_offset, records_offset, stride = h[7], h[6], h[5]
        with self.path.open("rb") as fh:
            fh.seek(placement_offset)
            placement = struct.unpack("<" + "I" * N, fh.read(4 * N))
        for expert in (0, N - 1):
            offset = records_offset + placement[expert] * stride
            with self.path.open("r+b") as fh:
                fh.seek(offset)
                byte = fh.read(1)
            tamper(self.path, offset, bytes([byte[0] ^ 0x40]))
        self.assertEqual(inspect_pack(self.path)["corrupt_experts"], [0, N - 1])

    def test_bitflip_in_alignment_padding_not_hashed(self):
        h = HEADER.unpack(self.path.read_bytes()[:HEADER.size])
        payload = canonical_layout(SIZES, N)[1]
        tamper(self.path, h[6] + payload, b"Z")
        self.assertEqual(inspect_pack(self.path)["status"], "OK")

    def test_corrupt_checksum_table(self):
        h = HEADER.unpack(self.path.read_bytes()[:HEADER.size])
        tamper(self.path, h[8], b"\xff")
        self.assertEqual(inspect_pack(self.path)["corrupt_experts"], [0])

    def test_duplicate_placement_rejected(self):
        tamper(self.path, ALIGNMENT, struct.pack("<II", 0, 0))
        with self.assertRaisesRegex(PackError, "permutation"):
            inspect_pack(self.path)

    def test_invalid_header_fingerprint(self):
        tamper(self.path, 9, b"\x98")
        with self.assertRaisesRegex(PackError, "magic/version|checksum"):
            inspect_pack(self.path)

    def test_resealed_noncanonical_header(self):
        reseal(self.path, lambda h: h.__setitem__(6, h[6] + ALIGNMENT))
        with self.assertRaisesRegex(PackError, "non-canonical"):
            inspect_pack(self.path)

    def test_truncated_file_rejected(self):
        with self.path.open("r+b") as fh:
            fh.truncate(self.path.stat().st_size - 1)
        with self.assertRaisesRegex(PackError, "truncated"):
            inspect_pack(self.path)

    def test_wrong_model_key_rejected(self):
        with self.assertRaisesRegex(PackError, "trusted key"):
            inspect_pack(self.path, expect_key=1)

    def test_renamed_file_rejected(self):
        moved = self.path.rename(self.root / "not-otd.pack")
        with self.assertRaisesRegex(PackError, "filename"):
            inspect_pack(moved)

    def test_metadata_allocation_bound(self):
        with self.assertRaisesRegex(PackError, "safety bound"):
            inspect_pack(self.path, max_experts=4)

    def test_large_payload_streaming(self):
        sizes = [2 * (1 << 20) + 1] + [0] * 8
        self.path = fixture(self.root, sizes=sizes)
        self.assertEqual(inspect_pack(self.path)["status"], "OK")

    def test_trailing_bytes_are_reported_not_failed(self):
        with self.path.open("ab") as fh:
            fh.write(b"EXTRA")
        r = inspect_pack(self.path)
        self.assertEqual((r["status"], r["trailing_bytes"]), ("OK", 5))

    def test_cli_json_and_exit_codes(self):
        out = io.StringIO()
        with redirect_stdout(out):
            code = main([str(self.path), "--json", "--expect-key", hex(KEY)])
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(out.getvalue())["status"], "OK")
        h = HEADER.unpack(self.path.read_bytes()[:HEADER.size])
        tamper(self.path, h[8], b"\xff")
        with redirect_stdout(io.StringIO()):
            self.assertEqual(main([str(self.path)]), 1)
        with redirect_stdout(io.StringIO()):
            self.assertEqual(main([str(self.path), "--expect-key", "0x1"]), 2)


if __name__ == "__main__":
    unittest.main()
