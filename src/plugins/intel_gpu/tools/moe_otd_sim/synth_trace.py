# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Deterministic synthetic MOE_OTD_TRACE generator.

Produces traces with the same record semantics as the runtime (exact LRU slots,
overflowing leases rejected untouched and followed by sorted per-expert fallback) for testing the
simulator and for what-if studies before real router traces exist. Routing is a
per-layer Zipf distribution over a per-session permutation, so concurrent
sessions widen the expert union the way independent requests do.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from typing import List, Optional, TextIO

from otd_sim import ExactLRU, TRACE_HEADER


@dataclass
class SynthConfig:
    layers: int = 4
    num_expert: int = 128
    top_k: int = 8
    capacity: int = 32
    expert_bytes: int = 3 * 1024 * 1024
    sessions: int = 1
    prompt_tokens: int = 64
    output_tokens: int = 64
    # 0 = plain decode; N = verify windows of N draft positions plus the bonus token.
    spec_window: int = 0
    accept_prob: float = 0.7
    zipf_s: float = 1.1
    session_overlap: float = 0.5
    seed: int = 1234


class _Router:
    def __init__(self, cfg: SynthConfig, rng: random.Random):
        self.cfg = cfg
        self.rng = rng
        base = [list(range(cfg.num_expert)) for _ in range(cfg.layers)]
        for perm in base:
            rng.shuffle(perm)
        self.session_perm = []
        for _ in range(cfg.sessions):
            per_layer = []
            for perm in base:
                p = list(perm)
                swaps = int(len(p) * (1.0 - cfg.session_overlap))
                for _ in range(swaps):
                    i, j = rng.randrange(len(p)), rng.randrange(len(p))
                    p[i], p[j] = p[j], p[i]
                per_layer.append(p)
            self.session_perm.append(per_layer)
        weights = [1.0 / ((rank + 1) ** cfg.zipf_s) for rank in range(cfg.num_expert)]
        total = sum(weights)
        self.cdf = []
        acc = 0.0
        for w in weights:
            acc += w / total
            self.cdf.append(acc)

    def _rank(self) -> int:
        u = self.rng.random()
        lo, hi = 0, len(self.cdf) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if self.cdf[mid] < u:
                lo = mid + 1
            else:
                hi = mid
        return lo

    def topk(self, session: int, layer: int) -> List[int]:
        perm = self.session_perm[session][layer]
        chosen: List[int] = []
        while len(chosen) < self.cfg.top_k:
            e = perm[self._rank()]
            if e not in chosen:
                chosen.append(e)
        return chosen


class _Emitter:
    def __init__(self, cfg: SynthConfig, out: TextIO):
        self.cfg = cfg
        self.out = out
        self.call = 0
        self.t_ns = 0
        self.caches = [ExactLRU(cfg.capacity) for _ in range(cfg.layers)]
        out.write(TRACE_HEADER + "\n")
        for layer in range(cfg.layers):
            out.write(f"L,{layer},{layer:016x},{0xC0FFEE:016x},{cfg.capacity},{cfg.num_expert},{cfg.top_k}\n")

    def _event(self, call: int, layer: int, expert: int, pinned: set) -> str:
        hit, slot, evicted = self.caches[layer].access(expert, pinned)
        pinned.add(expert)
        self.t_ns += 1000
        if hit:
            return f"E,{call},{self.t_ns},{layer},{expert},{slot},1,-1,0,0,0,0,0\n"
        self.t_ns += 50_000
        ev = -1 if evicted is None else evicted
        return f"E,{call},{self.t_ns},{layer},{expert},{slot},0,{ev},{self.cfg.expert_bytes},9,40000,6000,4000\n"

    def lease(self, layer: int, routed_ids: List[int]) -> None:
        unique: List[int] = []
        for e in routed_ids:
            if e not in unique:
                unique.append(e)
        call = self.call
        self.call += 1
        start = self.t_ns
        pinned: set = set()
        overflow = len(unique) > self.cfg.capacity
        events = [] if overflow else [self._event(call, layer, e, pinned) for e in unique]
        self.out.write(f"C,{call},{start},{layer},S,{len(routed_ids)},{'overflow' if overflow else 'ok'},{' '.join(map(str, unique))}\n")
        self.out.writelines(events)
        if overflow:
            for e in sorted(unique):
                one = self.call
                self.call += 1
                t = self.t_ns
                line = self._event(one, layer, e, set())
                self.out.write(f"C,{one},{t},{layer},O,1,ok,{e}\n")
                self.out.write(line)


def generate(cfg: SynthConfig, out: TextIO) -> dict:
    rng = random.Random(cfg.seed)
    router = _Router(cfg, rng)
    emit = _Emitter(cfg, out)

    def step(tokens_per_session: List[int]) -> None:
        for layer in range(cfg.layers):
            routed: List[int] = []
            for session, tokens in enumerate(tokens_per_session):
                for _ in range(tokens):
                    routed.extend(router.topk(session, layer))
            if routed:
                emit.lease(layer, routed)

    step([cfg.prompt_tokens] * cfg.sessions)
    committed = [0] * cfg.sessions
    proposed = rejected = rounds = 0
    while min(committed) < cfg.output_tokens:
        tokens = []
        for s in range(cfg.sessions):
            if committed[s] >= cfg.output_tokens:
                tokens.append(0)
                continue
            if cfg.spec_window == 0:
                tokens.append(1)
                committed[s] += 1
                continue
            accepted = 0
            while accepted < cfg.spec_window and rng.random() < cfg.accept_prob:
                accepted += 1
            tokens.append(cfg.spec_window + 1)
            proposed += cfg.spec_window
            rejected += cfg.spec_window - accepted
            committed[s] = min(cfg.output_tokens, committed[s] + accepted + 1)
        rounds += 1
        step(tokens)
    return {
        "committed_tokens": sum(committed),
        "verify_rounds": rounds,
        "proposed_tokens": proposed,
        "rejected_tokens": rejected,
        "config": cfg.__dict__,
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("output", help="trace file to write")
    parser.add_argument("--meta", help="write run metadata (committed tokens etc.) as JSON")
    for name, value in SynthConfig().__dict__.items():
        parser.add_argument("--" + name.replace("_", "-"), type=type(value), default=value)
    args = parser.parse_args(argv)
    cfg = SynthConfig(**{k: getattr(args, k) for k in SynthConfig().__dict__})
    with open(args.output, "w", encoding="utf-8", newline="\n") as fh:
        meta = generate(cfg, fh)
    if args.meta:
        with open(args.meta, "w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2)
    json.dump({k: v for k, v in meta.items() if k != "config"}, sys.stdout)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
