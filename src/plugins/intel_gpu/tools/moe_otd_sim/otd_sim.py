# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Offline replay of MOE_OTD_TRACE expert-routing traces.

Reads the "moe_otd_trace v1" format written by the Intel GPU plugin
(see moe_otd_trace.hpp), verifies that the exact LRU replay reproduces the
runtime's slot/hit/eviction decisions, and compares cache policies at equal
capacity without touching the runtime.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import OrderedDict, defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

TRACE_HEADER = "# moe_otd_trace v1"


@dataclass
class Layer:
    layer: int
    layer_key: str
    model_key: str
    capacity: int
    num_expert: int
    top_k: int
    expert_bytes: int = 0


@dataclass
class Event:
    expert: int
    slot: int
    hit: bool
    evicted: Optional[int]
    bytes: int
    reads: int
    read_ns: int
    transform_ns: int
    copy_ns: int


@dataclass
class Call:
    call: int
    t_ns: int
    layer: int
    kind: str  # "S" simultaneous lease, "O" single expert
    routed: int
    overflow: bool
    unique: List[int]
    events: List[Event] = field(default_factory=list)
    # For "O" calls produced by the runtime's per-expert fallback after an overflowing "S" call.
    fallback_of: Optional[int] = None


@dataclass
class Trace:
    layers: Dict[int, Layer]
    calls: List[Call]


class TraceFormatError(ValueError):
    pass


def parse_trace(lines: Iterable[str]) -> Trace:
    layers: Dict[int, Layer] = {}
    calls: List[Call] = []
    by_id: Dict[int, Call] = {}
    saw_header = False
    for lineno, raw in enumerate(lines, 1):
        line = raw.rstrip("\r\n")
        if not line:
            continue
        if line.startswith("#"):
            if line.strip() == TRACE_HEADER:
                saw_header = True
            continue
        fields = line.split(",")
        tag = fields[0]
        try:
            if tag == "L":
                layer = Layer(int(fields[1]), fields[2], fields[3], int(fields[4]), int(fields[5]), int(fields[6]))
                layers[layer.layer] = layer
            elif tag == "C":
                experts = [int(x) for x in fields[7].split()] if len(fields) > 7 and fields[7] else []
                call = Call(int(fields[1]), int(fields[2]), int(fields[3]), fields[4], int(fields[5]), fields[6] == "overflow", experts)
                calls.append(call)
                by_id[call.call] = call
            elif tag == "E":
                call = by_id.get(int(fields[1]))
                if call is None:
                    raise TraceFormatError(f"line {lineno}: E record for unknown call {fields[1]}")
                evicted = int(fields[7])
                call.events.append(
                    Event(
                        int(fields[4]), int(fields[5]), fields[6] == "1", None if evicted < 0 else evicted,
                        int(fields[8]), int(fields[9]), int(fields[10]), int(fields[11]), int(fields[12]),
                    )
                )
            else:
                raise TraceFormatError(f"line {lineno}: unknown record type {tag!r}")
        except (IndexError, ValueError) as exc:
            if isinstance(exc, TraceFormatError):
                raise
            raise TraceFormatError(f"line {lineno}: malformed {tag} record: {exc}") from exc
    if not saw_header:
        raise TraceFormatError("missing '# moe_otd_trace v1' header")
    for call in calls:
        if call.layer not in layers:
            raise TraceFormatError(f"call {call.call} references unregistered layer {call.layer}")
        for ev in call.events:
            if ev.bytes > 0:
                layers[call.layer].expert_bytes = max(layers[call.layer].expert_bytes, ev.bytes)
    _link_fallbacks(calls)
    return Trace(layers, calls)


def _link_fallbacks(calls: List[Call]) -> None:
    """Marks the per-expert "O" calls that service an overflowing "S" call on the same layer."""
    pending: Dict[int, Tuple[Call, Set[int]]] = {}
    for call in calls:
        if call.kind == "S":
            pending.pop(call.layer, None)
            if call.overflow:
                pending[call.layer] = (call, set(call.unique))
            continue
        entry = pending.get(call.layer)
        if entry is None:
            continue
        owner, remaining = entry
        if len(call.unique) == 1 and call.unique[0] in remaining:
            call.fallback_of = owner.call
            remaining.discard(call.unique[0])
            if not remaining:
                del pending[call.layer]
        else:
            del pending[call.layer]


# --------------------------------------------------------------------------------------------
# Policies. access() must never evict an expert in `pinned` (already leased by the current call).
# --------------------------------------------------------------------------------------------


class Policy:
    name = "base"

    def __init__(self, capacity: int):
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = capacity

    def access(self, expert: int, pinned: Set[int]) -> Tuple[bool, int, Optional[int]]:
        raise NotImplementedError

    def resident(self) -> Set[int]:
        raise NotImplementedError


class ExactLRU(Policy):
    """Bit-exact model of ov::intel_gpu::ocl::moe::LRUCache including slot numbering."""

    name = "lru"

    def __init__(self, capacity: int):
        super().__init__(capacity)
        self._order: "OrderedDict[int, int]" = OrderedDict()  # expert -> slot, oldest first

    def access(self, expert, pinned):
        slot = self._order.get(expert)
        if slot is not None:
            self._order.move_to_end(expert)
            return True, slot, None
        evicted = None
        if len(self._order) >= self.capacity:
            evicted, slot = self._order.popitem(last=False)
        else:
            slot = len(self._order)
        self._order[expert] = slot
        return False, slot, evicted

    def resident(self):
        return set(self._order)


class _SlotPool:
    def __init__(self, capacity: int):
        self._free = list(range(capacity - 1, -1, -1))
        self.slot_of: Dict[int, int] = {}

    def take(self, expert: int, from_expert: Optional[int] = None) -> int:
        slot = self.slot_of.pop(from_expert) if from_expert is not None else self._free.pop()
        self.slot_of[expert] = slot
        return slot


class Clock(Policy):
    name = "clock"

    def __init__(self, capacity: int):
        super().__init__(capacity)
        self._ring: List[int] = []
        self._ref: Dict[int, bool] = {}
        self._hand = 0
        self._slots = _SlotPool(capacity)

    def access(self, expert, pinned):
        if expert in self._ref:
            self._ref[expert] = True
            return True, self._slots.slot_of[expert], None
        if len(self._ring) < self.capacity:
            self._ring.append(expert)
            self._ref[expert] = False
            return False, self._slots.take(expert), None
        for _ in range(2 * len(self._ring) + 1):
            victim = self._ring[self._hand]
            if victim in pinned:
                self._hand = (self._hand + 1) % len(self._ring)
                continue
            if self._ref[victim]:
                self._ref[victim] = False
                self._hand = (self._hand + 1) % len(self._ring)
                continue
            del self._ref[victim]
            self._ring[self._hand] = expert
            self._ref[expert] = False
            self._hand = (self._hand + 1) % len(self._ring)
            return False, self._slots.take(expert, victim), victim
        raise RuntimeError("clock: every resident expert is pinned")

    def resident(self):
        return set(self._ref)


class TwoQ(Policy):
    """Simplified 2Q: probationary FIFO (A1in), protected LRU (Am) and a ghost list (A1out)."""

    name = "2q"

    def __init__(self, capacity: int, in_fraction: float = 0.25, ghost_fraction: float = 0.5):
        super().__init__(capacity)
        self._kin = max(1, int(capacity * in_fraction))
        self._kout = max(1, int(capacity * ghost_fraction))
        self._a1in: "OrderedDict[int, None]" = OrderedDict()
        self._am: "OrderedDict[int, None]" = OrderedDict()
        self._a1out: "OrderedDict[int, None]" = OrderedDict()
        self._slots = _SlotPool(capacity)

    def _victim(self, pinned):
        prefer_in = len(self._a1in) > self._kin or not self._am
        queues = (self._a1in, self._am) if prefer_in else (self._am, self._a1in)
        for queue in queues:
            for expert in queue:
                if expert not in pinned:
                    return queue, expert
        raise RuntimeError("2q: every resident expert is pinned")

    def access(self, expert, pinned):
        if expert in self._am:
            self._am.move_to_end(expert)
            return True, self._slots.slot_of[expert], None
        if expert in self._a1in:
            return True, self._slots.slot_of[expert], None
        evicted = None
        if len(self._a1in) + len(self._am) >= self.capacity:
            queue, evicted = self._victim(pinned)
            del queue[evicted]
            if queue is self._a1in:
                self._a1out[evicted] = None
                while len(self._a1out) > self._kout:
                    self._a1out.popitem(last=False)
            slot = self._slots.take(expert, evicted)
        else:
            slot = self._slots.take(expert)
        if expert in self._a1out:
            del self._a1out[expert]
            self._am[expert] = None
        else:
            self._a1in[expert] = None
        return False, slot, evicted

    def resident(self):
        return set(self._a1in) | set(self._am)


class Belady(Policy):
    """Offline optimum (evict the resident expert reused furthest in the future). Upper bound only."""

    name = "opt"

    def __init__(self, capacity: int, future: Sequence[int]):
        super().__init__(capacity)
        self._future = future
        self._pos = 0
        self._next_use: Dict[int, deque] = defaultdict(deque)
        for i, expert in enumerate(future):
            self._next_use[expert].append(i)
        self._resident: Set[int] = set()
        self._slots = _SlotPool(capacity)

    def access(self, expert, pinned):
        if self._pos >= len(self._future) or self._future[self._pos] != expert:
            raise RuntimeError("opt: access sequence diverged from the precomputed future")
        uses = self._next_use[expert]
        uses.popleft()
        self._pos += 1
        if expert in self._resident:
            return True, self._slots.slot_of[expert], None
        evicted = None
        if len(self._resident) >= self.capacity:
            candidates = [e for e in self._resident if e not in pinned]
            if not candidates:
                raise RuntimeError("opt: every resident expert is pinned")
            evicted = max(candidates, key=lambda e: self._next_use[e][0] if self._next_use[e] else math.inf)
            self._resident.discard(evicted)
            slot = self._slots.take(expert, evicted)
        else:
            slot = self._slots.take(expert)
        self._resident.add(expert)
        return False, slot, evicted

    def resident(self):
        return set(self._resident)


POLICIES = ("lru", "clock", "2q", "opt")


# --------------------------------------------------------------------------------------------
# Replay
# --------------------------------------------------------------------------------------------


def phase_of(call: Call, layer: Layer) -> str:
    if call.kind == "O":
        return "fallback" if call.fallback_of is not None else "per_expert"
    tokens = call.routed / layer.top_k if layer.top_k else call.routed
    if tokens <= 1:
        return "decode_1tok"
    if tokens <= 8:
        return "multi_2to8tok"
    return "prefill_gt8tok"


@dataclass
class Counters:
    accesses: int = 0
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    bytes: int = 0
    lease_calls: int = 0
    overflow_calls: int = 0
    overflow_partial_loads: int = 0

    def add_access(self, hit: bool, evicted: Optional[int], expert_bytes: int) -> None:
        self.accesses += 1
        if hit:
            self.hits += 1
        else:
            self.misses += 1
            self.bytes += expert_bytes
        if evicted is not None:
            self.evictions += 1

    def as_dict(self) -> dict:
        d = dict(self.__dict__)
        d["hit_rate"] = self.hits / self.accesses if self.accesses else None
        return d


def _logical_steps(trace: Trace, capacities: Dict[int, int], legacy_partial_lease: bool):
    """Yields (call, experts_to_access, overflowed) in replay order for the given capacities.

    Recorded fallback calls are skipped when the simulated lease succeeds; when a simulated lease
    overflows without a recorded fallback, the per-expert fallback is synthesized in sorted order.
    """
    fallbacks: Dict[int, List[Call]] = defaultdict(list)
    for call in trace.calls:
        if call.fallback_of is not None:
            fallbacks[call.fallback_of].append(call)
    for call in trace.calls:
        if call.fallback_of is not None:
            continue
        cap = capacities[call.layer]
        if call.kind == "O":
            yield call, call.unique, False
            continue
        overflow = len(call.unique) > cap
        if not overflow:
            yield call, call.unique, False
            continue
        yield call, (call.unique[:cap] if legacy_partial_lease else []), True
        recorded = fallbacks.get(call.call)
        if recorded:
            for fb in recorded:
                yield fb, fb.unique, False
        else:
            for expert in sorted(call.unique):
                synthetic = Call(-1, call.t_ns, call.layer, "O", 1, False, [expert], fallback_of=call.call)
                yield synthetic, [expert], False


def replay(trace: Trace,
           policy: str,
           capacity: Optional[int] = None,
           capacity_fraction: Optional[float] = None,
           legacy_partial_lease: bool = False) -> dict:
    capacities: Dict[int, int] = {}
    for lid, layer in trace.layers.items():
        if capacity is not None:
            capacities[lid] = capacity
        elif capacity_fraction is not None:
            capacities[lid] = max(1, int(round(layer.num_expert * capacity_fraction)))
        else:
            capacities[lid] = layer.capacity

    steps = list(_logical_steps(trace, capacities, legacy_partial_lease))
    caches: Dict[int, Policy] = {}
    for lid in trace.layers:
        if policy == "opt":
            future = [e for call, experts, _ in steps if call.layer == lid for e in experts]
            caches[lid] = Belady(capacities[lid], future)
        elif policy == "lru":
            caches[lid] = ExactLRU(capacities[lid])
        elif policy == "clock":
            caches[lid] = Clock(capacities[lid])
        elif policy == "2q":
            caches[lid] = TwoQ(capacities[lid])
        else:
            raise ValueError(f"unknown policy {policy!r}")

    total = Counters()
    by_phase: Dict[str, Counters] = defaultdict(Counters)
    by_model: Dict[str, Counters] = defaultdict(Counters)
    for call, experts, overflowed in steps:
        layer = trace.layers[call.layer]
        cache = caches[call.layer]
        phase = phase_of(call, layer)
        pinned: Set[int] = set()
        loads = 0
        for expert in experts:
            hit, _slot, evicted = cache.access(expert, pinned)
            pinned.add(expert)
            loads += 0 if hit else 1
            for c in (total, by_phase[phase], by_model[layer.model_key]):
                c.add_access(hit, evicted, layer.expert_bytes)
        if call.kind == "S":
            for c in (total, by_phase[phase], by_model[layer.model_key]):
                c.lease_calls += 1
                if overflowed:
                    c.overflow_calls += 1
                    c.overflow_partial_loads += loads

    return {
        "policy": policy,
        "legacy_partial_lease": legacy_partial_lease,
        "capacity": capacity,
        "capacity_fraction": capacity_fraction,
        "total": total.as_dict(),
        "by_phase": {k: v.as_dict() for k, v in sorted(by_phase.items())},
        "by_model": {k: v.as_dict() for k, v in sorted(by_model.items())},
        "bytes_known": all(l.expert_bytes > 0 for l in trace.layers.values()),
    }


def verify_lru(trace: Trace) -> dict:
    """Replays recorded capacities with the exact LRU model and compares every recorded decision."""
    caches = {lid: ExactLRU(layer.capacity) for lid, layer in trace.layers.items()}
    checked = 0
    mismatches: List[dict] = []
    for call in trace.calls:
        cache = caches[call.layer]
        pinned: Set[int] = set()
        # The runtime rejects an overflowing lease before touching the cache.
        expected = [] if call.kind == "S" and call.overflow else call.unique
        if call.kind == "S" and call.overflow != (len(call.unique) > trace.layers[call.layer].capacity):
            mismatches.append({"call": call.call, "reason": "overflow status does not match capacity"})
        if [ev.expert for ev in call.events] != list(expected):
            mismatches.append({"call": call.call, "reason": "recorded experts differ from lease order",
                               "recorded": [ev.expert for ev in call.events], "expected": list(expected)})
            continue
        for ev in call.events:
            hit, slot, evicted = cache.access(ev.expert, pinned)
            pinned.add(ev.expert)
            checked += 1
            if (hit, slot, evicted) != (ev.hit, ev.slot, ev.evicted):
                mismatches.append({"call": call.call, "expert": ev.expert,
                                   "recorded": [ev.hit, ev.slot, ev.evicted], "replayed": [hit, slot, evicted]})
    return {"checked_events": checked, "mismatches": len(mismatches), "first_mismatches": mismatches[:10]}


def _quantile(sorted_values: List[int], q: float) -> Optional[int]:
    if not sorted_values:
        return None
    idx = min(len(sorted_values) - 1, max(0, int(math.ceil(q * len(sorted_values))) - 1))
    return sorted_values[idx]


def measured_summary(trace: Trace) -> dict:
    """Aggregates the timings the runtime actually recorded (P2 dashboard input)."""
    loads = [ev for call in trace.calls for ev in call.events if not ev.hit]
    hits = sum(1 for call in trace.calls for ev in call.events if ev.hit)
    per_load_ns = sorted(ev.read_ns + ev.transform_ns + ev.copy_ns for ev in loads)
    read_ns = sum(ev.read_ns for ev in loads)
    transform_ns = sum(ev.transform_ns for ev in loads)
    copy_ns = sum(ev.copy_ns for ev in loads)
    busy = read_ns + transform_ns + copy_ns
    span_ns = (trace.calls[-1].t_ns - trace.calls[0].t_ns) if len(trace.calls) > 1 else 0
    return {
        "layers": len(trace.layers),
        "models": len({l.model_key for l in trace.layers.values()}),
        "calls": len(trace.calls),
        "overflow_calls": sum(1 for c in trace.calls if c.kind == "S" and c.overflow),
        "fallback_calls": sum(1 for c in trace.calls if c.fallback_of is not None),
        "hits": hits,
        "loads": len(loads),
        "bytes": sum(ev.bytes for ev in loads),
        "reads": sum(ev.reads for ev in loads),
        "read_ms": read_ns / 1e6,
        "transform_ms": transform_ns / 1e6,
        "copy_ms": copy_ns / 1e6,
        "share_read": read_ns / busy if busy else None,
        "share_transform": transform_ns / busy if busy else None,
        "share_copy": copy_ns / busy if busy else None,
        "load_us_p50": _quantile(per_load_ns, 0.50) / 1e3 if per_load_ns else None,
        "load_us_p95": _quantile(per_load_ns, 0.95) / 1e3 if per_load_ns else None,
        "load_us_p99": _quantile(per_load_ns, 0.99) / 1e3 if per_load_ns else None,
        "trace_span_ms": span_ns / 1e6,
    }


def _per_token(result: dict, committed_tokens: Optional[int]) -> None:
    if not committed_tokens:
        return
    total = result["total"]
    total["misses_per_committed_token"] = total["misses"] / committed_tokens
    total["bytes_per_committed_token"] = total["bytes"] / committed_tokens if result["bytes_known"] else "NOT_EXPOSED"


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p_sum = sub.add_parser("summary", help="aggregate recorded load timings")
    p_sum.add_argument("trace")

    p_ver = sub.add_parser("verify", help="check the exact LRU replay against recorded decisions")
    p_ver.add_argument("trace")

    p_sim = sub.add_parser("simulate", help="replay with alternative policies/capacities")
    p_sim.add_argument("trace")
    p_sim.add_argument("--policy", action="append", choices=POLICIES, help="repeatable; default: all")
    group = p_sim.add_mutually_exclusive_group()
    group.add_argument("--capacity", type=int, action="append", help="slots per layer; repeatable")
    group.add_argument("--capacity-fraction", type=float, action="append", help="fraction of num_expert; repeatable")
    p_sim.add_argument("--legacy-partial-lease", action="store_true",
                       help="model the pre-fix runtime that loads experts of an overflowing lease before rejecting it")
    p_sim.add_argument("--committed-tokens", type=int, help="accepted output tokens of the traced run")

    args = parser.parse_args(argv)
    with open(args.trace, "r", encoding="utf-8") as fh:
        trace = parse_trace(fh)

    if args.command == "summary":
        out = measured_summary(trace)
    elif args.command == "verify":
        out = verify_lru(trace)
    else:
        policies = args.policy or list(POLICIES)
        sizes: List[Tuple[Optional[int], Optional[float]]]
        if args.capacity:
            sizes = [(c, None) for c in args.capacity]
        elif args.capacity_fraction:
            sizes = [(None, f) for f in args.capacity_fraction]
        else:
            sizes = [(None, None)]
        out = []
        for cap, frac in sizes:
            for policy in policies:
                result = replay(trace, policy, cap, frac, args.legacy_partial_lease)
                _per_token(result, args.committed_tokens)
                out.append(result)
    json.dump(out, sys.stdout, indent=2)
    sys.stdout.write("\n")
    if args.command == "verify" and out["mismatches"]:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
