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
import csv
import hashlib
import json
import math
import sys
from collections import OrderedDict, defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

TRACE_HEADER = "# moe_otd_trace v1"
INCOMPLETE_PREFIX = "# incomplete "
END_PREFIX = "# end,"
END_FIELDS = ("calls", "experts", "incomplete_calls", "incomplete_experts")
UINT64_MAX = (1 << 64) - 1

# Value reported for any quantity the plugin trace and the run metadata do not provide.
NOT_EXPOSED = "NOT_EXPOSED"


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
    # Begun but never ended (written by TraceWriter::close()); events are a prefix of `unique`.
    incomplete: bool = False


@dataclass
class TraceIntegrity:
    """What is known about how complete a parsed trace is. Never inferred beyond the records."""

    records: int = 0
    # Counts from the "# end" record, or None when the writer never closed the trace.
    end_record: Optional[Dict[str, int]] = None
    incomplete_calls: int = 0
    incomplete_events: int = 0
    # Line number of a truncated final record discarded in salvage mode.
    salvaged_line: Optional[int] = None
    source_sha256: Optional[str] = None
    warnings: List[str] = field(default_factory=list)

    @property
    def complete(self) -> bool:
        # File-writing completeness, not proof that each expert load finished.
        return self.end_record is not None and self.salvaged_line is None

    @property
    def policy_evidence_eligible(self) -> bool:
        # An incomplete/aborted call may have changed real LRU slot state without
        # a matching E record. Such a trace can be inspected and replayed
        # diagnostically, but MUST NOT authorize a production policy decision.
        return self.complete and self.incomplete_calls == 0 and self.incomplete_events == 0

    def as_dict(self) -> dict:
        return {
            "complete": self.complete,
            "policy_evidence_eligible": self.policy_evidence_eligible,
            "end_record": self.end_record is not None,
            "records": self.records,
            "incomplete_calls": self.incomplete_calls,
            "incomplete_events": self.incomplete_events,
            "salvaged_final_record_line": self.salvaged_line,
            "source_sha256": self.source_sha256,
            "warnings": list(self.warnings),
        }


@dataclass
class Trace:
    layers: Dict[int, Layer]
    calls: List[Call]
    integrity: TraceIntegrity = field(default_factory=TraceIntegrity)


class TraceFormatError(ValueError):
    pass


def _uint(text: str, what: str) -> int:
    value = int(text)
    if value < 0 or value > UINT64_MAX:
        raise ValueError(f"{what} {value} is outside the uint64 range")
    return value


def _parse_end(body: str) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for item in body.split(","):
        key, sep, value = item.partition("=")
        if not sep or key not in END_FIELDS or key in counts:
            raise ValueError(f"unexpected end record field {item!r}")
        counts[key] = _uint(value, key)
    if set(counts) != set(END_FIELDS):
        raise ValueError(f"end record lacks {sorted(set(END_FIELDS) - set(counts))}")
    return counts


def _expect_fields(fields: List[str], count: int, tag: str) -> None:
    if len(fields) != count:
        raise ValueError(f"{tag} record has {len(fields)} fields, expected {count}")


def parse_trace(lines: Iterable[str], salvage: bool = False) -> Trace:
    """Parses a v1 trace strictly: malformed, inconsistent or truncated records raise TraceFormatError.

    With salvage=True a truncated final record (the last line lacks the terminator the earlier
    lines have) is discarded and reported in Trace.integrity; nothing else is ever dropped.
    """
    raw_lines = list(lines)
    integrity = TraceIntegrity()
    terminated = any(raw.endswith("\n") for raw in raw_lines[:-1])
    if raw_lines and terminated and not raw_lines[-1].endswith("\n") and raw_lines[-1].strip():
        lineno = len(raw_lines)
        if not salvage:
            raise TraceFormatError(f"line {lineno}: final record is truncated (no line terminator); "
                                   "salvage mode can discard it")
        integrity.salvaged_line = lineno
        integrity.warnings.append(f"discarded truncated final record at line {lineno}")
        raw_lines = raw_lines[:-1]
    elif len(raw_lines) > 1 and not terminated:
        integrity.warnings.append("input lines carry no terminators; a truncated final record cannot be detected")

    layers: Dict[int, Layer] = {}
    calls: List[Call] = []
    incomplete: List[Call] = []
    by_id: Dict[int, Call] = {}
    saw_header = False
    end_lineno: Optional[int] = None
    for lineno, raw in enumerate(raw_lines, 1):
        line = raw.rstrip("\r\n")
        if not line:
            continue
        if end_lineno is not None:
            raise TraceFormatError(f"line {lineno}: record after the end record on line {end_lineno}")
        is_incomplete = line.startswith(INCOMPLETE_PREFIX)
        if is_incomplete:
            line = line[len(INCOMPLETE_PREFIX):]
        elif line.startswith(END_PREFIX):
            try:
                integrity.end_record = _parse_end(line[len(END_PREFIX):])
            except ValueError as exc:
                raise TraceFormatError(f"line {lineno}: malformed end record: {exc}") from exc
            end_lineno = lineno
            continue
        elif line.startswith("#"):
            if line.strip() == TRACE_HEADER:
                if saw_header:
                    raise TraceFormatError(f"line {lineno}: second trace header (concatenated traces?)")
                saw_header = True
            continue
        fields = line.split(",")
        tag = fields[0]
        try:
            if tag == "L" and not is_incomplete:
                _expect_fields(fields, 7, tag)
                layer = Layer(_uint(fields[1], "layer"), fields[2], fields[3], _uint(fields[4], "capacity"),
                              _uint(fields[5], "num_expert"), _uint(fields[6], "top_k"))
                if layer.layer in layers:
                    raise TraceFormatError(f"line {lineno}: duplicate layer {layer.layer}")
                layers[layer.layer] = layer
            elif tag == "C":
                _expect_fields(fields, 8, tag)
                kind, status = fields[4], fields[6]
                if kind not in ("S", "O"):
                    raise TraceFormatError(f"line {lineno}: unknown call kind {kind!r}")
                if status not in (("incomplete",) if is_incomplete else ("ok", "overflow")):
                    raise TraceFormatError(f"line {lineno}: unexpected call status {status!r}")
                experts = [_uint(x, "expert") for x in fields[7].split()]
                if len(set(experts)) != len(experts):
                    raise TraceFormatError(f"line {lineno}: duplicate expert in unique list")
                call = Call(_uint(fields[1], "call"), _uint(fields[2], "t_ns"), _uint(fields[3], "layer"), kind,
                            _uint(fields[5], "routed"), status == "overflow", experts, incomplete=is_incomplete)
                if call.routed < len(experts):
                    raise TraceFormatError(f"line {lineno}: routed {call.routed} is less than {len(experts)} unique experts")
                if call.call in by_id:
                    raise TraceFormatError(f"line {lineno}: duplicate call id {call.call}")
                by_id[call.call] = call
                (incomplete if is_incomplete else calls).append(call)
            elif tag == "E":
                _expect_fields(fields, 13, tag)
                call = by_id.get(_uint(fields[1], "call"))
                if call is None:
                    raise TraceFormatError(f"line {lineno}: E record for unknown call {fields[1]}")
                if call.incomplete != is_incomplete:
                    raise TraceFormatError(f"line {lineno}: E record and call {call.call} disagree on completeness")
                if _uint(fields[3], "layer") != call.layer:
                    raise TraceFormatError(f"line {lineno}: E record layer differs from call {call.call}")
                if call.overflow:
                    raise TraceFormatError(f"line {lineno}: overflowing call {call.call} has expert records")
                if fields[6] not in ("0", "1"):
                    raise TraceFormatError(f"line {lineno}: hit flag {fields[6]!r} is not 0 or 1")
                hit = fields[6] == "1"
                evicted = int(fields[7])
                if evicted < -1 or (hit and evicted != -1):
                    raise TraceFormatError(f"line {lineno}: invalid evicted expert {evicted}")
                call.events.append(
                    Event(
                        _uint(fields[4], "expert"), _uint(fields[5], "slot"), hit, None if evicted < 0 else evicted,
                        _uint(fields[8], "bytes"), _uint(fields[9], "reads"), _uint(fields[10], "read_ns"),
                        _uint(fields[11], "transform_ns"), _uint(fields[12], "copy_ns"),
                    )
                )
            else:
                raise TraceFormatError(f"line {lineno}: unknown record type {tag!r}")
            integrity.records += 1
        except (IndexError, ValueError) as exc:
            if isinstance(exc, TraceFormatError):
                raise
            raise TraceFormatError(f"line {lineno}: malformed {tag} record: {exc}") from exc
    if not saw_header:
        raise TraceFormatError("missing '# moe_otd_trace v1' header")
    for call in calls + incomplete:
        layer = layers.get(call.layer)
        if layer is None:
            raise TraceFormatError(f"call {call.call} references unregistered layer {call.layer}")
        if any(e >= layer.num_expert for e in call.unique):
            raise TraceFormatError(f"call {call.call} routes an expert outside [0, {layer.num_expert})")
        for ev in call.events:
            if ev.expert >= layer.num_expert or (ev.evicted is not None and ev.evicted >= layer.num_expert):
                raise TraceFormatError(f"call {call.call} records an expert outside [0, {layer.num_expert})")
            if ev.slot >= layer.capacity:
                raise TraceFormatError(f"call {call.call} records slot {ev.slot} beyond capacity {layer.capacity}")
            if ev.bytes > 0:
                layer.expert_bytes = max(layer.expert_bytes, ev.bytes)

    integrity.incomplete_calls = len(incomplete)
    integrity.incomplete_events = sum(len(c.events) for c in incomplete)
    if integrity.end_record is not None:
        parsed = {"calls": len(calls), "experts": sum(len(c.events) for c in calls),
                  "incomplete_calls": integrity.incomplete_calls, "incomplete_experts": integrity.incomplete_events}
        if parsed != integrity.end_record:
            raise TraceFormatError(f"end record {integrity.end_record} does not match parsed records {parsed}; records were lost")
    else:
        integrity.warnings.append("no end record: the writer was not closed (killed process, still running, "
                                  "or a writer predating end records); the tail may be missing")
    if incomplete:
        integrity.warnings.append(f"{len(incomplete)} call(s) never completed; they are replayed last, "
                                  "so replay after their real position may diverge")
    if not integrity.policy_evidence_eligible:
        integrity.warnings.append("policy selection blocked: trace lacks complete, reliable call evidence; "
                                  "diagnostic replay is NOT production acceptance")
    _link_fallbacks(calls)
    return Trace(layers, calls + incomplete, integrity)


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
        # `bytes` is kept for compatibility; it is the logical expert payload the plugin would read.
        d["logical_expert_bytes"] = self.bytes
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
        if call.incomplete:
            # Only the recorded prefix of an aborted call reached the cache.
            yield call, [ev.expert for ev in call.events], False
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
        # A simulated policy has no device counter; physical bytes come only from a measured run.
        "physical_nvme_read_bytes": NOT_EXPOSED,
        "trace_integrity": trace.integrity.as_dict(),
    }


def verify_lru(trace: Trace) -> dict:
    """Replays recorded capacities with the exact LRU model and compares every recorded decision."""
    caches = {lid: ExactLRU(layer.capacity) for lid, layer in trace.layers.items()}
    checked = 0
    mismatches: List[dict] = []
    for call in trace.calls:
        cache = caches[call.layer]
        pinned: Set[int] = set()
        # The runtime rejects an overflowing lease before touching the cache; an incomplete call
        # stopped after a prefix of its unique experts.
        if call.incomplete:
            expected = call.unique[:len(call.events)]
        else:
            expected = [] if call.kind == "S" and call.overflow else call.unique
        if call.kind == "S" and not call.incomplete and call.overflow != (len(call.unique) > trace.layers[call.layer].capacity):
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
    return {"checked_events": checked, "mismatches": len(mismatches), "first_mismatches": mismatches[:10],
            "trace_integrity": trace.integrity.as_dict()}


def _quantile(sorted_values: List[int], q: float) -> Optional[int]:
    if not sorted_values:
        return None
    idx = min(len(sorted_values) - 1, max(0, int(math.ceil(q * len(sorted_values))) - 1))
    return sorted_values[idx]


def measured_summary(trace: Trace, run_meta: Optional[dict] = None) -> dict:
    """Aggregates the timings the runtime actually recorded (P2 dashboard input).

    Byte and read counts are logical (requested by the plugin). Physical device bytes are reported
    only when the run metadata carries an OS-level counter; they are never derived from the trace.
    """
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
        "incomplete_calls": trace.integrity.incomplete_calls,
        "logical_expert_bytes": sum(ev.bytes for ev in loads),
        "logical_reads": sum(ev.reads for ev in loads),
        "physical_nvme_read_bytes": _meta_value(run_meta, "physical_nvme_read_bytes"),
        "physical_read_counter_source": _meta_value(run_meta, "physical_read_counter_source"),
        "trace_integrity": trace.integrity.as_dict(),
    }


def _per_token(result: dict, committed_tokens: Optional[int]) -> None:
    if not committed_tokens:
        return
    total = result["total"]
    total["misses_per_committed_token"] = total["misses"] / committed_tokens
    total["bytes_per_committed_token"] = total["bytes"] / committed_tokens if result["bytes_known"] else NOT_EXPOSED
    total["logical_expert_bytes_per_committed_token"] = total["bytes_per_committed_token"]


# --------------------------------------------------------------------------------------------
# Run metadata sidecar ("otd_run_meta v1"): facts about a benchmark run that the trace cannot
# carry. Every field is optional except the schema tag; absent fields are reported NOT_EXPOSED.
# --------------------------------------------------------------------------------------------

RUN_META_SCHEMA = "otd_run_meta v1"
_STR = (str,)
RUN_META_FIELDS: Dict[str, Any] = {
    "trace_sha256": _STR,            # SHA-256 of the trace file this metadata describes
    "openvino_sha": _STR,
    "genai_sha": _STR,
    "ovms_sha": _STR,
    "runtime": _STR,                 # driver / runtime identity string
    "quant": _STR,
    "target_model": _STR,
    "drafter_model": _STR,
    "workload": ("C1", "C2", "C4"),
    "kv_cache_precision": _STR,      # e.g. "u4"
    "context_tokens": int,
    "proposed_tokens": int,          # draft tokens proposed by the drafter
    "accepted_tokens": int,          # draft tokens accepted by the target
    "committed_tokens": int,         # output tokens committed to the response
    "parser_result": ("pass", "fail"),
    "tool_loop_result": ("pass", "fail"),
    "physical_nvme_read_bytes": int,  # from an OS device counter, never from the trace
    "physical_read_counter_source": _STR,
}


class RunMetaError(ValueError):
    pass


def validate_run_meta(meta: Any) -> dict:
    """Returns the metadata with every known field present (None when absent); raises RunMetaError."""
    if not isinstance(meta, dict):
        raise RunMetaError("run metadata must be a JSON object")
    if meta.get("schema") != RUN_META_SCHEMA:
        raise RunMetaError(f"run metadata schema must be {RUN_META_SCHEMA!r}")
    unknown = sorted(set(meta) - set(RUN_META_FIELDS) - {"schema"})
    if unknown:
        raise RunMetaError(f"unknown run metadata fields {unknown}")
    out: Dict[str, Any] = {"schema": RUN_META_SCHEMA}
    for name, rule in RUN_META_FIELDS.items():
        value = meta.get(name)
        if value is not None:
            if rule is int:
                if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                    raise RunMetaError(f"{name} must be a non-negative integer")
            elif rule is _STR:
                if not isinstance(value, str) or not value:
                    raise RunMetaError(f"{name} must be a non-empty string")
            elif value not in rule:
                raise RunMetaError(f"{name} must be one of {list(rule)}")
        out[name] = value
    if out["accepted_tokens"] is not None and out["proposed_tokens"] is not None and out["accepted_tokens"] > out["proposed_tokens"]:
        raise RunMetaError("accepted_tokens exceeds proposed_tokens")
    if (out["physical_nvme_read_bytes"] is None) != (out["physical_read_counter_source"] is None):
        raise RunMetaError("physical_nvme_read_bytes and physical_read_counter_source must be given together")
    return out


def _meta_value(run_meta: Optional[dict], name: str) -> Any:
    value = run_meta.get(name) if run_meta else None
    return NOT_EXPOSED if value is None else value


def _report_meta(run_meta: Optional[dict]) -> dict:
    return {name: _meta_value(run_meta, name) for name in RUN_META_FIELDS}


# --------------------------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------------------------


def _flatten(value: Any, prefix: str, rows: List[Tuple[str, str]]) -> None:
    if isinstance(value, dict):
        for key in value:
            _flatten(value[key], f"{prefix}.{key}" if prefix else str(key), rows)
    elif isinstance(value, list):
        rows.append((prefix, json.dumps(value)))
    else:
        rows.append((prefix, "" if value is None else str(value)))


def write_csv(out: Any, stream) -> None:
    """One row per metric; `result` indexes the entries of a list output (simulate)."""
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(["result", "metric", "value"])
    for index, item in enumerate(out if isinstance(out, list) else [out]):
        rows: List[Tuple[str, str]] = []
        _flatten(item, "", rows)
        for metric, value in rows:
            writer.writerow([index, metric, value])


def read_trace_file(path: str, salvage: bool = False) -> Trace:
    with open(path, "rb") as fh:
        data = fh.read()
    trace = parse_trace(data.decode("utf-8").splitlines(keepends=True), salvage=salvage)
    trace.integrity.source_sha256 = hashlib.sha256(data).hexdigest()
    return trace


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("trace")
    common.add_argument("--salvage", action="store_true",
                        help="discard a truncated final record instead of failing (reported in trace_integrity)")
    common.add_argument("--format", choices=("json", "csv"), default="json")

    p_sum = sub.add_parser("summary", parents=[common], help="aggregate recorded load timings")
    p_sum.add_argument("--run-meta", help=f"{RUN_META_SCHEMA} JSON sidecar of the traced run")

    sub.add_parser("verify", parents=[common], help="check the exact LRU replay against recorded decisions")

    p_sim = sub.add_parser("simulate", parents=[common], help="replay with alternative policies/capacities")
    p_sim.add_argument("--run-meta", help=f"{RUN_META_SCHEMA} JSON sidecar of the traced run")
    p_sim.add_argument("--policy", action="append", choices=POLICIES, help="repeatable; default: all")
    group = p_sim.add_mutually_exclusive_group()
    group.add_argument("--capacity", type=int, action="append", help="slots per layer; repeatable")
    group.add_argument("--capacity-fraction", type=float, action="append", help="fraction of num_expert; repeatable")
    p_sim.add_argument("--legacy-partial-lease", action="store_true",
                       help="model the pre-fix runtime that loads experts of an overflowing lease before rejecting it")
    p_sim.add_argument("--committed-tokens", type=int, help="accepted output tokens of the traced run")

    args = parser.parse_args(argv)
    try:
        trace = read_trace_file(args.trace, salvage=args.salvage)
        run_meta = None
        if getattr(args, "run_meta", None):
            with open(args.run_meta, "r", encoding="utf-8") as fh:
                run_meta = validate_run_meta(json.load(fh))
            if run_meta["trace_sha256"] not in (None, trace.integrity.source_sha256):
                raise RunMetaError("run metadata trace_sha256 does not match the trace file")
            meta_tokens = run_meta["committed_tokens"]
            if args.command == "simulate" and meta_tokens is not None:
                if args.committed_tokens is not None and args.committed_tokens != meta_tokens:
                    raise RunMetaError("--committed-tokens differs from run metadata committed_tokens")
                args.committed_tokens = meta_tokens
    except (TraceFormatError, RunMetaError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        sys.stderr.write(f"otd_sim: {exc}\n")
        return 2

    if args.command == "summary":
        out = measured_summary(trace, run_meta)
        out["run_meta"] = _report_meta(run_meta)
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
                result["run_meta"] = _report_meta(run_meta)
                out.append(result)
    if args.format == "csv":
        write_csv(out, sys.stdout)
    else:
        json.dump(out, sys.stdout, indent=2)
        sys.stdout.write("\n")
    if args.command == "verify" and out["mismatches"]:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
