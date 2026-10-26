"""Cross-lane sharing of loop work that does not depend on a thread's lane axes.

In Emmy's attention schedule every thread computes the scores q[row].k[key] for its own
output columns, so G adjacent lanes that share a row compute the same scores. For a key
loop L, the statements that depend on the key but not on the lane axes (nor on state
carried across keys) are split off as the distributed part. Keys run in chunks of G: lane
i computes the distributed part for key chunk*G + i, then for each key of the chunk every
lane receives those values with shfl.sync.idx and runs the rest of the body. Keys keep
their order and every value is computed by the same instructions, so results are
bit-identical; the distributed work drops G times.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

LOOPS = ("Loop", "StridedLoop")
PURE = ("Load", "Assign", "Let", "Accum", "Loop", "StridedLoop")


def kind(s) -> str:
    return type(s).__name__


def _free(e) -> set[str]:
    return set(e.free_vars()) if hasattr(e, "free_vars") else set()


def defs_uses(s) -> tuple[set[str], set[str]]:
    """Names a statement (with everything nested in it) defines and reads from outside."""
    k = kind(s)
    if k == "Load":
        return set(s.names), set().union(*map(_free, s.index)) if s.index else set()
    if k == "Assign":
        return {s.name}, set(s.args)
    if k == "Let":
        return {s.name}, _free(s.value)
    if k == "Accum":
        return {s.name}, {s.value, s.name} | ({s.base} if s.base else set())
    if k == "Init":
        return {s.name}, set()
    if k == "Select":
        return {s.name}, set().union(*({b.value} | _free(b.select) for b in s.branches))
    if k == "Write":
        return set(), set(s.values) | set().union(*map(_free, s.index)) if s.index else set(s.values)
    if k in LOOPS or k == "Cond":
        d, u = set(), set()
        if k == "StridedLoop":
            u |= _free(s.start) | _free(s.step) | (_free(s.end) if s.end is not None else set())
        if k == "Cond":
            u |= _free(s.cond)
        inner = set()
        if k in LOOPS:
            inner.add(s.axis.name)
            if s.seed:  # the loop seeds its own accumulators
                d |= {st.name for st in s.body if kind(st) == "Accum"}
        for st in list(s.body) + list(getattr(s, "else_body", ()) or ()):
            dd, uu = defs_uses(st)
            u |= uu - d - inner
            d |= dd
        return d, u
    return set(), set()


def _pure_loop(s, readonly: set[str]) -> bool:
    """A nested loop we can run for a different key: loads from read-only buffers, arithmetic,
    and accumulators the loop seeds itself."""
    for st in s.body:
        k = kind(st)
        if k not in PURE:
            return False
        if k == "Load" and st.input not in readonly:
            return False
        if k in LOOPS and not (st.seed and _pure_loop(st, readonly)):
            return False
    return s.seed


@dataclass
class Plan:
    group: int  # lanes sharing the work (a power of two <= 32)
    lane_axes: tuple[str, ...]
    distributed: list[int]  # indices into the loop body
    interface: list[str]  # names the rest of the body reads from the distributed part
    chunks: int  # full chunks of G keys
    tail: int  # keys left over after the full chunks


def plan(loop, tile_axes: list[tuple[str, int]], before, readonly: set[str], block: int) -> Plan | None:
    """``tile_axes``: (name, extent) outermost first; ``before``: the Tile statements ahead of the loop."""
    if kind(loop) == "Loop":
        extent = int(loop.axis.extent.as_static()) if hasattr(loop.axis.extent, "as_static") else int(loop.axis.extent)
    else:
        return None
    key = loop.axis.name
    body = list(loop.body)
    info = [defs_uses(s) for s in body]
    carried = {s.name for s in body if kind(s) in ("Accum", "Init")}

    def split(lane_axes: set[str]):
        dep = lane_dependent(before, lane_axes) | lane_axes
        laneish, carry, keyed, cand = set(), set(carried), {key}, []
        for i, (s, (d, u)) in enumerate(zip(body, info)):
            k = kind(s)
            is_lane = bool(u & (dep | laneish))
            is_carry = bool(u & carry) or k in ("Accum", "Init", "Write", "Cond", "Select") or (k in LOOPS and not _pure_loop(s, readonly))
            is_key = bool(u & keyed)
            if k == "Load" and s.input not in readonly:
                is_carry = True
            if is_lane:
                laneish |= d
            if is_carry:
                carry |= d
            if is_key:
                keyed |= d
            cand.append(not is_lane and not is_carry and is_key)
        # keep a candidate only if it is heavy (a loop or a load) or feeds only distributed statements
        users = {i: [j for j in range(len(body)) if info[i][0] & info[j][1]] for i in range(len(body))}
        dist = [c and kind(body[i]) in ("Load",) + LOOPS for i, c in enumerate(cand)]
        changed = True
        while changed:
            changed = False
            for i, c in enumerate(cand):
                if c and not dist[i] and users[i] and all(dist[j] for j in users[i]):
                    dist[i] = changed = True
        return [i for i, x in enumerate(dist) if x]

    # Lane axes: the innermost Tile axes (extent-1 axes are constant). A group of G consecutive
    # lanes, G a power of two dividing the axes' extent, agrees on every other axis.
    lane_axes, group, best = [], 1, None
    for name, ext in reversed(tile_axes):
        if ext == 1:
            continue
        take = 1
        while ext % (take * 2) == 0 and group * take * 2 <= 32:
            take *= 2
        if take == 1 or block % (group * take):
            break
        trial = split(set(lane_axes) | {name})
        if not any(kind(body[i]) in LOOPS for i in trial):
            break
        lane_axes.append(name)
        group *= take
        best = trial
        if take < ext:  # only part of this axis fits: the axes outside it differ within a warp
            break
    if best is None or group < 2:
        return None
    dist = set(best)
    interface = sorted({n for i in dist for n in info[i][0]} & set().union(*(info[j][1] for j in range(len(body)) if j not in dist)))
    return Plan(group, tuple(lane_axes), sorted(dist), interface, extent // group, extent % group)


def lane_dependent(stmts, lane_axes: set[str]) -> set[str]:
    """Names defined by ``stmts`` (before the loop) that vary with the given axes or with the lane."""
    dep = set(lane_axes) | {"lane", "warp"}
    out = set()
    for s in stmts:
        d, u = defs_uses(s)
        if u & (dep | out):
            out |= d
    return out
