"""Merge adjacent aligned scalar loads into .v2/.v4 loads.

ptxas cannot do this because it cannot prove a base register's alignment; codegen records it.
Only loads in one basic block are merged, and for shared memory only with no store, barrier
or async copy between them.
"""

from __future__ import annotations

import re

_LOAD = re.compile(r"^\tld\.(global\.nc|shared)\.(f32|f64|s32|u32|b32)\t(%\w+), \[(%\w+)(?:\+(-?\d+))?\];$")
_BARRIER = re.compile(r"^(\$\S+:|\t(@\S+ )?(bra|ret|bar|st\.|red\.|atom\.|cp\.async|membar|fence))")
_WIDTH = {"f32": 4, "f64": 8, "s32": 4, "u32": 4, "b32": 4}


def _block_safe(lines: list[str], lo: int, hi: int, space: str) -> bool:
    for line in lines[lo + 1:hi]:
        if line is None:
            continue
        if line.startswith(("$", "\t@", "\tbra", "\tret")):
            return False
        if space == "shared" and _BARRIER.match(line):
            return False
    return True


def vectorize_loads(lines: list[str], align: dict[str, int]) -> tuple[list[str], int]:
    """Returns (new lines, number of groups merged)."""
    loads: dict[tuple, list[tuple[int, int, str]]] = {}
    for i, line in enumerate(lines):
        m = _LOAD.match(line)
        if m and m.group(4) in align:
            space, ty, dst, base, off = m.group(1), m.group(2), m.group(3), m.group(4), int(m.group(5) or 0)
            loads.setdefault((space, ty, base), []).append((off, i, dst))
    out = list(lines)
    merged = 0
    for (space, ty, base), group in loads.items():
        by_off = {}
        for off, i, dst in group:
            by_off.setdefault(off, (i, dst))
        used: set[int] = set()
        for off in sorted(by_off):
            if by_off[off][0] in used:
                continue
            # PTX has no scalar-to-vector benefit for f64 beyond pairs here;
            # v2.f64 is 16 bytes and can use the same alignment proof.
            widths = (2,) if ty == "f64" else (4, 2)
            for n in widths:
                width = _WIDTH[ty]
                need = [off + k * width for k in range(n)]
                if (align[base] % (n * width)) or (off % (n * width)):
                    continue
                if not all(o in by_off and by_off[o][0] not in used for o in need):
                    continue
                idx = [by_off[o][0] for o in need]
                lo, hi = min(idx), max(idx)
                if not _block_safe(out, lo, hi, space):
                    continue
                dsts = ", ".join(by_off[o][1] for o in need)
                addr = f"[{base}+{off}]" if off else f"[{base}]"
                out[lo] = f"\tld.{space}.v{n}.{ty}\t{{{dsts}}}, {addr};"
                for i in idx:
                    if i != lo:
                        out[i] = None
                used.update(idx)
                merged += 1
                break
    return [line for line in out if line is not None], merged
