"""PTX emission primitives: typed values, virtual registers, labels.

Every value the code generator handles is a ``Val``: either a virtual register
(``%f12``) or a compile-time constant that is printed as an immediate. Keeping
constants symbolic lets the lowering fold index arithmetic (unrolled loop
counters, literal offsets) before anything reaches PTX.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np

# Register class per value type. s32/u32 share the .b32 file, s64/u64 the .b64 one.
_PREFIX = {"pred": "%p", "s32": "%r", "u32": "%r", "s64": "%rd", "u64": "%rd", "f32": "%f", "b16": "%h", "f64": "%fd"}
_DECL = {"%p": ".pred", "%r": ".b32", "%rd": ".b64", "%f": ".f32", "%h": ".b16", "%fd": ".f64"}

INT_TYPES = ("s32", "s64", "u32", "u64")


@dataclass(frozen=True)
class Val:
    """A PTX operand: register name or immediate text, its type, and its constant value if known."""

    text: str
    ty: str
    const: object = None

    @property
    def is_const(self) -> bool:
        return self.const is not None

    @property
    def is_int(self) -> bool:
        return self.ty in INT_TYPES


def f32_round(x: float) -> float:
    """Round a Python float to the nearest float32, as the device would store it."""
    return float(np.float32(x))


def f32_hex(x: float) -> str:
    return "0f%08X" % int(np.array(x, dtype=np.float32).view(np.uint32))


def imm_f32(x: float) -> Val:
    x = f32_round(x)
    return Val(f32_hex(x), "f32", x)


def imm_f64(x: float) -> Val:
    return Val("0d%016X" % int(np.array(x, dtype=np.float64).view(np.uint64)), "f64", float(x))


def imm_int(n: int, ty: str = "s32") -> Val:
    return Val(str(int(n)), ty, int(n))


def imm_pred(b: bool) -> Val:
    return Val("1" if b else "0", "pred", bool(b))


def ascii_only(text: str) -> str:
    """PTX rejects non-ASCII anywhere, comments included."""
    return text.encode("ascii", "replace").decode("ascii")


class Emitter:
    """Collects instructions for one kernel body and hands out fresh virtual registers.

    ``ptxas`` performs the real register allocation, so registers are never reused.
    """

    def __init__(self) -> None:
        self.lines: list[str] = []
        self._counts: dict[str, int] = defaultdict(int)
        self._labels = 0

    def reg(self, ty: str) -> Val:
        prefix = _PREFIX[ty]
        self._counts[prefix] += 1
        return Val(f"{prefix}{self._counts[prefix]}", ty)

    def ins(self, opcode: str, *operands: str) -> None:
        self.lines.append(f"\t{opcode}\t{', '.join(operands)};" if operands else f"\t{opcode};")

    def pred_ins(self, pred: str, opcode: str, *operands: str) -> None:
        """An instruction guarded by a predicate, e.g. ``@%p1 bra L``."""
        self.lines.append(f"\t@{pred} {opcode}\t{', '.join(operands)};")

    def comment(self, text: str) -> None:
        self.lines.append(f"\t// {ascii_only(text)}")

    def new_label(self, hint: str) -> str:
        self._labels += 1
        return f"$L_{hint}_{self._labels}"

    def label(self, name: str) -> None:
        self.lines.append(f"{name}:")

    def declarations(self) -> list[str]:
        return [f"\t.reg {_DECL[p]}\t{p}<{n + 1}>;" for p, n in sorted(self._counts.items())]
