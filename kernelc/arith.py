"""Typed scalar arithmetic over ``Val``s with constant folding.

``Arith`` is the only place that turns an abstract operation (``add`` of two
values) into PTX text. It folds constants, applies the trivial identities
(``x*1``, ``x+0``) and inserts the conversions mixed operand types need, so the
statement lowering above it never spells an opcode for scalar math.

Float ops use explicit ``.rn`` rounding so that ``ptxas`` never contracts a
``mul`` + ``add`` pair into an ``fma``: elementwise results then match numpy's
separately rounded float32 arithmetic bit for bit.
"""

from __future__ import annotations

import math

import numpy as np

from .ptx import INT_TYPES, Emitter, Val, f32_round, imm_f32, imm_f64, imm_int, imm_pred

_CMP = {"<": "lt", "<=": "le", ">": "gt", ">=": "ge", "==": "eq", "!=": "ne"}
_PY_CMP = {
    "<": lambda a, b: a < b,
    "<=": lambda a, b: a <= b,
    ">": lambda a, b: a > b,
    ">=": lambda a, b: a >= b,
    "==": lambda a, b: a == b,
    "!=": lambda a, b: a != b,
}


def _c_div(a: int, b: int) -> int:
    """C integer division (truncates toward zero)."""
    q = abs(a) // abs(b)
    return q if (a >= 0) == (b >= 0) else -q


def _c_rem(a: int, b: int) -> int:
    return a - _c_div(a, b) * b


def _wrap32(n: int) -> int:
    n &= 0xFFFFFFFF
    return n - (1 << 32) if n & 0x80000000 else n


class Arith:
    def __init__(self, em: Emitter, fast_math: bool = False) -> None:
        self.em = em
        # fast_math: f32 add/sub/mul without .rn (lets ptxas fuse to fma) and div.full
        self.fast_math = fast_math

    # ------------------------------------------------------------------ helpers
    def reg(self, ty: str) -> Val:
        return self.em.reg(ty)

    def mov(self, dst: Val, src: Val) -> None:
        """Copy ``src`` into the existing register ``dst`` (converting if the types differ)."""
        src = self.convert(src, dst.ty)
        if dst.ty == "pred":
            if src.is_const:
                tmp = self.reg("s32")
                self.em.ins("mov.b32", tmp.text, "1" if src.const else "0")
                self.em.ins("setp.ne.s32", dst.text, tmp.text, "0")
            else:
                self.em.ins("mov.pred", dst.text, src.text)
            return
        self.em.ins(f"mov.{_bits(dst.ty)}", dst.text, src.text)

    def materialize(self, v: Val) -> Val:
        """Force a value into a register (for operands that must not be immediates)."""
        if not v.is_const:
            return v
        r = self.reg(v.ty)
        self.mov(r, v)
        return r

    def const(self, value, ty: str) -> Val:
        if ty == "f32":
            return imm_f32(value)
        if ty == "f64":
            return imm_f64(value)
        if ty == "pred":
            return imm_pred(bool(value))
        return imm_int(int(value), ty)

    # -------------------------------------------------------------- conversions
    def convert(self, v: Val, ty: str) -> Val:
        if v.ty == ty or (v.ty in ("s32", "u32") and ty in ("s32", "u32")) or (v.ty in ("s64", "u64") and ty in ("s64", "u64")):
            return v if v.ty == ty else Val(v.text, ty, v.const)
        if v.is_const:
            if ty == "f32":
                return imm_f32(float(v.const))
            if ty == "f64":
                return imm_f64(float(v.const))
            if ty == "pred":
                return imm_pred(bool(v.const))
            if ty in INT_TYPES:
                return imm_int(int(v.const), ty)
        if v.ty == "pred":
            one = imm_f32(1.0) if ty == "f32" else imm_int(1, ty)
            zero = imm_f32(0.0) if ty == "f32" else imm_int(0, ty)
            r = self.reg(ty)
            self.em.ins(f"selp.{_bits(ty) if ty != 'f32' else 'f32'}", r.text, one.text, zero.text, v.text)
            return r
        if ty == "pred":
            return self.cmp("!=", v, self.const(0, v.ty))
        r = self.reg(ty)
        if v.ty == "f32" and ty in INT_TYPES:
            self.em.ins(f"cvt.rzi.{ty}.f32", r.text, v.text)
        elif v.ty in INT_TYPES and ty == "f32":
            self.em.ins(f"cvt.rn.f32.{v.ty}", r.text, v.text)
        elif v.ty == "f32" and ty == "f64":
            self.em.ins("cvt.f64.f32", r.text, v.text)
        elif v.ty == "f64" and ty == "f32":
            self.em.ins("cvt.rn.f32.f64", r.text, v.text)
        elif v.ty in INT_TYPES and ty == "f64":
            self.em.ins(f"cvt.rn.f64.{v.ty}", r.text, v.text)
        elif v.ty == "f64" and ty in INT_TYPES:
            self.em.ins(f"cvt.rzi.{ty}.f64", r.text, v.text)
        else:  # integer width change
            self.em.ins(f"cvt.{ty}.{v.ty}", r.text, v.text)
        return r

    def unify(self, a: Val, b: Val) -> tuple[Val, Val, str]:
        """Bring two operands to one arithmetic type (C-like promotion)."""
        if a.ty == b.ty and a.ty != "pred":
            return a, b, a.ty
        if "f64" in (a.ty, b.ty):
            ty = "f64"
        elif "f32" in (a.ty, b.ty):
            ty = "f32"
        elif "s64" in (a.ty, b.ty) or "u64" in (a.ty, b.ty):
            ty = "s64"
        else:
            ty = "s32"
        return self.convert(a, ty), self.convert(b, ty), ty

    # ------------------------------------------------------------ integer + float
    def binop(self, op: str, a: Val, b: Val) -> Val:
        """C-level binary operator on two values: + - * / % << >> & | ^ and comparisons."""
        if op in _CMP:
            return self.cmp(op, a, b)
        if op in ("&&", "||"):
            return self.logic(op, a, b)
        if op in ("&", "|", "^") and a.ty == "pred" and b.ty == "pred":
            return self.logic({"&": "&&", "|": "||", "^": "^"}[op], a, b)
        a, b, ty = self.unify(a, b)
        if ty in ("f32", "f64"):
            return self._float_binop(op, a, b, ty)
        return self._int_binop(op, a, b, ty)

    def _int_binop(self, op: str, a: Val, b: Val, ty: str) -> Val:
        if a.is_const and b.is_const:
            x, y = a.const, b.const
            folded = {
                "+": lambda: x + y,
                "-": lambda: x - y,
                "*": lambda: x * y,
                "/": lambda: _c_div(x, y),
                "//": lambda: _c_div(x, y),  # truncates, like the runtime div.s32
                "%": lambda: _c_rem(x, y),
                "<<": lambda: x << y,
                ">>": lambda: x >> y,
                "&": lambda: x & y,
                "|": lambda: x | y,
                "^": lambda: x ^ y,
                "min": lambda: min(x, y),
                "max": lambda: max(x, y),
            }[op]()
            return imm_int(_wrap32(folded) if ty in ("s32", "u32") else folded, ty)
        # algebraic identities
        if op == "+":
            if a.const == 0:
                return b
            if b.const == 0:
                return a
        if op == "-" and b.const == 0:
            return a
        if op == "*":
            if a.const == 1:
                return b
            if b.const == 1:
                return a
            if a.const == 0 or b.const == 0:
                return imm_int(0, ty)
        if op in ("/", "//") and b.const == 1:
            return a
        if op == "%" and b.const == 1:
            return imm_int(0, ty)
        if op in ("<<", ">>") and b.const == 0:
            return a
        if a.is_const and op in ("+", "*", "&", "|", "^", "min", "max"):
            a, b = b, a  # keep the immediate in the second slot
        a = self.materialize(a)
        r = self.reg(ty)
        sty = "s64" if ty in ("s64", "u64") else "s32"
        bty = "b64" if sty == "s64" else "b32"
        opcode = {
            "+": f"add.{sty}",
            "-": f"sub.{sty}",
            "*": f"mul.lo.{sty}",
            "/": f"div.{sty}",
            "//": f"div.{sty}",
            "%": f"rem.{sty}",
            "<<": f"shl.{bty}",
            ">>": f"shr.{sty}",
            "&": f"and.{bty}",
            "|": f"or.{bty}",
            "^": f"xor.{bty}",
            "min": f"min.{sty}",
            "max": f"max.{sty}",
        }[op]
        if op in ("<<", ">>"):
            b = self.convert(b, "u32") if not b.is_const else imm_int(b.const, "u32")
        self.em.ins(opcode, r.text, a.text, b.text)
        return r

    def _float_binop(self, op: str, a: Val, b: Val, ty: str) -> Val:
        if ty == "f32" and a.is_const and b.is_const and op in ("+", "-", "*", "/"):
            x, y = np.float32(a.const), np.float32(b.const)
            with np.errstate(all="ignore"):
                folded = {"+": x + y, "-": x - y, "*": x * y, "/": x / y}[op]
            if math.isfinite(float(folded)):
                return imm_f32(float(folded))
        if op == "*":
            if a.const == 1.0:
                return b
            if b.const == 1.0:
                return a
        if op == "/" and b.const == 1.0:
            return a
        if op == "%":
            return self.fmod(a, b)
        r = self.reg(ty)
        opcode = {"+": "add.rn", "-": "sub.rn", "*": "mul.rn", "/": "div.rn", "min": "min", "max": "max"}[op]
        if self.fast_math and ty == "f32":
            opcode = {"+": "add", "-": "sub", "*": "mul", "/": "div.full"}.get(op, opcode)
        self.em.ins(f"{opcode}.{ty}", r.text, a.text, b.text)
        return r

    # convenience wrappers used by the math library
    def add(self, a: Val, b: Val) -> Val:
        return self.binop("+", a, b)

    def sub(self, a: Val, b: Val) -> Val:
        return self.binop("-", a, b)

    def mul(self, a: Val, b: Val) -> Val:
        return self.binop("*", a, b)

    def div(self, a: Val, b: Val) -> Val:
        return self.binop("/", a, b)

    def fma(self, a: Val, b: Val, c: Val) -> Val:
        """a * b + c with a single rounding."""
        if "f64" in (a.ty, b.ty, c.ty):
            a, b, c = (self.materialize(self.convert(v, "f64")) if i == 0 else self.convert(v, "f64") for i, v in enumerate((a, b, c)))
            r = self.reg("f64")
            self.em.ins("fma.rn.f64", r.text, a.text, b.text, c.text)
            return r
        a, b, c = (self.convert(v, "f32") for v in (a, b, c))
        if a.is_const and b.is_const and c.is_const:
            return imm_f32(float(np.float64(a.const) * np.float64(b.const) + np.float64(c.const)))
        r = self.reg("f32")
        self.em.ins("fma.rn.f32", r.text, a.text, b.text, c.text)
        return r

    def fmax(self, a: Val, b: Val) -> Val:
        return self.binop("max", a, b)

    def fmin(self, a: Val, b: Val) -> Val:
        return self.binop("min", a, b)

    def unary(self, opcode: str, a: Val, ty: str = "f32") -> Val:
        """One-operand instruction such as ``abs.f32`` or ``sqrt.rn.f32``."""
        a = self.convert(a, ty)
        r = self.reg(ty)
        self.em.ins(f"{opcode}.{ty}", r.text, a.text)
        return r

    def fmod_c(self, a: Val, b: Val) -> Val:
        """C fmod. The rounded quotient can be one off, which fma(-q, b, a) exposes exactly."""
        a, b = self.convert(a, "f32"), self.convert(b, "f32")
        q = self.unary("cvt.rzi.f32", self._exact_div(a, b))
        r = self.fma(self.unary("neg", q), b, a)
        sb = self._copysign(b, a)
        over = self.logic("&&", self.cmp("!=", r, imm_f32(0.0)), self._sign_differs(r, a))
        r = self.select(over, self.add_exact(r, sb), r)
        under = self.cmp(">=", self.unary("abs", r), self.unary("abs", b))
        return self.select(under, self.sub_exact(r, sb), r)

    def fmod(self, a: Val, b: Val) -> Val:
        """numpy remainder: fmod shifted by b when its sign differs from b's."""
        r = self.fmod_c(a, b)
        b = self.convert(b, "f32")
        shift = self.logic("&&", self.cmp("!=", r, imm_f32(0.0)), self._sign_differs(r, b))
        return self.select(shift, self.add_exact(r, b), r)

    def floor_div(self, a: Val, b: Val) -> Val:
        """numpy floor_divide, as in npy_divmod."""
        a, b = self.convert(a, "f32"), self.convert(b, "f32")
        mod = self.fmod_c(a, b)
        div = self._exact_div(self.sub_exact(a, mod), b)
        adjust = self.logic("&&", self.cmp("!=", mod, imm_f32(0.0)), self._sign_differs(mod, b))
        div = self.select(adjust, self.sub_exact(div, imm_f32(1.0)), div)
        return self.unary("cvt.rni.f32", div)

    # always .rn, even under fast_math
    def _exact_div(self, a: Val, b: Val) -> Val:
        r = self.reg("f32")
        self.em.ins("div.rn.f32", r.text, self.materialize(a).text, b.text)
        return r

    def add_exact(self, a: Val, b: Val) -> Val:
        r = self.reg("f32")
        self.em.ins("add.rn.f32", r.text, self.materialize(a).text, b.text)
        return r

    def sub_exact(self, a: Val, b: Val) -> Val:
        r = self.reg("f32")
        self.em.ins("sub.rn.f32", r.text, self.materialize(a).text, b.text)
        return r

    def _copysign(self, mag: Val, sign: Val) -> Val:
        m, s = self.reg("s32"), self.reg("s32")
        self.em.ins("mov.b32", m.text, self.materialize(mag).text)
        self.em.ins("mov.b32", s.text, self.materialize(sign).text)
        m2 = self.binop("|", self.binop("&", m, imm_int(0x7FFFFFFF)), self.binop("&", s, imm_int(-0x80000000)))
        r = self.reg("f32")
        self.em.ins("mov.b32", r.text, self.materialize(m2).text)
        return r

    def _sign_differs(self, x: Val, y: Val) -> Val:
        return self.logic("^", self.cmp("<", x, imm_f32(0.0)), self.cmp("<", y, imm_f32(0.0)))

    def isnan(self, x: Val) -> Val:
        r = self.reg("pred")
        x = self.materialize(x)
        self.em.ins(f"setp.nan.{x.ty}", r.text, x.text, x.text)
        return r

    # ------------------------------------------------------------- predicates
    def cmp(self, op: str, a: Val, b: Val) -> Val:
        if a.ty == "pred" or b.ty == "pred":
            a, b = self.convert(a, "s32"), self.convert(b, "s32")
        a, b, ty = self.unify(a, b)
        if a.is_const and b.is_const:
            return imm_pred(_PY_CMP[op](a.const, b.const))
        if a.is_const:  # setp wants a register first; mirror the comparison
            a, b = b, a
            op = {"<": ">", "<=": ">=", ">": "<", ">=": "<=", "==": "==", "!=": "!="}[op]
        r = self.reg("pred")
        suffix = {"s32": "s32", "u32": "u32", "s64": "s64", "u64": "u64", "f32": "f32", "f64": "f64"}[ty]
        self.em.ins(f"setp.{_CMP[op]}.{suffix}", r.text, a.text, b.text)
        return r

    def logic(self, op: str, a: Val, b: Val) -> Val:
        a, b = self.convert(a, "pred"), self.convert(b, "pred")
        if a.is_const and b.is_const:
            return imm_pred({"&&": a.const and b.const, "||": a.const or b.const, "^": a.const != b.const}[op])
        if a.is_const:
            a, b = b, a
        if b.is_const:
            if op == "&&":
                return a if b.const else imm_pred(False)
            if op == "||":
                return imm_pred(True) if b.const else a
            return a if not b.const else self.lnot(a)
        r = self.reg("pred")
        self.em.ins({"&&": "and.pred", "||": "or.pred", "^": "xor.pred"}[op], r.text, a.text, b.text)
        return r

    def lnot(self, a: Val) -> Val:
        a = self.convert(a, "pred")
        if a.is_const:
            return imm_pred(not a.const)
        r = self.reg("pred")
        self.em.ins("not.pred", r.text, a.text)
        return r

    def select(self, cond: Val, a: Val, b: Val) -> Val:
        """``cond ? a : b``."""
        cond = self.convert(cond, "pred")
        if cond.is_const:
            return a if cond.const else b
        if a.ty == "pred" and b.ty == "pred":
            return self.logic("||", self.logic("&&", cond, a), self.logic("&&", self.lnot(cond), b))
        a, b, ty = self.unify(a, b)
        if a.is_const and b.is_const and a.const == b.const:
            return a
        r = self.reg(ty)
        self.em.ins(f"selp.{'f32' if ty == 'f32' else _bits(ty)}", r.text, a.text, b.text, cond.text)
        return r


def _bits(ty: str) -> str:
    return {"s32": "b32", "u32": "b32", "f32": "f32", "s64": "b64", "u64": "b64", "f64": "f64", "b16": "b16"}[ty]


__all__ = ["Arith", "f32_round"]
