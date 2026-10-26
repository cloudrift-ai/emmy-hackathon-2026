"""Kernel IR -> PTX lowering.

``KernelCompiler`` walks the KernelOp body once, in program order, translating
each statement into PTX through ``Arith`` (scalar math) and ``mathlib``
(elementwise ops). The state it carries is small:

* ``env``     -- IR name -> ``Val`` for every axis variable and SSA value in scope.
                 Nested bodies work on a copy, so their definitions vanish at
                 scope exit, exactly as in Kernel IR (only accumulators, whose
                 registers are allocated *before* the loop, outlive it).
* ``bufs``    -- buffer name -> ``Buffer`` (global parameter, shared slab, or an
                 inlined constant).
* ``memo``    -- scoped cache of compiled index expressions, so repeated
                 subexpressions (the same flattened offset in several loads)
                 are computed once.

Loops whose trip count is the same for every thread are unrolled at compile
time (fully, or a few copies per trip) with the induction variable bound to a
constant or ``i + k*step``; everything that depends on it then folds to
immediates or address offsets.
"""

from __future__ import annotations

import logging
import math
import os
from contextlib import contextmanager
from dataclasses import dataclass

from emmy.compiler.ir.expr import BinaryExpr, Literal

from . import mathlib
from .arith import Arith
from .loader import KernelSource
from .peephole import vectorize_loads
from .ptx import INT_TYPES, Emitter, Val, imm_f32, imm_f64, imm_int

log = logging.getLogger("kernelc")

PTX_VERSION = "8.0"
PTX_TARGET = "sm_89"
STATIC_SMEM_LIMIT = 48 * 1024
DEFAULT_BLOCK = 256  # Emmy's scalar-tier block size (Tile.block_threads is None)
UNROLL_BUDGET = 2048  # max IR leaf statements a fully unrolled ``unroll`` loop may emit
SMALL_UNROLL_LEAVES = 64  # unflagged loops with a uniform trip count unroll fully up to this
PARTIAL_UNROLL_LEAVES = 32  # max leaves in one trip of a partially unrolled loop
INT32_MAX = 2**31 - 1
GLOBAL_ALIGN = 16  # assumed base alignment of a global buffer (Emmy's float4 casts rely on it)
FAST_MATH_MIN_RTOL = 1e-4

# dtype name -> (bytes, ld/st type, register type the value lives in)
DTYPES = {
    "f32": (4, "f32", "f32"),
    "f16": (2, "b16", "f32"),
    "bf16": (2, "b16", "f32"),
    "f64": (8, "f64", "f64"),
    "i32": (4, "s32", "s32"),
    "u32": (4, "u32", "s32"),
    "i64": (8, "s64", "s64"),
    "u64": (8, "u64", "s64"),
    "i16": (2, "s16", "s32"),
    "u16": (2, "u16", "s32"),
    "i8": (1, "s8", "s32"),
    "u8": (1, "u8", "s32"),
    "bool": (1, "u8", "s32"),
}
# Smem C types -> dtype names
CTYPES = {"float": "f32", "__half": "f16", "__nv_bfloat16": "bf16", "int": "i32", "unsigned": "u32",
          "unsigned long long": "u64", "long long": "i64", "long": "i64", "double": "f64"}
SWIZZLE_XOR = {"B128": (6, 0x7), "B64": (6, 0x3), "B32": (6, 0x1)}


@dataclass
class Buffer:
    name: str
    space: str  # "global" | "shared" | "literal"
    shape: tuple[int, ...]
    dtype: str
    base: Val | None = None  # u64 global address, or u32 shared address
    symbol: str | None = None  # shared: the .shared symbol
    offset: int = 0  # shared: byte offset of this slab inside ``symbol``
    readonly: bool = False
    literal: float | None = None
    align: int = GLOBAL_ALIGN

    @property
    def esize(self) -> int:
        return DTYPES[self.dtype][0]


def kind(node) -> str:
    return type(node).__name__


def walk(body):
    """Pre-order walk over every statement, nested bodies included."""
    for s in body:
        yield s
        for attr in ("body", "else_body"):
            inner = getattr(s, attr, None)
            if inner:
                yield from walk(inner)


def leaf_count(body) -> int:
    return sum(1 for s in walk(body) if not getattr(s, "body", None))


def dtype_name(dt, default: str = "f32") -> str:
    return getattr(dt, "name", None) or default


def static_extent(axis) -> int:
    ext = axis.extent
    if hasattr(ext, "is_static") and not ext.is_static:
        raise NotImplementedError(f"symbolic extent on axis {axis.name!r} is not supported")
    return int(ext.as_static()) if hasattr(ext, "as_static") else int(ext)


class KernelCompiler:
    def __init__(self, src: KernelSource) -> None:
        self.src = src
        self.em = Emitter()
        self.fast_math = self._decide_fast_math(src)
        self.ar = Arith(self.em, fast_math=self.fast_math)
        self.align: dict[str, int] = {}  # address register -> byte alignment
        self.env: dict[str, Val] = {}
        self.memo: list[dict] = [{}]
        self.mutable: set[str] = set()  # names whose register is updated in place
        self.owned: set[str] = set()  # register texts that belong to exactly one mutable name
        self.products: dict[str, tuple[Val, Val]] = {}  # register -> factors of the multiply that made it
        self.ranges: dict[str, tuple[int, int]] = {}  # register -> [lo, hi] of a Tile axis value
        self.offsets: dict[str, tuple[Val, int]] = {}  # register -> (base, constant)
        self.divisor: dict[str, int] = {}  # register -> known divisor of its value
        self.bufs: dict[str, Buffer] = {}
        self.module_decls: list[str] = []
        self.exit_label = self.em.new_label("exit")
        self.block_threads = src.block or DEFAULT_BLOCK
        self.quiet = 0  # >0 while emitting the 2nd+ copy of an unrolled body (no IR comments)

    @staticmethod
    def _decide_fast_math(src: KernelSource) -> bool:
        """On when meta.json's rtol >= 1e-4 (worst error is ~5e-6). KERNELC_FAST_MATH=0/1 overrides."""
        forced = os.environ.get("KERNELC_FAST_MATH")
        if forced in ("0", "1"):
            log.info("%s: fast math %s (forced by KERNELC_FAST_MATH)", src.name, "on" if forced == "1" else "off")
            return forced == "1"
        on = src.rtol is not None and src.rtol >= FAST_MATH_MIN_RTOL
        log.info("%s: fast math %s (rtol=%s, threshold %g)", src.name, "on" if on else "off", src.rtol, FAST_MATH_MIN_RTOL)
        return on

    # ================================================================ driver
    def compile(self) -> str:
        body = self.src.body
        written = {s.output for s in walk(body) if kind(s) == "Write"} | {s.dst for s in walk(body) if kind(s) == "ZeroPrologue"}
        tile = next((s for s in body if kind(s) == "Tile"), None)
        if tile is not None:
            self.block_threads = (tile.block_threads or self.src.block or DEFAULT_BLOCK) + tile.aux_threads
        if self.src.block is not None and self.src.block != self.block_threads:
            log.warning("%s: meta.json launches %d threads but the IR implies %d; using the launch's",
                        self.src.name, self.src.block, self.block_threads)
            self.block_threads = self.src.block
        self._plan_shared_memory(body)
        params = self._prologue(written)
        for s in body:
            self.lower(s)
        self.em.label(self.exit_label)
        self.em.ins("ret")
        return self._assemble(params)

    def _assemble(self, params: list[str]) -> str:
        lines, merged = vectorize_loads(self.em.lines, self.align)
        log.info("%s: merged %d groups of scalar loads into vector loads", self.src.name, merged)
        out = [
            "//",
            f"// {self.src.name}: generated by kernelc (Kernel IR -> PTX)",
            "//",
            f".version {PTX_VERSION}",
            f".target {PTX_TARGET}",
            ".address_size 64",
            "",
            *self.module_decls,
            "",
            f".visible .entry {self.src.name}(",
            ",\n".join(f"\t{p}" for p in params),
            f")\n.maxntid {self.block_threads}, 1, 1",
            "{",
            *self.em.declarations(),
            "",
            *lines,
            "}",
            "",
        ]
        return "\n".join(out)

    def _plan_shared_memory(self, body) -> None:
        """Pack every Smem slab in body order. Up to 48 KiB they become static ``.shared``
        arrays; beyond that they alias one ``.extern`` dynamic pool (the launch passes its size)."""
        slabs = [s for s in walk(body) if kind(s) == "Smem"]
        layout, total, max_align = [], 0, 16
        for s in slabs:
            dt = CTYPES.get(s.dtype, "f32")
            size = math.prod(int(e) for e in s.extents) * DTYPES[dt][0]
            align = max(DTYPES[dt][0], int(s.align or 0))
            max_align = max(max_align, align)
            total = (total + align - 1) // align * align
            layout.append((s, dt, total, size, align))
            total += size
        dynamic = total > STATIC_SMEM_LIMIT
        if dynamic:
            self.module_decls.append(f".extern .shared .align {max_align} .b8 smem_pool[];")
        for s, dt, offset, size, align in layout:
            if dynamic:
                symbol = "smem_pool"
            else:
                symbol, offset = f"smem_{s.name}", 0
                self.module_decls.append(f".shared .align {align} .b8 {symbol}[{size}];")
            shape = tuple(int(e) for e in s.extents)
            base_align = max_align if dynamic else align
            self.bufs[s.name] = Buffer(s.name, "shared", shape, dt, symbol=symbol, offset=offset,
                                       align=math.gcd(base_align, offset) if offset else base_align)

    def _prologue(self, written: set[str]) -> list[str]:
        em, ar = self.em, self.ar
        params = []
        for i, p in enumerate(self.src.params):
            pname = f"{self.src.name}_param_{i}"
            if p.scalar:
                vty = DTYPES.get(p.dtype, DTYPES["f32"])[2]
                params.append(f".param .{vty} {pname}")
                r = ar.reg(vty)
                em.ins(f"ld.param.{vty}", r.text, f"[{pname}]")
                self.env[p.name] = r
                continue
            params.append(f".param .u64 {pname}")
            raw, glob = ar.reg("u64"), ar.reg("u64")
            em.ins("ld.param.u64", raw.text, f"[{pname}]")
            em.ins("cvta.to.global.u64", glob.text, raw.text)
            shape = self.src.shapes.get(p.name, ())
            self.bufs[p.name] = Buffer(p.name, "global", shape, p.dtype, base=glob, readonly=p.name not in written)
        # constants that are not kernel parameters are inlined as literals
        for name, value in self.src.constants.items():
            if name not in self.bufs:
                dt = "f64" if self.src.dtypes.get(name) == "f64" else "f32"
                self.bufs[name] = Buffer(name, "literal", (1,), dt, literal=value)
        for buf in self.bufs.values():  # shared slab base addresses, computed where they dominate every use
            if buf.space == "shared":
                buf.base = self.ar.add(self.special_symbol(buf.symbol), imm_int(buf.offset))
        tid = self.special("%tid.x")
        self.env["lane"] = ar.binop("&", tid, imm_int(31))
        self.env["warp"] = ar.binop(">>", tid, imm_int(5))
        return params

    def special(self, sreg: str) -> Val:
        r = self.ar.reg("s32")
        self.em.ins("mov.u32", r.text, sreg)
        return r

    # ================================================================ scopes
    @contextmanager
    def scope(self):
        saved = self.env
        self.env = dict(saved)
        self.memo.append({})
        try:
            yield
        finally:
            self.env = saved
            self.memo.pop()

    def lookup(self, name: str) -> Val:
        try:
            return self.env[name]
        except KeyError:
            raise KeyError(f"IR name {name!r} is not bound at this point") from None

    def bind(self, name: str, val: Val) -> None:
        if val.text in self.owned:  # never alias a register that is updated in place
            copy = self.ar.reg(val.ty)
            self.ar.mov(copy, val)
            val = copy
        self.env[name] = val

    def new_mutable(self, name: str, init: Val, ty: str) -> Val:
        """Allocate a fresh register for a carried value (accumulator, Init) and seed it."""
        r = self.ar.reg(ty)
        self.ar.mov(r, init)
        self.env[name] = r
        self.mutable.add(name)
        self.owned.add(r.text)
        return r

    def mutable_reg(self, name: str) -> Val:
        """The register an in-place update of ``name`` writes (re-homed if not yet owned)."""
        cur = self.lookup(name)
        if cur.text in self.owned:
            return cur
        return self.new_mutable(name, cur, cur.ty)

    # =========================================================== expressions
    def expr(self, e) -> Val:
        k = kind(e)
        if k == "Literal":
            return self._literal(e)
        if k == "Var":
            return self.lookup(e.name)
        free = tuple(sorted(e.free_vars())) if hasattr(e, "free_vars") else ()
        key = None
        if not any(v in self.mutable for v in free):
            try:
                key = (e, tuple(self.lookup(v).text for v in free))
            except KeyError:
                key = None
            if key is not None:
                for cache in reversed(self.memo):
                    if key in cache:
                        return cache[key]
        val = self._expr(e, k)
        if key is not None:
            self.memo[-1][key] = val
        return val

    def _literal(self, e) -> Val:
        v = e.value
        if isinstance(v, bool):
            return self.ar.const(v, "pred")
        if e.dtype in ("int", "i32", "long", "i64") or (isinstance(v, int) and e.dtype not in ("float", "f32")):
            return imm_int(int(v), "s64" if e.dtype in ("long", "i64") else "s32")
        return imm_f32(float(v))

    def _expr(self, e, k: str) -> Val:
        ar = self.ar
        if k == "BinaryExpr":
            return ar.binop(e.op, self.expr(e.left), self.expr(e.right))
        if k == "TernaryExpr":
            return ar.select(self.expr(e.cond), self.expr(e.if_true), self.expr(e.if_false))
        if k == "FuncCallExpr":
            name = {"fmax": "maximum", "fmin": "minimum", "fabs": "abs", "expf": "exp", "__expf": "exp"}.get(e.name, e.name)
            return mathlib.apply(ar, name, [self.expr(a) for a in e.args])
        if k == "CastExpr":
            return self._cast(self.expr(e.expr), e.dtype)
        if k == "Builtin":
            return self._builtin(e.name)
        if k == "FlatIndex":
            return self.flat_index(e.buffer, e.index)
        raise NotImplementedError(f"expression {k} is not supported")

    def _cast(self, v: Val, ctype: str) -> Val:
        target = {"float": "f32", "int": "s32", "unsigned": "s32", "long long": "s64", "long": "s64",
                  "double": "f32", "bool": "pred"}.get(ctype)
        if target is not None:
            return self.ar.convert(v, target)
        if ctype in ("__half", "half", "__nv_bfloat16"):
            return self.round_to(v, "f16" if "half" in ctype else "bf16")
        raise NotImplementedError(f"cast to {ctype!r} is not supported")

    def _builtin(self, name: str) -> Val:
        sreg = {"threadIdx.x": "%tid.x", "threadIdx.y": "%tid.y", "threadIdx.z": "%tid.z",
                "blockIdx.x": "%ctaid.x", "blockIdx.y": "%ctaid.y", "blockIdx.z": "%ctaid.z",
                "blockDim.x": "%ntid.x", "blockDim.y": "%ntid.y", "gridDim.x": "%nctaid.x",
                "gridDim.y": "%nctaid.y"}.get(name)
        if sreg is None:
            raise NotImplementedError(f"builtin {name!r} is not supported")
        return self.special(sreg)

    def round_to(self, v: Val, dt: str) -> Val:
        """Round an f32 value to f16/bf16 precision and back (the storage rounding of a narrow Assign)."""
        v = self.ar.materialize(self.ar.convert(v, "f32"))
        h, r = self.ar.reg("b16"), self.ar.reg("f32")
        if dt == "f16":
            self.em.ins("cvt.rn.f16.f32", h.text, v.text)
            self.em.ins("cvt.f32.f16", r.text, h.text)
        else:
            self.em.ins("cvt.rn.bf16.f32", h.text, v.text)
            self.em.ins("cvt.f32.bf16", r.text, h.text)
        return r

    # ============================================================== addressing
    def buffer(self, name: str) -> Buffer:
        if name not in self.bufs:
            raise KeyError(f"buffer {name!r} is neither a kernel parameter, a constant nor a shared slab")
        return self.bufs[name]

    def flat_index(self, name: str, index) -> Val:
        """Row-major element offset of ``index`` in buffer ``name`` (K4: a single
        coordinate is already flat; otherwise every dimension is spelled)."""
        ar = self.ar
        if len(index) == 0:
            return imm_int(0)
        if len(index) == 1:
            v = self.expr(index[0])
            return ar.convert(v, "s32") if v.ty == "f32" else v
        shape = self.bufs[name].shape if name in self.bufs else ()
        if len(shape) != len(index):
            raise ValueError(f"buffer {name!r} has shape {shape} but is indexed with {len(index)} coordinates "
                             "(an index is either one flat offset or one coordinate per dimension)")
        flat = imm_int(0)
        for d, idx in enumerate(index):
            stride = math.prod(shape[d + 1:])
            flat = ar.add(flat, ar.mul(self.expr(idx), imm_int(stride)))
        return flat

    # -- affine addressing: base register + immediate offset ----------------------------------
    def _affine(self, e):
        """Split an index into (variable Expr or None, constant, gcd of the variable's multipliers)."""
        k = kind(e)
        if k == "Literal" and isinstance(e.value, int) and not isinstance(e.value, bool):
            return None, int(e.value), 0
        if k == "Var":
            v = self.lookup(e.name)
            if v.is_const and v.ty in INT_TYPES:
                return None, int(v.const), 0
            if v.text in self.offsets:
                base, c = self.offsets[v.text]
                name = f"{e.name}@base"
                self.env[name] = base
                return type(e)(name=name), c, self.divisor.get(base.text, 1)
            return e, 0, self.divisor.get(v.text, 1)
        if k == "BinaryExpr" and e.op in ("+", "-", "*"):
            la, lc, lg = self._affine(e.left)
            ra, rc, rg = self._affine(e.right)
            if e.op == "*":
                if la is None or ra is None:
                    kk, (ne, c, g) = (lc, (ra, rc, rg)) if la is None else (rc, (la, lc, lg))
                    if kk == 0 or ne is None:
                        return None, c * kk, 0
                    return BinaryExpr("*", ne, Literal(kk, "int")), c * kk, g * abs(kk)
                return e, 0, 1
            c = lc + rc if e.op == "+" else lc - rc
            if la is None and ra is None:
                return None, c, 0
            if e.op == "+":
                ne = la if ra is None else ra if la is None else BinaryExpr("+", la, ra)
            else:
                ne = la if ra is None else BinaryExpr("-", la if la is not None else Literal(0, "int"), ra)
            return ne, c, math.gcd(lg, rg)
        v = self.expr(e)
        if v.is_const and v.ty in INT_TYPES:
            return None, int(v.const), 0
        return e, 0, 1

    def split_index(self, name: str, index) -> tuple[Val | None, int, int]:
        """flat_index as (variable part, constant offset, gcd)."""
        if len(index) == 0:
            return None, 0, 0
        shape = self.bufs[name].shape if name in self.bufs else ()
        if len(index) == 1:
            terms = [(index[0], 1)]
        elif len(shape) == len(index):
            terms = [(idx, math.prod(shape[d + 1:])) for d, idx in enumerate(index)]
        else:
            raise ValueError(f"buffer {name!r} has shape {shape} but is indexed with {len(index)} coordinates "
                             "(an index is either one flat offset or one coordinate per dimension)")
        var_e, const, g = None, 0, 0
        for e, stride in terms:
            ne, c, ge = self._affine(e)
            const += c * stride
            if ne is not None:
                term = ne if stride == 1 else BinaryExpr("*", ne, Literal(stride, "int"))
                var_e = term if var_e is None else BinaryExpr("+", var_e, term)
                g = math.gcd(g, ge * stride)
        if var_e is None:
            return None, const, 0
        var = self.expr(var_e)
        if var.ty == "f32":
            var = self.ar.convert(var, "s32")
        if var.is_const:
            return None, const + int(var.const), 0
        return var, const, max(g, 1)

    def address_of(self, buf: "Buffer", var: Val | None, const: int, g: int, k: int = 0) -> str:
        """[reg+imm] for element var + const + k, with one register per var per scope."""
        if var is None:
            return self.address(buf, imm_int(const), k)
        es = buf.esize
        key = ("addr", buf.name, var.text)
        # current scope only: reusing an outer one keeps it live across loops and spills
        reg = None if var.text in self.owned else self.memo[-1].get(key)
        if reg is None:
            if buf.space == "shared":
                reg = self.ar.reg("s32")
                self.em.ins("mad.lo.s32", reg.text, self.ar.materialize(var).text, str(es), self.ar.materialize(buf.base).text)
            else:
                wide = self.ar.reg("s64")
                if var.ty in ("s64", "u64"):
                    self.em.ins("mul.lo.s64", wide.text, var.text, str(es))
                else:
                    self.em.ins("mul.wide.s32", wide.text, self.ar.materialize(var).text, str(es))
                reg = self.ar.reg("u64")
                self.em.ins("add.s64", reg.text, buf.base.text, wide.text)
            self.align[reg.text] = math.gcd(g * es, buf.align)
            if var.text not in self.owned:
                self.memo[-1][key] = reg
        off = (const + k) * es
        return f"[{reg.text}+{off}]" if off else f"[{reg.text}]"

    def swizzle(self, mode: str, flat: Val) -> Val:
        """Emmy's slab swizzle ``e ^ (((e >> shift) & mask) << 3)``."""
        base, _, shift = (mode or "NONE").partition("@")
        if base not in SWIZZLE_XOR:
            if base not in ("NONE", ""):
                raise NotImplementedError(f"swizzle {mode!r} is not supported")
            return flat
        sh, mask = SWIZZLE_XOR[base]
        sh = int(shift) if shift else sh
        ar = self.ar
        bits = ar.binop("<<", ar.binop("&", ar.binop(">>", flat, imm_int(sh)), imm_int(mask)), imm_int(3))
        return ar.binop("^", flat, bits)

    def address(self, buf: Buffer, flat: Val, k: int = 0) -> str:
        """``[addr+imm]`` operand for element ``flat + k`` of ``buf``."""
        es = buf.esize
        if buf.space == "shared":
            if flat.is_const:
                return f"[{buf.symbol}+{buf.offset + (flat.const + k) * es}]"
            addr = self.ar.reg("s32")
            self.em.ins("mad.lo.s32", addr.text, self.ar.materialize(flat).text, str(es), self.ar.materialize(buf.base).text)
            return f"[{addr.text}+{k * es}]" if k else f"[{addr.text}]"
        if flat.is_const:
            off = (flat.const + k) * es
            return f"[{buf.base.text}+{off}]" if off else f"[{buf.base.text}]"
        wide = self.ar.reg("s64")
        if flat.ty in ("s64", "u64"):
            self.em.ins("mul.lo.s64", wide.text, flat.text, str(es))
        else:
            self.em.ins("mul.wide.s32", wide.text, flat.text, str(es))
        addr = self.ar.reg("u64")
        self.em.ins("add.s64", addr.text, buf.base.text, wide.text)
        return f"[{addr.text}+{k * es}]" if k else f"[{addr.text}]"

    def special_symbol(self, symbol: str) -> Val:
        r = self.ar.reg("s32")
        self.em.ins("mov.u32", r.text, symbol)
        return r

    # ============================================================== statements
    def lower_body(self, body) -> None:
        for s in body:
            self.lower(s)

    def lower(self, s) -> None:
        k = kind(s)
        handler = getattr(self, f"_lower_{k}", None)
        if handler is None:
            raise NotImplementedError(f"Kernel IR statement {k} is not supported")
        if not self.quiet and k not in ("Tile", "Loop", "StridedLoop", "Cond"):
            for line in s.pretty():
                self.em.comment(line.strip())
        handler(s)

    # -- launch geometry -------------------------------------------------------
    def _lower_Tile(self, s) -> None:
        """Bind the Tile's axes from the thread's linear id ``gid = ctaid * C + tid``.

        ``0 <= tid < C`` lets most axes decode from ``tid`` or ``ctaid`` alone:
        an axis whose span ``stride*extent`` divides C lives inside one CTA, and
        one whose stride is a multiple of C is constant across it.
        """
        ar, em = self.ar, self.em
        extents = [static_extent(a) for a in s.axes]
        n = math.prod(extents) if extents else 1
        c = s.block_threads or self.src.block or DEFAULT_BLOCK
        em.comment(f"Tile[{', '.join(a.name for a in s.axes)}] (N={n}), {c} cells per CTA")
        tid, ctaid = self.special("%tid.x"), self.special("%ctaid.x")
        if n > INT32_MAX:
            tid, ctaid = ar.convert(tid, "s64"), ar.convert(ctaid, "s64")
        t = ar.binop("%", tid, imm_int(c)) if s.aux_threads else tid
        gid = ar.add(ar.mul(ctaid, imm_int(c)), t)
        if s.aux_threads or n % c:
            em.pred_ins(ar.cmp(">=", gid, imm_int(n)).text, "bra", self.exit_label)
        strides = [math.prod(extents[i + 1:]) for i in range(len(extents))]
        raster = self._raster_plan(s, extents)
        with self.scope():
            for i, (axis, e, st) in enumerate(zip(s.axes, extents, strides)):
                if raster is not None and i in raster:
                    self.env[axis.name] = raster[i](gid)
                elif e == 1:
                    self.env[axis.name] = imm_int(0)
                elif c % (st * e) == 0:  # the axis lives inside one CTA
                    v = ar.binop("/", t, imm_int(st))
                    self.env[axis.name] = v if st * e == c else ar.binop("%", v, imm_int(e))
                elif st % c == 0:  # constant across the CTA
                    v = ar.binop("/", ctaid, imm_int(st // c))
                    self.env[axis.name] = v if i == 0 else ar.binop("%", v, imm_int(e))
                else:
                    v = ar.binop("/", gid, imm_int(st))
                    self.env[axis.name] = v if i == 0 else ar.binop("%", v, imm_int(e))
                self.ranges[self.env[axis.name].text] = (0, e - 1)
            self.lower_body(s.body)

    def _raster_plan(self, s, extents):
        """Grouped CTA rasterization of the (m, n) block axes (the standard threadblock swizzle)."""
        if not (s.raster_group and s.raster_axes):
            return None
        names = [a.name for a in s.axes]
        if s.raster_axes[0] not in names or s.raster_axes[1] not in names:
            return None
        mi, ni = names.index(s.raster_axes[0]), names.index(s.raster_axes[1])
        if ni != mi + 1:
            return None
        gi, oi = (mi, ni) if s.raster_orient == "m" else (ni, mi)
        g = min(s.raster_group, extents[gi])
        if extents[gi] <= 1 or g <= 1:
            return None
        ar = self.ar
        ea, eb = extents[gi], extents[oi]
        sn = math.prod(extents[ni + 1:])
        cache: dict[str, Val] = {}

        def decode(gid: Val) -> tuple[Val, Val]:
            if "a" not in cache:
                sub = ar.binop("/", gid, imm_int(sn))
                if mi != 0:
                    sub = ar.binop("%", sub, imm_int(ea * eb))
                inner = ar.binop("%", sub, imm_int(g * eb))
                if ea % g == 0:
                    cache["a"] = ar.add(ar.mul(ar.binop("/", sub, imm_int(g * eb)), imm_int(g)), ar.binop("%", sub, imm_int(g)))
                    cache["b"] = ar.binop("/", inner, imm_int(g))
                else:
                    grp = ar.binop("/", sub, imm_int(g * eb))
                    size = ar.binop("min", imm_int(g), ar.sub(imm_int(ea), ar.mul(grp, imm_int(g))))
                    cache["a"] = ar.add(ar.mul(grp, imm_int(g)), ar.binop("%", inner, size))
                    cache["b"] = ar.binop("/", inner, size)
            return cache["a"], cache["b"]

        return {gi: lambda gid: decode(gid)[0], oi: lambda gid: decode(gid)[1]}

    def _lower_ZeroPrologue(self, s) -> None:
        """CTA 0 zeroes ``words`` 32-bit words of another kernel's accumulator."""
        ar, em = self.ar, self.em
        buf = self.buffer(s.dst)
        skip, top = em.new_label("zp_skip"), em.new_label("zp_top")
        em.pred_ins(ar.cmp("!=", self.special("%ctaid.x"), imm_int(0)).text, "bra", skip)
        i = ar.reg("s32")
        ar.mov(i, self.special("%tid.x"))
        step = self.special("%ntid.x")
        em.pred_ins(ar.cmp(">=", i, imm_int(s.words)).text, "bra", skip)
        em.label(top)
        word = Buffer(buf.name, "global", (), "u32", base=buf.base)
        em.ins("st.global.u32", self.address(word, i), "0")
        em.ins("add.s32", i.text, i.text, step.text)
        em.pred_ins(ar.cmp("<", i, imm_int(s.words)).text, "bra", top)
        em.label(skip)

    # -- control flow ----------------------------------------------------------
    def _seed_accumulators(self, body, seed: bool) -> None:
        """A reduce loop seeds each Accum it folds with ``op.identity`` before the first trip."""
        seen: set[str] = set()

        def visit(stmts):
            for st in stmts:
                if kind(st) == "Accum" and st.name not in seen:
                    seen.add(st.name)
                    if seed or st.name not in self.env:
                        ty = DTYPES.get(dtype_name(st.dtype), DTYPES["f32"])[2]
                        ident = st.op.identity
                        if ident is None:
                            raise ValueError(f"Accum {st.name!r}: op {st.op.name!r} has no identity")
                        self.new_mutable(st.name, self.ar.const(ident, ty), ty)
            # Accums nested in a Cond are not seeded, as in Emmy's Loop.render

        visit(body)

    def _rehome_accumulators(self, body) -> None:
        """Own every name a nested body accumulates into here, so the update survives the body."""
        for st in walk(body):
            if kind(st) == "Accum" and st.name in self.env and self.env[st.name].text not in self.owned:
                self.mutable_reg(st.name)

    def _trip_count(self, start: Val, step: Val, bound: Val) -> int | None:
        """The loop's trip count when it is the same for every thread, else None.

        A start that is a Tile axis has a known range [lo, hi]; with constant step
        and bound, every thread then runs ceil((bound - start) / step) trips, which
        is uniform whenever the two ends of the range agree (e.g. ``a0_co..1024:128``
        with ``a0_co`` in [0, 128) gives 8 trips for all lanes).
        """
        if not (step.is_const and bound.is_const) or step.const <= 0:
            return None
        if start.is_const:
            return max(0, -(-(bound.const - start.const) // step.const))
        lo, hi = self.ranges.get(start.text, (None, None))
        if lo is None:
            return None
        most = max(0, -(-(bound.const - lo) // step.const))
        least = max(0, -(-(bound.const - hi) // step.const))
        return most if most == least else None

    def _unroll_factor(self, body, trips: int | None, flagged: bool) -> int:
        """How many copies of the body to emit per trip; ``trips`` means fully unrolled.

        Loops Emmy flags ``unroll`` unroll fully within UNROLL_BUDGET; any loop with a
        uniform trip count unrolls fully when tiny, or partially (8/4/2 copies) when
        that keeps the copy small -- more independent loads in flight per trip.
        """
        if trips is None:
            return 1
        leaves = leaf_count(body)
        if trips * leaves <= (UNROLL_BUDGET if flagged else SMALL_UNROLL_LEAVES):
            return trips
        for u in (8, 4, 2):
            if trips % u == 0 and u * leaves <= PARTIAL_UNROLL_LEAVES:
                return u
        return 1

    def _loop(self, axis: str, start: Val, step: Val, bound: Val, body, flagged: bool) -> None:
        """``for (axis = start; axis < bound; axis += step) body`` -- unrolled at compile
        time, partially unrolled, or as a bottom-tested runtime loop."""
        ar, em = self.ar, self.em
        trips = self._trip_count(start, step, bound)
        if trips == 0:
            return
        copies = self._unroll_factor(body, trips, flagged)
        if trips is not None and copies == trips:
            em.comment(f"  (unrolled x{trips})")
            self._emit_copies(axis, [self._offset_val(start, k, step) for k in range(trips)], body)
            return
        top, end = em.new_label(f"{axis}_top"), em.new_label(f"{axis}_end")
        i = ar.reg("s32")
        ar.mov(i, start)
        if start.is_const and step.is_const:
            self.divisor[i.text] = math.gcd(int(start.const), copies * int(step.const))
        if trips is None:
            em.pred_ins(ar.cmp(">=", i, bound).text, "bra", end)
        if copies > 1:
            em.comment(f"  ({trips} trips, {copies} copies per trip)")
        em.label(top)
        with self.scope():
            # not folded into offsets: ptxas then hoists every copy's loads and spills
            self._emit_copies(axis, [ar.add(i, ar.mul(imm_int(k), step)) for k in range(copies)], body)
        ar.mov(i, ar.add(i, ar.mul(imm_int(copies), step)))
        em.pred_ins(ar.cmp("<", i, bound).text, "bra", top)
        em.label(end)

    def _offset_val(self, base: Val, k: int, step: Val) -> Val:
        """base + k*step, recorded as base plus a constant for addressing."""
        v = self.ar.add(base, self.ar.mul(imm_int(k), step))
        if k and step.is_const and not base.is_const and not v.is_const:
            self.offsets[v.text] = (base, k * int(step.const))
        return v

    def _emit_copies(self, axis: str, values: list[Val], body) -> None:
        """Emit ``body`` once per value of ``axis``; each copy gets its own SSA scope."""
        outer = self.env
        self.memo.append({})
        try:
            for n, v in enumerate(values):
                self.env = dict(outer)
                self.env[axis] = v
                if n == 1:
                    self.quiet += 1
                self.lower_body(body)
            if len(values) > 1:
                self.quiet -= 1
        finally:
            self.env = outer
            self.memo.pop()

    def _lower_Loop(self, s) -> None:
        self._seed_accumulators(s.body, s.seed)
        self._rehome_accumulators(s.body)
        trips = static_extent(s.axis)
        self.em.comment(f"for {s.axis.name} in 0..{trips}")
        self._loop(s.axis.name, imm_int(0), imm_int(1), imm_int(trips), s.body, s.unroll)

    def _lower_StridedLoop(self, s) -> None:
        self._seed_accumulators(s.body, s.seed)
        self._rehome_accumulators(s.body)
        name = s.axis.name
        start, step = self.expr(s.start), self.expr(s.step)
        bound = self.expr(s.end) if s.end is not None else imm_int(static_extent(s.axis))
        if s.end is not None:
            self.env[f"{name}_end"] = bound
        self.em.comment(f"for {name} in {s.start.pretty()}..{bound.text}:{step.text}")
        self._loop(name, start, step, bound, s.body, s.unroll)

    def _lower_Cond(self, s) -> None:
        ar, em = self.ar, self.em
        self._rehome_accumulators(tuple(s.body) + tuple(s.else_body))
        cond = ar.convert(self.expr(s.cond), "pred")
        if cond.is_const:  # statically decided (common inside unrolled loops)
            with self.scope():
                self.lower_body(s.body if cond.const else s.else_body)
            return
        em.comment(f"if ({s.cond.pretty()})")
        else_l, end_l = em.new_label("else"), em.new_label("endif")
        em.pred_ins(f"!{cond.text}", "bra", else_l)
        with self.scope():
            self.lower_body(s.body)
        if s.else_body:
            em.ins("bra.uni", end_l)
            em.label(else_l)
            with self.scope():
                self.lower_body(s.else_body)
            em.label(end_l)
        else:
            em.label(else_l)

    # -- scalar bindings -------------------------------------------------------
    def _lower_Assign(self, s) -> None:
        args = [self.lookup(a) for a in s.args]
        val = mathlib.apply(self.ar, s.op.name, args)
        dt = dtype_name(s.dtype, "")
        if dt in ("f16", "bf16"):
            val = self.round_to(val, dt)
        elif dt in DTYPES and DTYPES[dt][2] != val.ty and val.ty != "pred":
            val = self.ar.convert(val, DTYPES[dt][2])
        if s.op.name == "multiply" and val.ty == "f32" and not val.is_const and not any(a.text in self.owned for a in args):
            self.products[val.text] = (self.ar.convert(args[0], "f32"), self.ar.convert(args[1], "f32"))
        self.bind(s.name, val)

    def _lower_Let(self, s) -> None:
        val = self.expr(s.value)
        dt = dtype_name(s.dtype, "")
        if dt in DTYPES:
            val = self.ar.convert(val, DTYPES[dt][2])
        self.bind(s.name, val)

    def _lower_Init(self, s) -> None:
        ty = DTYPES.get(dtype_name(s.dtype), DTYPES["f32"])[2]
        self.new_mutable(s.name, self.ar.const(s.identity, ty), ty)

    def _lower_Select(self, s) -> None:
        branches = list(s.branches)
        val = self.ar.convert(self.lookup(branches[-1].value), "f32")
        for b in reversed(branches[:-1]):
            val = self.ar.select(self.expr(b.select), self.ar.convert(self.lookup(b.value), "f32"), val)
        self.bind(s.name, val)

    def _lower_Accum(self, s) -> None:
        """``name = op(base ?? name, value)`` in the accumulator's own register.
        An add of a fresh product becomes one ``fma``."""
        ar = self.ar
        if s.name not in self.env:
            ty = DTYPES.get(dtype_name(s.dtype), DTYPES["f32"])[2]
            self.new_mutable(s.name, ar.const(s.op.identity, ty), ty)
        target = self.mutable_reg(s.name)
        left = self.lookup(s.base) if s.base is not None else target
        value = self.lookup(s.value)
        if s.op.name == "add" and target.ty == "f32" and value.text in self.products:
            a, b = self.products[value.text]
            result = ar.fma(a, b, left)
        else:
            result = mathlib.apply(ar, s.op.name, [left, value])
        ar.mov(target, result)

    # -- memory ----------------------------------------------------------------
    def load_value(self, buf: Buffer, addr: str) -> Val:
        """One scalar load, converted to the register type the value is computed in."""
        ar, em = self.ar, self.em
        nbytes, ldty, vty = DTYPES[buf.dtype]
        space = "global.nc" if buf.space == "global" and buf.readonly else buf.space
        if ldty == "b16":
            h, r = ar.reg("b16"), ar.reg("f32")
            em.ins(f"ld.{space}.b16", h.text, addr)
            em.ins("cvt.f32.f16" if buf.dtype == "f16" else "cvt.f32.bf16", r.text, h.text)
            return r
        r = ar.reg(vty)
        em.ins(f"ld.{space}.{ldty}", r.text, addr)
        return r

    def store_value(self, buf: Buffer, addr: str, val: Val, atomic: bool = False) -> None:
        ar, em = self.ar, self.em
        nbytes, stty, vty = DTYPES[buf.dtype]
        if atomic:
            if buf.dtype not in ("f32", "i32", "u32"):
                raise NotImplementedError(f"atomic add on {buf.dtype} is not supported")
            v = ar.materialize(ar.convert(val, vty))
            em.ins(f"red.{buf.space}.add.{'f32' if vty == 'f32' else 's32'}", addr, v.text)
            return
        if stty == "b16":
            f = ar.materialize(ar.convert(val, "f32"))
            out = ar.reg("b16")
            cvt = {"f16": "cvt.rn.f16.f32", "bf16": "cvt.rn.bf16.f32"}[buf.dtype]
            em.ins(cvt, out.text, f.text)
            em.ins(f"st.{buf.space}.{stty}", addr, out.text)
            return
        v = ar.materialize(ar.convert(val, vty))
        em.ins(f"st.{buf.space}.{stty}", addr, v.text)

    def _lower_Load(self, s) -> None:
        buf = self.buffer(s.input)
        if buf.space == "literal":
            for name in s.names:
                self.bind(name, imm_f64(buf.literal) if buf.dtype == "f64" else imm_f32(buf.literal))
            return
        var, const, g = self.split_index(s.input, s.index)
        n = len(s.names)
        if n in (2, 4) and buf.dtype in ("f32", "i32"):
            vty = DTYPES[buf.dtype][2]
            regs = [self.ar.reg(vty) for _ in range(n)]
            space = "global.nc" if buf.space == "global" and buf.readonly else buf.space
            self.em.ins(f"ld.{space}.v{n}.{DTYPES[buf.dtype][1]}", "{" + ", ".join(r.text for r in regs) + "}", self.address_of(buf, var, const, g))
            for name, r in zip(s.names, regs):
                self.bind(name, r)
            return
        for k, name in enumerate(s.names):
            self.bind(name, self.load_value(buf, self.address_of(buf, var, const, g, k)))

    def _lower_Write(self, s) -> None:
        buf = self.buffer(s.output)
        vals = [self.lookup(v) for v in s.values]
        swz = getattr(s, "swizzle", "NONE") or "NONE"
        if swz != "NONE":
            flat = self.flat_index(s.output, s.index)
            for k, v in enumerate(vals):
                self.store_value(buf, self.address(buf, self.swizzle(swz, self.ar.add(flat, imm_int(k)))), v, s.atomic)
            return
        var, const, g = self.split_index(s.output, s.index)
        n = len(vals)
        if n in (2, 4) and buf.dtype in ("f32", "i32") and not s.atomic:
            vty = DTYPES[buf.dtype][2]
            regs = [self.ar.materialize(self.ar.convert(v, vty)) for v in vals]
            self.em.ins(f"st.{buf.space}.v{n}.{DTYPES[buf.dtype][1]}", self.address_of(buf, var, const, g), "{" + ", ".join(r.text for r in regs) + "}")
            return
        for k, v in enumerate(vals):
            self.store_value(buf, self.address_of(buf, var, const, g, k), v, s.atomic)

    def _lower_Smem(self, s) -> None:
        pass  # laid out up front by _plan_shared_memory

    # -- barriers and asynchronous copies ---------------------------------------
    def _lower_Sync(self, s) -> None:
        if s.warp:
            self.em.ins("bar.warp.sync", "-1")
        elif s.barrier_id == 0:
            self.em.ins("bar.sync", "0")
        else:
            if s.count is None:
                raise ValueError(f"Sync(barrier_id={s.barrier_id}) requires a thread count")
            self.em.ins("bar.sync", str(s.barrier_id), str(s.count))

    def _lower_CpAsyncCopy(self, s) -> None:
        ar = self.ar
        dst_buf, src_buf = self.buffer(s.smem), self.buffer(s.src)
        dst = self.flat_index(s.smem, s.smem_index)
        if s.lane_index is not None:
            cols = dst_buf.shape[-1]
            lane = ar.add(ar.mul(self.expr(s.lane_index[0]), imm_int(cols)), self.expr(s.lane_index[1]))
            dst = ar.add(dst, lane)
        dst = self.swizzle(s.swizzle, dst)
        src = self.flat_index(s.src, s.src_index)
        cache = "cg" if s.nbytes == 16 else "ca"
        operands = [self.address(dst_buf, dst), self.address(src_buf, src), str(s.nbytes)]
        valid = getattr(s, "valid", None)
        if valid is not None:  # zero-fill: copy 0 source bytes when the lane is out of range
            operands.append(ar.materialize(ar.select(self.expr(valid), imm_int(s.nbytes, "u32"), imm_int(0, "u32"))).text)
        self.em.ins(f"cp.async.{cache}.shared.global", *operands)

    def _lower_CpAsyncCommit(self, s) -> None:
        self.em.ins("cp.async.commit_group")

    def _lower_CpAsyncWait(self, s) -> None:
        self.em.ins("cp.async.wait_group", str(s.group))

    # -- cross-thread combines -------------------------------------------------
    def _merge(self, program, state, env: dict[str, Val]) -> None:
        """Run a ``combine_states`` program; assignments to ``state`` names write their registers."""
        saved = self.env
        self.env = env
        try:
            for a in program:
                val = mathlib.apply(self.ar, a.op.name, [self.lookup(x) for x in a.args])
                if a.name in state:
                    self.ar.mov(env[a.name], val)
                else:
                    env[a.name] = val
        finally:
            self.env = saved

    def _lower_WarpShuffle(self, s) -> None:
        """XOR butterfly over ``length`` lanes: shuffle every state component, then merge."""
        ar, em = self.ar, self.em
        mask = ar.reg("u32")
        em.ins("activemask.b32", mask.text)
        regs = {st: self.mutable_reg(st) for st in s.state}
        if getattr(s, "prefix", False):
            raise NotImplementedError("prefix WarpShuffle (scan) is not supported")
        offset = int(s.length) // 2
        while offset > 0:
            env = dict(self.env)
            for st, sb in zip(s.state, s.state_b):
                other = ar.reg(regs[st].ty)
                if regs[st].ty in ("s64", "u64", "f64"):  # shfl is 32-bit
                    lo, hi, lo2, hi2 = (ar.reg("s32") for _ in range(4))
                    em.ins("mov.b64", f"{{{lo.text}, {hi.text}}}", regs[st].text)
                    em.ins("shfl.sync.bfly.b32", lo2.text, lo.text, str(offset), "0x1f", mask.text)
                    em.ins("shfl.sync.bfly.b32", hi2.text, hi.text, str(offset), "0x1f", mask.text)
                    em.ins("mov.b64", other.text, f"{{{lo2.text}, {hi2.text}}}")
                else:
                    em.ins("shfl.sync.bfly.b32", other.text, regs[st].text, str(offset), "0x1f", mask.text)
                env[sb] = other
            self._merge(s.combine_states, set(s.state), env)
            offset //= 2

    def _lower_TreeHalve(self, s) -> None:
        """Halving tree over pre-filled shared slabs, a barrier per level, then broadcast slot 0."""
        ar, em = self.ar, self.em
        tid = self.lookup(s.tid_var)
        bufs = [self.buffer(b) for b in s.bufs]
        inner = getattr(s, "inner", None)
        if s.barrier_id and s.barrier_count is None:
            raise ValueError(f"TreeHalve(barrier_id={s.barrier_id}) requires barrier_count")

        def slot(t: Val) -> Val:
            if inner is None:
                return t
            iv, scale = inner
            return ar.add(ar.mul(t, imm_int(scale)), self.lookup(iv))

        step = int(s.length) // 2
        while step > 0:
            skip = em.new_label("tree_skip")
            em.pred_ins(ar.cmp(">=", tid, imm_int(step)).text, "bra", skip)
            mine, theirs = slot(tid), slot(ar.add(tid, imm_int(step)))
            env = dict(self.env)
            for buf, st, sb in zip(bufs, s.state, s.state_b):
                env[st] = self.load_value(buf, self.address(buf, mine))
                env[sb] = self.load_value(buf, self.address(buf, theirs))
            self._merge(s.combine_states, set(s.state), env)
            for buf, st in zip(bufs, s.state):
                self.store_value(buf, self.address(buf, mine), env[st])
            em.label(skip)
            if s.barrier_id:
                em.ins("bar.sync", str(s.barrier_id), str(s.barrier_count))
            else:
                em.ins("bar.sync", "0")
            step //= 2
        root = self.lookup(inner[0]) if inner is not None else imm_int(0)
        for buf, st in zip(bufs, s.state):
            ar.mov(self.mutable_reg(st), self.load_value(buf, self.address(buf, root)))


def compile_kernel(src: KernelSource) -> str:
    return KernelCompiler(src).compile()
