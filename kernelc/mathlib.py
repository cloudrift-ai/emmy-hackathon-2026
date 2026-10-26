"""Elementwise operations (Emmy's ``ElementwiseImpl`` names) lowered to PTX.

Arithmetic that PTX has a correctly rounded instruction for (``add``,
``divide``, ``sqrt``, ``reciprocal``, ...) maps straight onto it. The
transcendental functions are built from range reduction plus a short
polynomial, because the hardware ``ex2.approx`` / ``lg2.approx`` /
``sin.approx`` units are only accurate to ~2^-22 absolute, which is not close
enough to numpy's float32 results for the tight tolerances of pointwise
problems. Each routine documents the identity it relies on.
"""

from __future__ import annotations

import math
from collections.abc import Callable

from .arith import Arith
from .ptx import Val, imm_f32, imm_f64, imm_int

C = imm_f32

LOG2E = 1.4426950408889634
LN2 = 0.6931471805599453
# Cody-Waite split of ln(2): the high part has trailing zero bits so n*LN2_HI is exact.
EXP_LN2_HI = 0.693145751953125  # 0x3F317200
EXP_LN2_LO = 1.428606765330187e-06  # 0x35BFBE8E
# musl logf constants
LOG_LN2_HI = 6.9313812256e-01  # 0x3f317180
LOG_LN2_LO = 9.0580006145e-06  # 0x3717f7d1
LG1 = 0xAAAAAA / 2**24
LG2 = 0xCCCE13 / 2**25
LG3 = 0x91E9EE / 2**25
LG4 = 0xF89E26 / 2**26
# pi/2 split into three parts (Cephes) for the sin/cos argument reduction
PIO2_1 = 1.5703125
PIO2_2 = 4.837512969970703125e-4
PIO2_3 = 7.54978995489188216e-8


def horner(ar: Arith, x: Val, coeffs: list[float]) -> Val:
    """Evaluate c0*x^n + ... + cn with fused multiply-adds (highest degree first)."""
    acc = C(coeffs[0])
    for c in coeffs[1:]:
        acc = ar.fma(acc, x, C(c))
    return acc


def bitcast(ar: Arith, v: Val, ty: str) -> Val:
    v = ar.materialize(v)
    r = ar.reg(ty)
    ar.em.ins("mov.b32", r.text, v.text)
    return r


def pow2i(ar: Arith, n: Val) -> Val:
    """2^n for an integer n in [-126, 127], built directly from the exponent bits."""
    biased = ar.binop("<<", ar.add(n, imm_int(127)), imm_int(23))
    return bitcast(ar, biased, "f32")


# ----------------------------------------------------------------- exp family
def _exp_reduced(ar: Arith, n: Val, r: Val) -> Val:
    """e^r * 2^n for |r| <= ln2/2 and integral float n.

    e^r uses its Taylor series to degree 7 (truncation < 6e-9 relative). The
    scale 2^n is applied in two halves so results in the subnormal range and
    overflow to infinity both come out right.
    """
    p = horner(ar, r, [1 / 5040, 1 / 720, 1 / 120, 1 / 24, 1 / 6, 0.5, 1.0, 1.0])
    ni = ar.convert(n, "s32")
    n1 = ar.binop(">>", ni, imm_int(1))
    n2 = ar.sub(ni, n1)
    return ar.mul(ar.mul(p, pow2i(ar, n1)), pow2i(ar, n2))


def _keep_nan(ar: Arith, x: Val, y: Val) -> Val:
    """min/max in the input clamp drop NaN; restore it."""
    if x.is_const:
        return y
    return ar.select(ar.isnan(x), x, y)


def exp(ar: Arith, x: Val) -> Val:
    """e^x = 2^n * e^r with n = rint(x / ln2), r = x - n*ln2 (Cody-Waite).
    Under fast_math, ex2.approx.ftz(x*log2e)."""
    if ar.fast_math:
        return ar.unary("ex2.approx.ftz", ar.mul(x, C(LOG2E)))
    xc = ar.fmin(ar.fmax(x, C(-104.0)), C(89.0))  # e^-104 underflows, e^89 overflows
    n = ar.unary("cvt.rni.f32", ar.mul(xc, C(LOG2E)))
    r = ar.fma(n, C(-EXP_LN2_HI), xc)
    r = ar.fma(n, C(-EXP_LN2_LO), r)
    return _keep_nan(ar, x, _exp_reduced(ar, n, r))


def exp2(ar: Arith, x: Val) -> Val:
    """2^x = 2^n * e^((x-n)*ln2); x - n is exact."""
    if ar.fast_math:
        return ar.unary("ex2.approx.ftz", x)
    xc = ar.fmin(ar.fmax(x, C(-150.0)), C(129.0))
    n = ar.unary("cvt.rni.f32", xc)
    return _keep_nan(ar, x, _exp_reduced(ar, n, ar.mul(ar.sub(xc, n), C(LN2))))


def reciprocal(ar: Arith, x: Val) -> Val:
    return ar.unary("rcp.approx.ftz" if ar.fast_math else "rcp.rn", x)


def sqrt(ar: Arith, x: Val) -> Val:
    return ar.unary("sqrt.approx.ftz" if ar.fast_math else "sqrt.rn", x)


def rsqrt(ar: Arith, x: Val) -> Val:
    if ar.fast_math:
        return ar.unary("rsqrt.approx.ftz", x)
    return ar.unary("rcp.rn", ar.unary("sqrt.rn", x))


def expm1(ar: Arith, x: Val) -> Val:
    """e^x - 1: Taylor series near zero (no cancellation), exp(x) - 1 elsewhere."""
    small = horner(ar, x, [1 / 362880, 1 / 40320, 1 / 5040, 1 / 720, 1 / 120, 1 / 24, 1 / 6, 0.5, 1.0])
    small = ar.mul(small, x)
    big = ar.sub(exp(ar, x), C(1.0))
    return ar.select(ar.cmp("<", ar.unary("abs", x), C(0.5)), small, big)


# ----------------------------------------------------------------- log family
def log(ar: Arith, x: Val) -> Val:
    """Natural log after musl's logf: x = 2^k * (1+f) with 1+f in [sqrt(2)/2, sqrt(2)),
    log(1+f) = 2*atanh(s), s = f/(2+f), evaluated as f - hfsq + s*(hfsq + R(s^2)).

    Zero, negative, infinite, NaN and subnormal inputs take the hardware
    ``lg2.approx`` path, which produces the right special values.
    """
    ix = bitcast(ar, x, "s32")
    ix = ar.add(ix, imm_int(0x3F800000 - 0x3F3504F3))
    k = ar.sub(ar.binop(">>", ix, imm_int(23)), imm_int(0x7F))
    m = bitcast(ar, ar.add(ar.binop("&", ix, imm_int(0x007FFFFF)), imm_int(0x3F3504F3)), "f32")
    f = ar.sub(m, C(1.0))
    s = ar.div(f, ar.add(C(2.0), f))
    z = ar.mul(s, s)
    w = ar.mul(z, z)
    t1 = ar.mul(w, ar.fma(w, C(LG4), C(LG2)))
    t2 = ar.mul(z, ar.fma(w, C(LG3), C(LG1)))
    big_r = ar.add(t2, t1)
    hfsq = ar.mul(ar.mul(C(0.5), f), f)
    dk = ar.convert(k, "f32")
    res = ar.add(ar.mul(s, ar.add(hfsq, big_r)), ar.mul(dk, C(LOG_LN2_LO)))
    res = ar.add(ar.sub(res, hfsq), f)
    res = ar.add(res, ar.mul(dk, C(LOG_LN2_HI)))
    approx = ar.mul(ar.unary("lg2.approx", ar.materialize(x)), C(LN2))
    normal = ar.logic("&&", ar.cmp(">=", x, C(1.1754943508222875e-38)), ar.cmp("<=", x, C(3.4028234663852886e38)))
    return ar.select(normal, res, approx)


def log1p(ar: Arith, x: Val) -> Val:
    """log(1+x) = log(u) * x / (u - 1) with u = 1 + x (Goldberg's correction)."""
    u = ar.add(C(1.0), x)
    corrected = ar.mul(log(ar, u), ar.div(x, ar.sub(u, C(1.0))))
    return ar.select(ar.cmp("==", u, C(1.0)), x, corrected)


# ------------------------------------------------------------- trigonometry
def _sincos_reduce(ar: Arith, x: Val) -> tuple[Val, Val, Val]:
    """x = q*pi/2 + r, |r| <= pi/4; returns (r, r*r, q & 3)."""
    q = ar.unary("cvt.rni.f32", ar.mul(x, C(2 / math.pi)))
    r = ar.fma(q, C(-PIO2_1), x)
    r = ar.fma(q, C(-PIO2_2), r)
    r = ar.fma(q, C(-PIO2_3), r)
    quadrant = ar.binop("&", ar.convert(q, "s32"), imm_int(3))
    return r, ar.mul(r, r), quadrant


def _sin_poly(ar: Arith, r: Val, z: Val) -> Val:
    return ar.fma(ar.mul(z, r), horner(ar, z, [1 / 362880, -1 / 5040, 1 / 120, -1 / 6]), r)


def _cos_poly(ar: Arith, z: Val) -> Val:
    return horner(ar, z, [-1 / 3628800, 1 / 40320, -1 / 720, 1 / 24, -0.5, 1.0])


def _quadrant_pick(ar: Arith, quadrant: Val, vals: list[Val]) -> Val:
    out = vals[3]
    for q in (2, 1, 0):
        out = ar.select(ar.cmp("==", quadrant, imm_int(q)), vals[q], out)
    return out


def sin(ar: Arith, x: Val) -> Val:
    r, z, q = _sincos_reduce(ar, x)
    s, c = _sin_poly(ar, r, z), _cos_poly(ar, z)
    return _quadrant_pick(ar, q, [s, c, ar.unary("neg", s), ar.unary("neg", c)])


def cos(ar: Arith, x: Val) -> Val:
    r, z, q = _sincos_reduce(ar, x)
    s, c = _sin_poly(ar, r, z), _cos_poly(ar, z)
    return _quadrant_pick(ar, q, [c, ar.unary("neg", s), ar.unary("neg", c), s])


def tanh(ar: Arith, x: Val) -> Val:
    """tanh|x| = t / (t + 2) with t = expm1(2|x|) (accurate near 0), or 1 - 2/(t + 2)
    for |x| > ln(3)/2 (musl); the sign of x is restored at the end."""
    a = ar.unary("abs", x)
    t = expm1(ar, ar.mul(C(2.0), a))
    small = ar.div(t, ar.add(t, C(2.0)))
    big = ar.sub(C(1.0), ar.div(C(2.0), ar.add(t, C(2.0))))
    mag = ar.select(ar.cmp(">", a, C(0.5493061443340549)), big, small)
    return copysign(ar, mag, x)


def copysign(ar: Arith, mag: Val, sign: Val) -> Val:
    m = ar.binop("&", bitcast(ar, mag, "s32"), imm_int(0x7FFFFFFF))
    s = ar.binop("&", bitcast(ar, sign, "s32"), imm_int(-0x80000000))
    return bitcast(ar, ar.binop("|", m, s), "f32")


def erf(ar: Arith, x: Val) -> Val:
    """erf after N. Juffa's single-precision erff (max error ~1 ulp): an odd polynomial
    for |x| < 0.9277, and 1 - exp(poly(|x|)) beyond it."""
    t = ar.unary("abs", x)
    s = ar.mul(x, x)
    # |x| >= 0.927734375
    r = ar.fma(C(-1.72853470e-5), t, C(3.83197126e-4))
    u = ar.fma(C(-3.88396438e-3), t, C(2.42546219e-2))
    r = ar.fma(r, s, u)
    r = ar.fma(r, t, C(-1.06777877e-1))
    r = ar.fma(r, t, C(-6.34846687e-1))
    r = ar.fma(r, t, C(-1.28717512e-1))
    r = ar.fma(r, t, ar.unary("neg", t))
    big = copysign(ar, ar.sub(C(1.0), exp(ar, r)), x)
    # |x| < 0.927734375
    p = horner(ar, s, [-5.96761703e-4, 4.99119423e-3, -2.67681349e-2, 1.12819925e-1, -3.76125336e-1, 1.28379166e-1])
    small = ar.fma(p, x, x)
    return ar.select(ar.cmp(">", t, C(0.927734375)), big, small)


# ------------------------------------------------------------------ powers
def power(ar: Arith, x: Val, y: Val) -> Val:
    if y.is_const:
        e = y.const
        if e == 1.0:
            return x
        if e == 2.0:
            return ar.mul(x, x)
        if e == 3.0:
            return ar.mul(ar.mul(x, x), x)
        if e == 0.5:
            return ar.unary("sqrt.rn", x)
        if e == -1.0:
            return ar.unary("rcp.rn", x)
        if e == 0.0:
            return C(1.0)
    ax = ar.unary("abs", x)
    mag = exp(ar, ar.mul(y, log(ar, ax)))
    y_int = ar.cmp("==", ar.unary("cvt.rzi.f32", y), y)
    half = ar.mul(y, C(0.5))
    y_odd = ar.logic("&&", y_int, ar.cmp("!=", ar.unary("cvt.rzi.f32", half), half))
    neg = ar.cmp("<", x, C(0.0))
    signed = ar.select(ar.logic("&&", neg, y_odd), ar.unary("neg", mag), mag)
    res = ar.select(ar.logic("&&", neg, ar.lnot(y_int)), C(float("nan")), signed)
    zero = ar.select(ar.cmp(">", y, C(0.0)), C(0.0), ar.select(ar.cmp("==", y, C(0.0)), C(1.0), C(float("inf"))))
    return ar.select(ar.cmp("==", x, C(0.0)), zero, res)


# ----------------------------------------------------------------- dispatch
def _flag(ar: Arith, p: Val) -> Val:
    """A predicate as numpy's float mask convention (1.0 / 0.0)."""
    return ar.convert(p, "f32")


def _truthy(ar: Arith, v: Val) -> Val:
    return ar.cmp("!=", v, C(0.0)) if v.ty == "f32" else ar.convert(v, "pred")


def _sigmoid(ar: Arith, x: Val) -> Val:
    return ar.div(C(1.0), ar.add(C(1.0), exp(ar, ar.unary("neg", x))))


def _gelu(ar: Arith, x: Val) -> Val:
    inner = erf(ar, ar.div(x, C(math.sqrt(2.0))))
    return ar.mul(ar.mul(C(0.5), x), ar.add(C(1.0), inner))


def _gelu_tanh(ar: Arith, x: Val) -> Val:
    cube = ar.mul(ar.mul(x, x), x)
    inner = ar.mul(C(math.sqrt(2.0 / math.pi)), ar.add(x, ar.mul(C(0.044715), cube)))
    return ar.mul(ar.mul(C(0.5), x), ar.add(C(1.0), tanh(ar, inner)))


def _softplus(ar: Arith, x: Val) -> Val:
    """logaddexp(0, x) = max(x, 0) + log1p(exp(-|x|))."""
    tail = log1p(ar, exp(ar, ar.unary("neg", ar.unary("abs", x))))
    return ar.add(ar.fmax(x, C(0.0)), tail)


def _sign(ar: Arith, x: Val) -> Val:
    return ar.sub(_flag(ar, ar.cmp(">", x, C(0.0))), _flag(ar, ar.cmp("<", x, C(0.0))))


def _cmp_op(op: str) -> Callable:
    return lambda ar, a, b: _flag(ar, ar.cmp(op, a, b))


def _logic_op(op: str) -> Callable:
    return lambda ar, a, b: _flag(ar, ar.logic(op, _truthy(ar, a), _truthy(ar, b)))


FLOAT_OPS: dict[str, Callable] = {
    # arithmetic
    "add": lambda ar, a, b: ar.add(a, b),
    "subtract": lambda ar, a, b: ar.sub(a, b),
    "multiply": lambda ar, a, b: ar.mul(a, b),
    "divide": lambda ar, a, b: ar.div(a, b),
    "true_divide": lambda ar, a, b: ar.div(a, b),
    "floor_divide": lambda ar, a, b: ar.floor_div(a, b),
    "remainder": lambda ar, a, b: ar.fmod(a, b),
    "mod": lambda ar, a, b: ar.fmod(a, b),
    "fmod": lambda ar, a, b: ar.fmod_c(a, b),
    "maximum": lambda ar, a, b: ar.fmax(a, b),
    "minimum": lambda ar, a, b: ar.fmin(a, b),
    "fmax": lambda ar, a, b: ar.fmax(a, b),
    "fmin": lambda ar, a, b: ar.fmin(a, b),
    "power": power,
    "pow": power,
    "copysign": copysign,
    # unary arithmetic
    "copy": lambda ar, x: x,
    "pad": lambda ar, x: x,
    "negative": lambda ar, x: ar.unary("neg", x),
    "abs": lambda ar, x: ar.unary("abs", x),
    "absolute": lambda ar, x: ar.unary("abs", x),
    "square": lambda ar, x: ar.mul(x, x),
    "reciprocal": reciprocal,
    "sqrt": sqrt,
    "rsqrt": rsqrt,
    "relu": lambda ar, x: ar.fmax(x, C(0.0)),
    "sign": _sign,
    "floor": lambda ar, x: ar.unary("cvt.rmi.f32", x),
    "ceil": lambda ar, x: ar.unary("cvt.rpi.f32", x),
    "trunc": lambda ar, x: ar.unary("cvt.rzi.f32", x),
    "rint": lambda ar, x: ar.unary("cvt.rni.f32", x),
    "round": lambda ar, x: ar.unary("cvt.rni.f32", x),
    # transcendental
    "exp": exp,
    "exp_fast": exp,
    "exp2": exp2,
    "expm1": expm1,
    "log": log,
    "log2": lambda ar, x: ar.mul(log(ar, x), C(LOG2E)),
    "log10": lambda ar, x: ar.mul(log(ar, x), C(1 / math.log(10.0))),
    "log1p": log1p,
    "sin": sin,
    "cos": cos,
    "tan": lambda ar, x: ar.div(sin(ar, x), cos(ar, x)),
    "tanh": tanh,
    "erf": erf,
    "sigmoid": _sigmoid,
    "silu": lambda ar, x: ar.div(x, ar.add(C(1.0), exp(ar, ar.unary("neg", x)))),
    "softplus": _softplus,
    "gelu": _gelu,
    "gelu_tanh": _gelu_tanh,
    # comparisons and masks (numpy float mask convention)
    "greater": _cmp_op(">"),
    "less": _cmp_op("<"),
    "greater_equal": _cmp_op(">="),
    "less_equal": _cmp_op("<="),
    "equal": _cmp_op("=="),
    "not_equal": _cmp_op("!="),
    "logical_and": _logic_op("&&"),
    "logical_or": _logic_op("||"),
    "logical_xor": _logic_op("^"),
    "bitwise_and": _logic_op("&&"),
    "bitwise_or": _logic_op("||"),
    "logical_not": lambda ar, x: _flag(ar, ar.lnot(_truthy(ar, x))),
    "bitwise_not": lambda ar, x: _flag(ar, ar.lnot(_truthy(ar, x))),
    "where": lambda ar, c, a, b: ar.select(_truthy(ar, c), a, b),
}

# Ops that stay in the integer domain when every argument is an integer.
INT_OPS: dict[str, str] = {
    "add": "+",
    "subtract": "-",
    "multiply": "*",
    "floor_divide": "//",
    "remainder": "%",
    "mod": "%",
    "left_shift": "<<",
    "right_shift": ">>",
    "bitwise_and": "&",
    "bitwise_or": "|",
    "bitwise_xor": "^",
    "maximum": "max",
    "minimum": "min",
}


# f64 runs natively; anything else on f64 raises
F64_OPS: dict[str, Callable] = {
    "add": lambda ar, a, b: ar.add(a, b),
    "subtract": lambda ar, a, b: ar.sub(a, b),
    "multiply": lambda ar, a, b: ar.mul(a, b),
    "divide": lambda ar, a, b: ar.div(a, b),
    "true_divide": lambda ar, a, b: ar.div(a, b),
    "maximum": lambda ar, a, b: ar.fmax(a, b),
    "minimum": lambda ar, a, b: ar.fmin(a, b),
    "negative": lambda ar, x: ar.unary("neg", x, "f64"),
    "abs": lambda ar, x: ar.unary("abs", x, "f64"),
    "absolute": lambda ar, x: ar.unary("abs", x, "f64"),
    "square": lambda ar, x: ar.mul(x, x),
    "sqrt": lambda ar, x: ar.unary("sqrt.rn", x, "f64"),
    "reciprocal": lambda ar, x: ar.unary("rcp.rn", x, "f64"),
    "relu": lambda ar, x: ar.fmax(x, imm_f64(0.0)),
    "where": lambda ar, c, a, b: ar.select(_truthy(ar, c), a, b),
    **{k: FLOAT_OPS[k] for k in ("greater", "less", "greater_equal", "less_equal", "equal", "not_equal")},
}


def apply(ar: Arith, name: str, args: list[Val]) -> Val:
    """Lower one elementwise op. Integer stays integral, f64 stays f64, the rest runs in f32."""
    if name in INT_OPS and args and all(a.is_int for a in args):
        return ar.binop(INT_OPS[name], args[0], args[1])
    if name in ("copy", "pad") and len(args) == 1:
        return args[0]
    if any(a.ty == "f64" for a in args):
        fn = F64_OPS.get(name)
        if fn is None:
            raise NotImplementedError(f"elementwise op {name!r} on f64 is not supported (it would be computed in f32)")
        if name == "where":
            return fn(ar, args[0], ar.convert(args[1], "f64"), ar.convert(args[2], "f64"))
        return fn(ar, *[ar.convert(a, "f64") for a in args])
    fn = FLOAT_OPS.get(name)
    if fn is None:
        raise NotImplementedError(f"elementwise op {name!r} is not supported")
    if name == "where":
        return fn(ar, args[0], ar.convert(args[1], "f32"), ar.convert(args[2], "f32"))
    return fn(ar, *[ar.convert(a, "f32") for a in args])
