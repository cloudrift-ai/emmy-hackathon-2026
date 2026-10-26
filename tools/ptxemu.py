#!/usr/bin/env python3
"""CPU emulator for the PTX kernelc emits, checked against expected.npz like submit.py.

    python tools/ptxemu.py <problem_dir>... [--compiler compile.py] [--ptx out.ptx]

Also reports out-of-bounds and misaligned accesses, shared-memory races, partial barriers,
uninitialized registers and cp.async data read before its wait.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

np.seterr(all="ignore")

NP = {"f32": np.float32, "f64": np.float64, "s32": np.int32, "u32": np.uint32, "b32": np.uint32,
      "s64": np.int64, "u64": np.uint64, "b64": np.uint64, "s16": np.int16, "u16": np.uint16,
      "b16": np.uint16, "f16": np.uint16, "bf16": np.uint16, "s8": np.int8, "u8": np.uint8,
      "b8": np.uint8, "pred": np.bool_}
STORE = {"%fd": np.float64, "%f": np.float32, "%rd": np.int64, "%r": np.int32, "%p": np.bool_, "%h": np.uint16}
DT = {"f32": np.float32, "f64": np.float64, "f16": np.float16, "i32": np.int32, "i64": np.int64,
      "u32": np.uint32, "u8": np.uint8, "bool": np.bool_}


class EmuError(Exception):
    pass


def reg_class(name: str):
    for p in ("%fd", "%rd", "%f", "%r", "%p", "%h"):
        if name.startswith(p) and name[len(p):].isdigit():
            return STORE[p]
    raise EmuError(f"unknown register {name}")


def split_ops(s: str) -> list[str]:
    out, depth, cur = [], 0, ""
    for ch in s:
        if ch in "[{":
            depth += 1
        elif ch in "]}":
            depth -= 1
        if ch == "," and depth == 0:
            out.append(cur.strip())
            cur = ""
        else:
            cur += ch
    if cur.strip():
        out.append(cur.strip())
    return out


class Kernel:
    def __init__(self, ptx: str):
        self.instrs: list[tuple] = []
        self.labels: dict[str, int] = {}
        self.params: list[tuple[str, str]] = []
        self.shared: dict[str, int] = {}
        self.static_smem = 0
        self.extern = None
        self.entry = None
        in_params = False
        for raw in ptx.splitlines():
            line = raw.split("//")[0].strip()
            if not line:
                continue
            m = re.match(r"\.visible \.entry (\w+)\(", line)
            if m:
                self.entry, in_params = m.group(1), True
                continue
            if in_params:
                m = re.match(r"\.param \.(\w+) (\w+)", line)
                if m:
                    self.params.append((m.group(2), m.group(1)))
                if line.startswith(")"):
                    in_params = False
                continue
            m = re.match(r"\.shared \.align (\d+) \.b8 (\w+)\[(\d+)\];", line)
            if m:
                al, name, size = int(m.group(1)), m.group(2), int(m.group(3))
                self.static_smem = -(-self.static_smem // al) * al
                self.shared[name] = self.static_smem
                self.static_smem += size
                continue
            m = re.match(r"\.extern \.shared \.align (\d+) \.b8 (\w+)\[\];", line)
            if m:
                self.extern = (m.group(2), int(m.group(1)))
                continue
            if line.startswith((".", "{", "}")):
                continue
            if line.endswith(":"):
                self.labels[line[:-1]] = len(self.instrs)
                continue
            line = line.rstrip(";")
            pred = None
            m = re.match(r"@(!?)(%p\d+)\s+(.*)", line)
            if m:
                pred = (m.group(1) == "!", m.group(2))
                line = m.group(3)
            parts = line.split(None, 1)
            self.instrs.append((parts[0], split_ops(parts[1]) if len(parts) > 1 else [], pred, raw.strip()))


class Machine:
    def __init__(self, k: Kernel, grid: int, block: int, smem_bytes: int, args: dict, buffers: dict):
        self.k, self.grid, self.block = k, grid, block
        self.T = grid * block
        t = np.arange(self.T, dtype=np.int64)
        self.cta, self.tidx = t // block, t % block
        self.lane = self.tidx % 32
        self.warp_key = self.cta * (-(-block // 32)) + self.tidx // 32
        self.nwarps = grid * (-(-block // 32))
        if k.extern:
            al = k.extern[1]
            self.extern_off = -(-k.static_smem // al) * al
            self.S = self.extern_off + smem_bytes
            k.shared[k.extern[0]] = self.extern_off
        else:
            self.S = k.static_smem
        self.smem = np.zeros(max(1, grid * self.S), dtype=np.uint8)
        nw = max(1, (grid * self.S + 3) // 4)
        self.epoch = np.zeros(grid, dtype=np.int64)
        BIG = np.iinfo(np.int64).max
        self.race = {k: np.full(nw, v, dtype=np.int64) for k, v in
                     (("wr_ep", -1), ("wr_lo", BIG), ("wr_hi", -1), ("rd_ep", -1), ("rd_lo", BIG), ("rd_hi", -1))}
        # global memory: each buffer gets its own aligned region with a guard gap
        self.bases, self.ends, self.names, chunks, off = [], [], [], [], 1 << 12
        for name, arr in buffers.items():
            b = np.ascontiguousarray(arr).view(np.uint8).ravel()
            self.bases.append(off)
            self.ends.append(off + b.size)
            self.names.append(name)
            chunks.append((off, b))
            off = -(-(off + b.size + 256) // 256) * 256
        self.gmem = np.zeros(off, dtype=np.uint8)
        for o, b in chunks:
            self.gmem[o:o + b.size] = b
        self.bases_np = np.array(self.bases)
        self.param_vals = {}
        for (pname, pty), aname in zip(k.params, args):
            if aname in self.names:
                self.param_vals[pname] = self.bases[self.names.index(aname)]
            else:
                raise EmuError(f"no buffer for parameter {aname}")
        self.regs: dict[str, np.ndarray] = {}
        self.written: dict[str, np.ndarray] = {}
        self.pc = np.zeros(self.T, dtype=np.int64)
        self.alive = np.ones(self.T, dtype=bool)
        self.stored_bufs: set[str] = set()
        self.nc_bufs: set[str] = set()
        self.pending: list[dict] = []
        self.committed = np.zeros(self.T, dtype=np.int64)
        self.issues: list[str] = []

    # ------------------------------------------------------------------ values
    def issue(self, msg: str) -> None:
        if msg not in self.issues:
            self.issues.append(msg)
        if len(self.issues) > 20:
            raise EmuError("too many issues: " + "; ".join(self.issues[:5]))

    def read(self, op: str, idx: np.ndarray, ty: str) -> np.ndarray:
        tgt = NP[ty]
        if op.startswith("%"):
            if op in ("%tid.x", "%ctaid.x", "%ntid.x", "%nctaid.x"):
                v = {"%tid.x": self.tidx, "%ctaid.x": self.cta, "%ntid.x": np.full(self.T, self.block),
                     "%nctaid.x": np.full(self.T, self.grid)}[op][idx]
                return v.astype(tgt)
            if op not in self.regs:
                raise EmuError(f"read of never-written register {op}")
            if not self.written[op][idx].all():
                self.issue(f"read of uninitialized register {op} on some threads")
            raw = self.regs[op][idx]
            if raw.dtype == tgt:
                return raw
            if raw.dtype.itemsize == np.dtype(tgt).itemsize and tgt is not np.bool_:
                return raw.view(tgt)
            return raw.astype(tgt)
        if op.startswith("0f"):
            v = np.array([int(op[2:], 16)], dtype=np.uint32).view(np.float32)[0]
            return np.full(idx.size, v, dtype=np.float32).astype(tgt) if tgt is not np.float32 else np.full(idx.size, v, np.float32)
        if op in self.k.shared:
            return np.full(idx.size, self.k.shared[op], dtype=tgt)
        if op in self.param_vals:
            return np.full(idx.size, self.param_vals[op], dtype=tgt)
        v = int(op, 0)
        if tgt is np.float32 or tgt is np.float64:
            return np.full(idx.size, v, dtype=tgt)
        return np.array([v], dtype=np.int64).astype(tgt).repeat(idx.size) if v < 0 else np.full(idx.size, v).astype(tgt)

    def write(self, op: str, idx: np.ndarray, val: np.ndarray) -> None:
        st = reg_class(op)
        if op not in self.regs:
            self.regs[op] = np.zeros(self.T, dtype=st)
            self.written[op] = np.zeros(self.T, dtype=bool)
        val = np.asarray(val)
        if val.dtype != st:
            if val.dtype.itemsize == np.dtype(st).itemsize and st is not np.bool_ and val.dtype != np.bool_:
                val = val.view(st)
            else:
                val = val.astype(st)
        self.regs[op][idx] = val
        self.written[op][idx] = True

    # ------------------------------------------------------------------ memory
    def addr(self, op: str, idx: np.ndarray) -> np.ndarray:
        m = re.match(r"\[([^+\]]+)(?:\+(-?\d+))?\]", op)
        base, off = m.group(1), int(m.group(2) or 0)
        if base.startswith("%"):
            return self.read(base, idx, "s64").astype(np.int64) + off
        if base in self.k.shared:
            return np.full(idx.size, self.k.shared[base] + off, dtype=np.int64)
        if base in self.param_vals:
            return np.full(idx.size, -1, dtype=np.int64)  # ld.param handled by caller
        raise EmuError(f"bad address {op}")

    def gcheck(self, a: np.ndarray, size: int, what: str) -> np.ndarray:
        if a.size == 0:
            return np.zeros(0, dtype=np.int64)
        if (a % size).any():
            raise EmuError(f"misaligned {what} ({size}-byte access at offset {int(a[(a % size) != 0][0]) % 256} mod 256)")
        b = np.searchsorted(self.bases_np, a, side="right") - 1
        ok = (b >= 0) & (a + size <= np.array(self.ends)[np.clip(b, 0, None)])
        if not ok.all():
            raise EmuError(f"out-of-bounds {what}")
        return b

    def sh(self, a: np.ndarray, idx: np.ndarray, size: int, what: str) -> np.ndarray:
        if a.size and ((a < 0).any() or (a + size > self.S).any()):
            raise EmuError(f"out-of-bounds shared {what} (offset {int(a.max())}, slab {self.S} bytes)")
        if a.size and (a % size).any():
            raise EmuError(f"misaligned shared {what}")
        return self.cta[idx] * self.S + a

    def track(self, g: np.ndarray, idx: np.ndarray, nbytes: int, write: bool) -> None:
        """Race check per 4-byte word between barriers; one warp's own accesses are ordered."""
        if g.size == 0:
            return
        r = self.race
        words = ((g[:, None] + np.arange(0, nbytes, 4)) // 4).ravel()
        wk = np.repeat(self.warp_key[idx], max(1, nbytes // 4) if nbytes >= 4 else 1)[: words.size]
        ep = np.repeat(self.epoch[self.cta[idx]], max(1, nbytes // 4) if nbytes >= 4 else 1)[: words.size]
        def other(kind):
            same = r[kind + "_ep"][words] == ep
            return same & ((r[kind + "_lo"][words] != wk) | (r[kind + "_hi"][words] != wk))
        BIG = np.iinfo(np.int64).max
        if write:
            if other("rd").any():
                self.issue("shared-memory write-after-read race: a slot is overwritten in the same barrier interval another warp read it")
            if other("wr").any():
                self.issue("shared-memory write-write race between warps in one barrier interval")
            kind = "wr"
        else:
            if other("wr").any():
                self.issue("shared-memory read-after-write race: a slot is read in the same barrier interval another warp wrote it")
            kind = "rd"
        stale = r[kind + "_ep"][words] != ep
        r[kind + "_lo"][words[stale]] = BIG
        r[kind + "_hi"][words[stale]] = -1
        r[kind + "_ep"][words] = ep
        np.minimum.at(r[kind + "_lo"], words, wk)
        np.maximum.at(r[kind + "_hi"], words, wk)

    def mem_load(self, space: str, a: np.ndarray, idx: np.ndarray, ty: str, n: int, nc: bool) -> list[np.ndarray]:
        es = np.dtype(NP[ty]).itemsize
        mem = self.gmem if space == "global" else self.smem
        if space == "global":
            b = self.gcheck(a, es * n, f"global load (.v{n}.{ty})" if n > 1 else f"global load .{ty}")
            if nc:
                for i in np.unique(b):
                    self.nc_bufs.add(self.names[i])
            g = a
        else:
            g = self.sh(a, idx, es * n, "load")
            self.track(g, idx, es * n, write=False)
        out = []
        for k in range(n):
            pos = g + k * es
            bytes_ = mem[pos[:, None] + np.arange(es)]
            out.append(bytes_.copy().view(NP[ty]).ravel())
        return out

    def mem_store(self, space: str, a: np.ndarray, idx: np.ndarray, ty: str, vals: list[np.ndarray]) -> None:
        es = np.dtype(NP[ty]).itemsize
        n = len(vals)
        if space == "global":
            b = self.gcheck(a, es * n, f"global store (.v{n}.{ty})" if n > 1 else f"global store .{ty}")
            for i in np.unique(b):
                self.stored_bufs.add(self.names[i])
            mem, g = self.gmem, a
        else:
            mem, g = self.smem, self.sh(a, idx, es * n, "store")
            self.track(g, idx, es * n, write=True)
        for k, v in enumerate(vals):
            v = np.ascontiguousarray(v.astype(NP[ty]) if v.dtype != NP[ty] else v)
            mem[(g + k * es)[:, None] + np.arange(es)] = v.view(np.uint8).reshape(-1, es)

    # ------------------------------------------------------------------ run
    def run(self, max_steps: int = 20_000_000) -> None:
        ins = self.k.instrs
        steps = 0
        while True:
            live = self.alive
            if not live.any():
                break
            p = int(self.pc[live].min())
            idx = np.nonzero(live & (self.pc == p))[0]
            steps += 1
            if steps > max_steps:
                raise EmuError("step limit hit (infinite loop?)")
            if p >= len(ins):
                raise EmuError("fell off the end of the kernel without ret")
            op, ops, pred, raw = ins[p]
            if pred is not None:
                pv = self.read(pred[1], idx, "pred")
                take = ~pv if pred[0] else pv
            else:
                take = np.ones(idx.size, dtype=bool)
            if op.startswith("bra"):
                tgt = self.k.labels[ops[0]]
                self.pc[idx] = np.where(take, tgt, p + 1)
                continue
            self.pc[idx] = p + 1
            idx = idx[take]
            if idx.size == 0:
                continue
            try:
                self.exec(op, ops, idx, p)
            except EmuError as e:
                raise EmuError(f"{e}  at `{raw}`") from None

    def exec(self, op: str, ops: list[str], idx: np.ndarray, p: int) -> None:
        parts = op.split(".")
        base = parts[0]
        ty = parts[-1]
        R = lambda o, t=ty: self.read(o, idx, t)  # noqa: E731
        W = lambda o, v: self.write(o, idx, v)  # noqa: E731

        if base == "ret":
            self.alive[idx] = False
            return
        if op == "ld.param.u64" or op.startswith("ld.param"):
            name = ops[1].strip("[]")
            v = self.param_vals[name]
            W(ops[0], np.full(idx.size, v, dtype=np.int64))
            return
        if op.startswith("cvta"):
            W(ops[0], R(ops[1], "s64"))
            return
        if base == "mov" and (ops[0].startswith("{") or ops[1].startswith("{")):
            if ops[0].startswith("{"):
                lo, hi = (x.strip() for x in ops[0].strip("{}").split(","))
                v = self.read(ops[1], idx, "u64")
                W(lo, (v & np.uint64(0xFFFFFFFF)).astype(np.uint32))
                W(hi, (v >> np.uint64(32)).astype(np.uint32))
            else:
                lo, hi = (x.strip() for x in ops[1].strip("{}").split(","))
                v = self.read(lo, idx, "u32").astype(np.uint64) | (self.read(hi, idx, "u32").astype(np.uint64) << np.uint64(32))
                W(ops[0], v.view(np.int64) if reg_class(ops[0]) is np.int64 else v.view(np.float64))
            return
        if base == "mov":
            src = ops[1]
            if src.startswith("%") and not src.startswith(("%tid", "%ctaid", "%ntid", "%nctaid")):
                W(ops[0], self.read(src, idx, ty if ty not in ("b32", "b64", "b16") else
                                    {"b32": "u32", "b64": "u64", "b16": "u16"}[ty]))
            else:
                W(ops[0], R(src))
            return
        if base in ("add", "sub", "mul", "div", "rem", "min", "max", "and", "or", "xor", "shl", "shr", "mad", "fma"):
            return self.arith(base, parts, ops, idx, R, W)
        if base in ("neg", "abs", "sqrt", "rsqrt", "rcp", "lg2", "ex2", "not"):
            a = R(ops[1])
            one = a.dtype.type(1)
            r = {"neg": lambda: -a, "abs": lambda: np.abs(a), "sqrt": lambda: np.sqrt(a),
                 "rsqrt": lambda: (one / np.sqrt(a)).astype(a.dtype),
                 "rcp": lambda: (one / a).astype(a.dtype), "lg2": lambda: np.log2(a),
                 "ex2": lambda: np.exp2(a), "not": lambda: ~a}[base]()
            if "ftz" in parts and r.dtype == np.float32:  # .ftz flushes subnormal results to zero
                r = np.where(np.abs(r) < np.float32(1.1754944e-38), np.float32(0), r).astype(np.float32)
            W(ops[0], r)
            return
        if base == "cvt":
            return self.cvt(parts, ops, idx, W)
        if base == "setp":
            cmp, t = parts[1], parts[2]
            a, b = R(ops[1], t), R(ops[2], t)
            if cmp == "nan":
                r = np.isnan(a) | np.isnan(b)
            elif cmp == "num":
                r = ~(np.isnan(a) | np.isnan(b))
            else:
                r = {"lt": a < b, "le": a <= b, "gt": a > b, "ge": a >= b, "eq": a == b, "ne": a != b}[cmp]
            if t.startswith("f") and cmp == "ne":
                r = r & ~np.isnan(a) & ~np.isnan(b)
            W(ops[0], r)
            return
        if base == "selp":
            t = ty if ty != "b32" else "u32"
            W(ops[0], np.where(R(ops[3], "pred"), R(ops[1], t), R(ops[2], t)))
            return
        if op == "activemask.b32":
            key = self.warp_key[idx]
            mask = np.zeros(self.nwarps, dtype=np.uint64)
            np.bitwise_or.at(mask, key, (np.uint64(1) << self.lane[idx].astype(np.uint64)))
            W(ops[0], mask[key].astype(np.uint32))
            return
        if op == "shfl.sync.idx.b32":
            a = self.regs[ops[1]]
            src_lane = R(ops[2], "u32").astype(np.int64) & 31
            member = R(ops[4], "u32").astype(np.uint64)
            src = idx - self.lane[idx] + src_lane
            executing = np.zeros(self.T, dtype=bool)
            executing[idx] = True
            srcc = np.clip(src, 0, self.T - 1)
            bad = (src >= self.T) | ((member >> src_lane.astype(np.uint64)) & 1 == 0) | ~executing[srcc] | (self.cta[srcc] != self.cta[idx])
            if bad.any():
                self.issue("shfl reads a lane that is not executing or not in the member mask")
            W(ops[0], a[srcc])
            return
        if op == "shfl.sync.bfly.b32":
            a = self.regs[ops[1]]
            off = int(ops[2], 0)
            member = R(ops[4], "u32").astype(np.uint64)
            src_lane = self.lane[idx] ^ off
            src = idx - self.lane[idx] + src_lane
            executing = np.zeros(self.T, dtype=bool)
            executing[idx] = True
            bad = (src_lane >= 32) | (src >= self.T) | ((member >> src_lane.astype(np.uint64)) & 1 == 0)
            srcc = np.clip(src, 0, self.T - 1)
            bad |= ~executing[srcc] | (self.cta[srcc] != self.cta[idx])
            if bad.any():
                self.issue("shfl reads a lane that is not executing or not in the member mask")
            W(ops[0], a[srcc])
            return
        if base == "bar":
            if parts[1] == "warp":
                return
            ctas = np.unique(self.cta[idx])
            here = np.zeros(self.T, dtype=bool)
            here[idx] = True
            for c in ctas:
                sel = (self.cta == c) & self.alive
                if not here[sel].all():
                    self.issue(f"bar.sync reached by only part of CTA {int(c)}'s live threads (hang/UB on a GPU)")
                    break
            self.epoch[ctas] += 1
            return
        if base in ("ld", "st"):
            return self.ldst(parts, ops, idx)
        if base == "red":
            a = self.addr(ops[0], idx)
            v = R(ops[1], parts[-1])
            es = np.dtype(NP[parts[-1]]).itemsize
            b = self.gcheck(a, es, "atomic")
            for i in np.unique(b):
                self.stored_bufs.add(self.names[i])
            view = self.gmem.view(NP[parts[-1]])
            np.add.at(view, a // es, v)
            return
        if base == "cp":
            return self.cpasync(parts, ops, idx)
        raise EmuError(f"unsupported instruction {op}")

    def arith(self, base, parts, ops, idx, R, W):
        ty = parts[-1]
        if base == "mul" and parts[1] == "wide":
            a, b = R(ops[1]).astype(np.int64), R(ops[2]).astype(np.int64)
            return W(ops[0], a * b)
        t = {"b32": "u32", "b64": "u64"}.get(ty, ty)
        if base in ("mad", "fma"):
            a, b, c = R(ops[1], t), R(ops[2], t), R(ops[3], t)
            if t == "f32":
                r = (a.astype(np.float64) * b.astype(np.float64) + c.astype(np.float64)).astype(np.float32)
            else:
                r = a * b + c
            return W(ops[0], r)
        a = R(ops[1], t)
        b = R(ops[2], "u32" if base in ("shl", "shr") else t)
        if base in ("shl", "shr"):
            bits = np.dtype(NP[t]).itemsize * 8
            big = b >= bits
            bb = np.minimum(b, bits - 1).astype(a.dtype)
            if base == "shl":
                r = np.where(big, 0, a << bb)
            else:
                r = a >> bb
                if not np.issubdtype(a.dtype, np.signedinteger):
                    r = np.where(big, 0, r)
            return W(ops[0], r.astype(a.dtype))
        if base in ("div", "rem") and np.issubdtype(a.dtype, np.integer):
            if (b == 0).any():
                self.issue("integer division by zero")
            bs = np.where(b == 0, 1, b)
            q = (np.abs(a.astype(np.int64)) // np.abs(bs.astype(np.int64))) * np.sign(a.astype(np.int64)) * np.sign(bs.astype(np.int64))
            r = q if base == "div" else a.astype(np.int64) - q * bs.astype(np.int64)
            return W(ops[0], r.astype(a.dtype))
        if base in ("min", "max") and t.startswith("f"):
            return W(ops[0], (np.fmin if base == "min" else np.fmax)(a, b))
        f = {"add": np.add, "sub": np.subtract, "mul": np.multiply, "div": np.divide, "min": np.minimum,
             "max": np.maximum, "and": np.bitwise_and, "or": np.bitwise_or, "xor": np.bitwise_xor}[base]
        if t == "pred":
            f = {"and": np.logical_and, "or": np.logical_or, "xor": np.logical_xor}[base]
        r = f(a, b)
        if t not in ("pred",):
            r = r.astype(NP[t])
        return W(ops[0], r)

    def cvt(self, parts, ops, idx, W):
        rnd = [x for x in parts[1:-2] if x in ("rn", "rz", "rm", "rp", "rni", "rzi", "rmi", "rpi")]
        dty, sty = parts[-2], parts[-1]
        rnd = rnd[0] if rnd else None
        src = self.read(ops[1], idx, sty)
        if sty == "f16":
            src = src.view(np.float16).astype(np.float32)
        elif sty == "bf16":
            src = (src.astype(np.uint32) << 16).view(np.float32)
        if dty == "f16":
            return W(ops[0], src.astype(np.float32).astype(np.float16).view(np.uint16))
        if dty == "bf16":
            u = src.astype(np.float32).view(np.uint32)
            u = u + np.uint32(0x7FFF) + ((u >> 16) & 1)
            return W(ops[0], (u >> 16).astype(np.uint16))
        if dty.startswith("f"):
            x = src.astype(np.float64) if sty.startswith("f") or sty in ("f16", "bf16") else src
            if rnd in ("rni", "rmi", "rpi", "rzi") and sty.startswith("f"):
                x = {"rni": np.rint, "rmi": np.floor, "rpi": np.ceil, "rzi": np.trunc}[rnd](src)
            return W(ops[0], np.asarray(x).astype(NP[dty]))
        if sty.startswith("f") or sty in ("f16", "bf16"):
            x = {"rni": np.rint, "rmi": np.floor, "rpi": np.ceil, "rzi": np.trunc, None: np.trunc}.get(rnd, np.trunc)(src.astype(np.float64))
            info = np.iinfo(NP[dty])
            x = np.where(np.isnan(x), 0, np.clip(x, info.min, info.max))
            return W(ops[0], x.astype(NP[dty]))
        return W(ops[0], src.astype(NP[dty]))

    def ldst(self, parts, ops, idx):
        space = parts[1]
        nc = "nc" in parts
        vec = next((int(x[1:]) for x in parts if re.fullmatch(r"v\d", x)), 1)
        ty = parts[-1]
        if parts[0] == "ld":
            a = self.addr(ops[1], idx)
            vals = self.mem_load(space, a, idx, ty, vec, nc)
            dsts = ops[0].strip("{}").split(",") if vec > 1 else [ops[0]]
            for d, v in zip(dsts, vals):
                self.write(d.strip(), idx, v)
        else:
            a = self.addr(ops[0], idx)
            srcs = ops[1].strip("{}").split(",") if vec > 1 else [ops[1]]
            vals = [self.read(s.strip(), idx, ty) for s in srcs]
            self.mem_store(space, a, idx, ty, vals)

    def cpasync(self, parts, ops, idx):
        if parts[2] == "commit_group":
            for e in self.pending:
                mine = np.isin(e["tid"], idx) & (e["grp"] < 0)
                e["grp"][mine] = self.committed[e["tid"][mine]] + 1
            self.committed[idx] += 1
            return
        if parts[2] == "wait_group":
            n = int(ops[0])
            keep = []
            for e in self.pending:
                done = np.isin(e["tid"], idx) & (e["grp"] >= 0) & (e["grp"] <= self.committed[e["tid"]] - n)
                if done.any():
                    g = self.cta[e["tid"][done]] * self.S + e["dst"][done]
                    self.track(g, e["tid"][done], e["data"].shape[1], write=True)
                    self.smem[g[:, None] + np.arange(e["data"].shape[1])] = e["data"][done]
                rest = ~done
                if rest.any():
                    keep.append({k: v[rest] for k, v in e.items()})
            self.pending = keep
            return
        dst = self.addr(ops[0], idx)
        src = self.addr(ops[1], idx)
        nbytes = int(ops[2])
        srcsize = self.read(ops[3], idx, "u32").astype(np.int64) if len(ops) > 3 else np.full(idx.size, nbytes)
        self.gcheck(src, nbytes, f"cp.async source ({nbytes}B)")
        if (dst < 0).any() or (dst + nbytes > self.S).any() or (dst % nbytes).any():
            raise EmuError("cp.async destination out of bounds or misaligned")
        # the copy may land any time before the wait
        self.track(self.cta[idx] * self.S + dst, idx, nbytes, write=True)
        data = self.gmem[src[:, None] + np.arange(nbytes)].copy()
        data[np.arange(nbytes)[None, :] >= srcsize[:, None]] = 0
        self.pending.append({"tid": idx.copy(), "dst": dst, "data": data, "grp": np.full(idx.size, -1, dtype=np.int64)})


def run_problem(pdir: Path, compiler: str | None = None, ptx_path: str | None = None, verbose: bool = True) -> dict:
    meta = json.loads((pdir / "meta.json").read_text())
    expected = dict(np.load(pdir / "expected.npz"))
    produced: dict[str, np.ndarray] = {}
    result = {"problem": pdir.name, "status": "PASS", "detail": "", "issues": []}
    for k in meta["kernels"]:
        if ptx_path:
            ptx = Path(ptx_path).read_text()
        else:
            ir = next(pdir.glob(f"{k['entry']}.ir.json"))
            r = subprocess.run([sys.executable, compiler, str(ir)], capture_output=True, text=True)
            if r.returncode:
                result.update(status="COMPILE-FAIL", detail=r.stderr.strip()[-300:])
                return result
            ptx = r.stdout
        kern = Kernel(ptx)
        bufs = {}
        for a in k["args"]:
            shape = a.get("shape") or [1]
            dt = DT[a["dtype"]]
            if a.get("intermediate"):
                bufs[a["name"]] = np.asarray(produced[a["name"]], dtype=dt).reshape(shape)
            elif "data" in a:
                bufs[a["name"]] = np.asarray(a["data"], dtype=dt).reshape(shape)
            elif "seed" in a:
                bufs[a["name"]] = np.random.default_rng(a["seed"]).standard_normal(shape).astype(dt)
            else:
                bufs[a["name"]] = np.zeros(shape, dtype=dt)
        grid, block = math.prod(k["grid"]), math.prod(k["block"])
        try:
            m = Machine(kern, grid, block, k["smem_bytes"], k["arg_order"], {n: bufs[n] for n in k["arg_order"]})
            m.run()
        except EmuError as e:
            result.update(status="RUNTIME-FAULT", detail=str(e))
            return result
        bad_nc = m.nc_bufs & m.stored_bufs
        if bad_nc:
            m.issue(f"ld.global.nc on buffer(s) the kernel writes: {sorted(bad_nc)}")
        result["issues"] += m.issues
        for a in k["args"]:
            if a.get("output"):
                i = m.names.index(a["name"])
                raw = m.gmem[m.bases[i]:m.ends[i]].copy().view(DT[a["dtype"]])
                produced[a["name"]] = raw
                if a["name"] in expected:
                    ref = expected[a["name"]]
                    # 64-bit outputs are compared in 64 bits
                    got = raw.astype(ref.dtype if ref.dtype.kind in "fi" and ref.dtype.itemsize == 8 else np.float32).reshape(ref.shape)
                    if not np.allclose(got, ref, rtol=meta["rtol"], atol=meta["atol"]):
                        err = np.abs(got.astype(np.float64) - ref)
                        worst = np.unravel_index(np.nanargmax(np.where(np.isnan(err), np.inf, err)), ref.shape)
                        nbad = int((~np.isclose(got, ref, rtol=meta["rtol"], atol=meta["atol"])).sum())
                        result.update(status="WRONG", detail=f"{a['name']}: {nbad}/{ref.size} elements off; worst at {tuple(int(x) for x in worst)} got {got[worst]!r} want {ref[worst]!r}")
                        return result
    if result["issues"] and result["status"] == "PASS":
        result["status"] = "PASS*"
        result["detail"] = "; ".join(result["issues"])
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("problems", nargs="+")
    ap.add_argument("--compiler", default="compile.py")
    ap.add_argument("--ptx")
    args = ap.parse_args()
    bad = 0
    for p in args.problems:
        r = run_problem(Path(p), args.compiler, args.ptx)
        print(f"{r['status']:<14} {r['problem']:<28} {r['detail']}", flush=True)
        bad += r["status"] not in ("PASS",)
    sys.exit(1 if bad else 0)
