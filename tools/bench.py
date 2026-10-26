#!/usr/bin/env python3
"""Time compiler variants on the runner's GPU.

    python tools/bench.py a01_attention xa02_attn_256 \\
        --variant shared_off:KERNELC_LANE_SHARE=0 --variant shared_on

A variant is NAME[@COMPILER][:ENV=VALUE,...]; COMPILER defaults to compile.py. Every variant
is compiled, checked against expected.npz, and launched with --warmup/--iters timed reps.
Rounds interleave the variants so clock drift does not favour one. Prints a table and
writes --csv.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def post(server: str, path: str, body: dict) -> dict:
    req = urllib.request.Request(server + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    for _ in range(60):
        with urllib.request.urlopen(req, timeout=120) as r:
            out = json.loads(r.read())
        if out.get("stage") != "busy":
            return out
        time.sleep(2)
    raise RuntimeError("runner stayed busy")


def find_problem(name: str) -> Path:
    for p in (Path(name), ROOT / "problems" / name, ROOT / "tests" / "problems" / name):
        if (p / "meta.json").exists():
            return p
    for p in (ROOT / "tests").glob(f"problems_*/{name}"):
        if (p / "meta.json").exists():
            return p
    sys.exit(f"unknown problem {name!r}")


def parse_variant(spec: str) -> tuple[str, str, dict]:
    name, _, env = spec.partition(":")
    name, _, compiler = name.partition("@")
    envs = dict(kv.split("=", 1) for kv in env.split(",") if kv)
    return name, compiler or str(ROOT / "compile.py"), envs


def compile_ptx(compiler: str, env: dict, ir: Path) -> str:
    r = subprocess.run([sys.executable, compiler, str(ir)], capture_output=True, text=True, env={**os.environ, **env})
    if r.returncode:
        raise RuntimeError(f"compiler failed: {r.stderr.strip()}")
    return r.stdout


def run_problem(server: str, pdir: Path, kernels: list, warmup: int, iters: int, check: bool) -> tuple[float, bool]:
    """Launch every kernel of the problem in order; returns (summed median_us, outputs correct)."""
    meta = json.loads((pdir / "meta.json").read_text())
    expected = dict(np.load(pdir / "expected.npz")) if check else {}
    produced, total, ok = {}, 0.0, True
    for k, cubin in zip(meta["kernels"], kernels):
        args = []
        for a in k["args"]:
            a = dict(a)
            if a.pop("intermediate", False):
                a["data"] = produced[a["name"]]
            args.append(a)
        need = max((math.prod(a["shape"]) for a in k["args"] if a.get("output")), default=1)
        r = post(server, "/run", {"cubin_id": cubin, "entry": k["entry"], "grid": k["grid"], "block": k["block"],
                                  "smem_bytes": k["smem_bytes"], "zero_outputs": k["zero_outputs"], "args": args,
                                  "warmup": warmup, "iters": iters, "max_return": max(4096, need)})
        if not r.get("ok"):
            raise RuntimeError(f"launch failed: {r.get('stderr', r)}")
        total += r["median_us"]
        for name, out in r["outputs"].items():
            produced[name] = out["values"]
            if name in expected:
                got = np.array(out["values"], dtype=np.float32).reshape(expected[name].shape)
                ok &= bool(np.allclose(got, expected[name], rtol=meta["rtol"], atol=meta["atol"]))
    return total, ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("problems", nargs="+")
    ap.add_argument("--variant", action="append", required=True)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--server", default="http://runner.riftstack.ai:48701")
    ap.add_argument("--csv", default="bench.csv")
    args = ap.parse_args()
    server = args.server.rstrip("/")
    variants = [parse_variant(v) for v in args.variant]

    rows = []
    for name in args.problems:
        pdir = find_problem(name)
        meta = json.loads((pdir / "meta.json").read_text())
        built = {}
        for vname, compiler, env in variants:
            cubins, regs = [], []
            for k in meta["kernels"]:
                ptx = compile_ptx(compiler, env, next(pdir.glob(f"{k['entry']}.ir.json")))
                c = post(server, "/compile", {"ptx": ptx})
                if not c.get("ok"):
                    raise RuntimeError(f"{pdir.name} [{vname}] ptxas: {c.get('stderr')}")
                cubins.append(c["cubin_id"])
                regs.append(c.get("registers", {}).get(k["entry"]))
            _, ok = run_problem(server, pdir, cubins, 0, 1, check=True)
            built[vname] = (cubins, regs, ok)
        times = {v[0]: [] for v in variants}
        for rnd in range(args.rounds):
            order = variants if rnd % 2 == 0 else variants[::-1]
            for vname, _, _ in order:
                t, _ = run_problem(server, pdir, built[vname][0], args.warmup, args.iters, check=False)
                times[vname].append(t)
        base = statistics.median(times[variants[0][0]])
        for vname, _, _ in variants:
            med = statistics.median(times[vname])
            rows.append({"problem": pdir.name, "variant": vname, "correct": built[vname][2],
                         "registers": "/".join(str(r) for r in built[vname][1]),
                         "median_us": round(med, 3), "min_us": round(min(times[vname]), 3),
                         "speedup": round(base / med, 2), "rounds_us": " ".join(f"{t:.2f}" for t in times[vname])})
            print(f"{pdir.name:<22} {vname:<14} correct={str(built[vname][2]):<5} regs={rows[-1]['registers']:<5} "
                  f"median {med:10.2f} us  min {min(times[vname]):10.2f} us  speedup {base / med:6.2f}x", flush=True)
    with open(args.csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {args.csv}")
    return 0 if all(r["correct"] for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
