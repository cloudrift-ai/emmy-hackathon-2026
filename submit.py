#!/usr/bin/env python3
"""Test a PTX file (or your whole compiler) against one problem.

    python submit.py r01_array_sum out.ptx
    python submit.py r01_array_sum --compiler ./compile.py

With --compiler, this runs `python compile.py <ir.json>` for each kernel in the
problem and submits whatever your compiler prints - the same way the hidden-set
evaluation will run it.

Exit code 0 = every kernel compiled and the final outputs matched expected.npz
within the problem's tolerance.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import requests

from kernelc.validation import result_array

PROBLEMS = Path(__file__).resolve().parent / "problems"   # works from any directory
RUNNER = "http://runner.riftstack.ai:48701"


def post(server: str, path: str, body: dict) -> dict:
    try:
        r = requests.post(f"{server}{path}", json=body, timeout=(10, 300))
    except (requests.ConnectionError, requests.Timeout) as exc:
        sys.exit(f"cannot reach the runner at {server} - check the URL ({type(exc).__name__})")
    r.raise_for_status()
    return r.json()


def not_ptx(ptx: str) -> str | None:
    """Why this cannot be PTX, or None. Catches the usual slips before ptxas does."""
    lines = [l.strip() for l in ptx.splitlines() if l.strip() and not l.strip().startswith(("//", "/*", "*"))]
    if not lines:
        return "it is empty"
    if not lines[0].startswith("."):
        return (f"it starts with {lines[0][:60]!r} instead of a directive like .version - "
                "stdout must be PTX only, so print debug output to stderr")
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("problem", help="problem id, e.g. r01_array_sum, or a problem folder")
    ap.add_argument("ptx", nargs="?", help="a PTX file (single-kernel problems)")
    ap.add_argument("--compiler", help="your compile.py; used instead of a fixed PTX file")
    ap.add_argument("--server", default=RUNNER, help=f"runner URL (default: {RUNNER})")
    ap.add_argument("--arch", default=None, help="override the target arch (default: the runner's GPU)")
    args = ap.parse_args()

    server = args.server.rstrip("/")
    pdir = Path(args.problem)
    if not (pdir / "meta.json").exists():
        pdir = PROBLEMS / args.problem
    if not (pdir / "meta.json").exists():
        sys.exit(f"unknown problem {args.problem!r}; choose from: "
                 + ", ".join(sorted(p.name for p in PROBLEMS.iterdir() if (p / "meta.json").exists())))
    meta = json.loads((pdir / "meta.json").read_text())
    expected = dict(np.load(pdir / "expected.npz"))

    if not args.compiler and not args.ptx:
        sys.exit("give a PTX file or --compiler")
    if not args.compiler and len(meta["kernels"]) > 1:
        sys.exit(f"{args.problem} has {len(meta['kernels'])} kernels - use --compiler")

    produced: dict[str, list] = {}   # buffer name -> values from earlier kernels
    ok = True

    for k in meta["kernels"]:
        # ---- get the PTX for this kernel ---------------------------------
        if args.compiler:
            ir = next(pdir.glob(f"{k['entry']}.ir.json"))
            r = subprocess.run([sys.executable, args.compiler, str(ir)],
                               capture_output=True, text=True)
            if r.returncode != 0:
                print(f"[{k['entry']}] your compiler failed (exit code {r.returncode}):\n"
                      f"{r.stderr or '(nothing on stderr)'}")
                return 1
            ptx = r.stdout
        else:
            ptx = Path(args.ptx).read_text()

        if why := not_ptx(ptx):
            print(f"[{k['entry']}] that is not PTX: {why}")
            return 1

        # ---- compile -----------------------------------------------------
        body = {"ptx": ptx, **({"arch": args.arch} if args.arch else {})}
        c = post(server, "/compile", body)
        if not c.get("ok"):
            print(f"[{k['entry']}] ptxas rejected it:\n{c.get('stderr', '')}")
            return 1
        regs = c.get("registers", {}).get(k["entry"])
        if regs is not None:
            print(f"[{k['entry']}] {regs} registers")
        if k["entry"] not in c.get("entry_points", []):
            print(f"[{k['entry']}] compiled, but entry points are {c.get('entry_points')} "
                  f"- the entry must be named {k['entry']!r}")
            return 1

        # ---- build args (chain intermediates from earlier kernels) -------
        run_args = []
        for a in k["args"]:
            a = dict(a)
            if a.pop("intermediate", False):
                if a["name"] not in produced:
                    print(f"[{k['entry']}] needs {a['name']} from an earlier kernel "
                          f"- kernels run in meta.json order")
                    return 1
                a["data"] = produced[a["name"]]
            run_args.append(a)

        # ask for enough values back to chain intermediates and check outputs
        need = max((math.prod(a["shape"]) for a in k["args"] if a.get("output")), default=1)
        for _ in range(60):  # "busy" means too many launches are queued; wait our turn
            r = post(server, "/run", {
                "cubin_id": c["cubin_id"], "entry": k["entry"],
                "grid": k["grid"], "block": k["block"], "smem_bytes": k["smem_bytes"],
                "zero_outputs": k["zero_outputs"], "args": run_args,
                "warmup": 0, "iters": 1, "max_return": max(4096, need),
            })
            if r.get("stage") != "busy":
                break
            time.sleep(2)
        if not r.get("ok"):
            print(f"[{k['entry']}] launch failed: {r.get('stderr', '')}")
            return 1

        # ---- collect outputs ---------------------------------------------
        for name, out in r["outputs"].items():
            if out.get("truncated"):
                print(f"[{k['entry']}] output {name} truncated by the server - "
                      f"ask the organizers to raise max_return")
                return 1
            produced[name] = out["values"]
            if name in expected:
                ref = expected[name]
                # Preserve the problem's dtype: narrowing f64 or i64 results here
                # can turn a correct kernel into a false failure.
                got = result_array(out["values"], ref)
                if np.allclose(got, ref, rtol=meta["rtol"], atol=meta["atol"]):
                    err = float(np.max(np.abs(got - ref))) if got.size else 0.0
                    print(f"[{k['entry']}] PASS  {name}: max abs err {err:.3e}  "
                          f"({r.get('median_us', '?')} us)")
                else:
                    err = float(np.max(np.abs(got - ref)))
                    print(f"[{k['entry']}] FAIL  {name}: max abs err {err:.3e} "
                          f"(tolerance rtol={meta['rtol']} atol={meta['atol']})\n"
                          f"    the parameters must be in meta.json's arg_order: {', '.join(k['arg_order'])}")
                    ok = False

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
