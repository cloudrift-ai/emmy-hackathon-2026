#!/usr/bin/env python3
"""IR regression cases: mutated open-problem IRs, compiled, assembled and run in tools/ptxemu.py.

    pip install nvidia-cuda-nvcc-cu12      # local ptxas
    python tests/kernelc_ir_cases.py
"""

from __future__ import annotations

import glob
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
from ptxemu import run_problem  # noqa: E402

OUT = Path(tempfile.mkdtemp(prefix="kernelc_cases_"))
F32 = "DataType(name='f32', np=dtype('float32'), nbytes=4)"
I64 = "DataType(name='i64', np=dtype('int64'), nbytes=8)"


def find_ptxas() -> str:
    if os.environ.get("PTXAS"):
        return os.environ["PTXAS"]
    hits = glob.glob(os.path.join(sys.prefix, "lib", "python3*", "*-packages", "nvidia", "cuda_nvcc", "bin", "ptxas"))
    hits += glob.glob("/usr/local/lib/python3*/dist-packages/nvidia/cuda_nvcc/bin/ptxas")
    return hits[0] if hits else (shutil.which("ptxas") or "ptxas")


PTXAS = find_ptxas()
results: list[tuple[str, bool, str]] = []


def mutate(case: str, src: str, subs, json_fn=None, meta_fn=None) -> Path:
    d = OUT / case
    shutil.copytree(REPO / src, d)
    ir = next(d.glob("*.ir.json"))
    text = ir.read_text()
    for old, new in subs:
        assert old in text, f"{case}: pattern not found: {old[:60]}"
        text = text.replace(old, new)
    if json_fn:
        j = json.loads(text)
        json_fn(j)
        text = json.dumps(j)
    ir.write_text(text)
    if meta_fn:
        m = json.loads((d / "meta.json").read_text())
        meta_fn(m)
        (d / "meta.json").write_text(json.dumps(m, indent=1))
    return d


def compile_and_assemble(d: Path) -> tuple[bool, str]:
    ir = next(d.glob("*.ir.json"))
    p = subprocess.run([sys.executable, str(REPO / "compile.py"), str(ir)], capture_output=True, text=True)
    if p.returncode:
        return False, "compile error: " + p.stderr.strip().splitlines()[-1]
    (d / "out.ptx").write_text(p.stdout)
    for arch in ("sm_89", "sm_120"):
        q = subprocess.run([PTXAS, f"-arch={arch}", str(d / "out.ptx"), "-o", os.devnull], capture_output=True, text=True)
        if q.returncode:
            return False, f"ptxas {arch}: " + q.stderr.strip().splitlines()[0]
    return True, "compiles and assembles"


def check_run(case: str, d: Path, expected: dict) -> None:
    np.savez(d / "expected.npz", **expected)
    ok, why = compile_and_assemble(d)
    if not ok:
        results.append((case, False, why))
        return
    r = run_problem(d, str(REPO / "compile.py"))
    results.append((case, r["status"] == "PASS", f"{r['status']} {r['detail']}".strip()))


def seeded(seed: int, shape, dtype=np.float32):
    return np.random.default_rng(seed).standard_normal(shape).astype(dtype)


# R1. int64 sum reduction
def r1_graph(j):
    j["nodes"]["x"]["outputs"][0]["dtype"] = "i64"
    j["nodes"]["sum_1"]["outputs"][0]["dtype"] = "i64"


def r1_meta(m):
    for a in m["kernels"][0]["args"]:
        a["dtype"] = "i64"


d = mutate("R1_int64_sum", "problems/r01_array_sum", [(F32, I64), ("dtype='float'", "dtype='long long'")], r1_graph, r1_meta)
check_run("R1 int64 sum reduction", d, {"sum_1": np.array(seeded(0, 1024, np.int64).sum(), dtype=np.int64)})

# R2. Init(acc0 = 5.0) then an Accum in a Cond inside a seeded loop: 128 threads start at 5.0
ACC = "Accum(name='acc0', value='in0', op=ElementwiseImpl('add'), dtype=" + F32 + ", axes=('a0',), base=None)"
COND = "Cond(cond=BinaryExpr(op='>=', left=Var(name='a0'), right=Literal(value=0, dtype='int')), body=(" + ACC + ",), else_body=())"
d = mutate("R2_cond_nested_accum", "problems/r01_array_sum",
           [("body=(StridedLoop(", "body=(Init(name='acc0', identity=5.0, dtype=" + F32 + "), StridedLoop("), (ACC, COND)])
check_run("R2 Accum in a Cond keeps its Init value", d,
          {"sum_1": np.array(seeded(0, 1024).astype(np.float64).sum() + 640, dtype=np.float32)})

# R3. an Accum in a Cond must be visible after it
A = "Assign(name='v0__u0', op=ElementwiseImpl('add'), args=('in0__u0', 'in1__u0'), dtype=" + F32 + ")"
ACC3 = "Accum(name='v0__u0', value='in0__u0', op=ElementwiseImpl('add'), dtype=" + F32 + ", axes=(), base=None)"
COND3 = "Cond(cond=BinaryExpr(op='>=', left=Var(name='a0'), right=Literal(value=0, dtype='int')), body=(" + ACC3 + ",), else_body=())"
d = mutate("R3_accum_in_cond", "problems/p00_scalar_add", [(A, A + ", " + COND3)])
x0, x1 = seeded(0, 128), seeded(1, 128)
want = x0 + x1
want[0::2] = (x0[0::2] + x1[0::2]) + x0[0::2]
check_run("R3 Accum in a Cond is visible after it", d, {"add": want})

# R4. named barrier without a count is a compile error
TH = "barrier_id=0, barrier_count=None, inner=None"
d = mutate("R4_named_barrier_no_count", "problems/r01_array_sum", [(TH, "barrier_id=1, barrier_count=None, inner=None")])
ok, why = compile_and_assemble(d)
results.append(("R4 named barrier without count is a compile error", (not ok) and "requires barrier_count" in why, why))

# R5. index rank mismatch is a compile error
d = mutate("R5_index_rank_mismatch", "problems/r04_row_sum",
           [("index=(Var(name='a0'), Var(name='a1'))", "index=(Var(name='a0'), Var(name='a1'), Literal(value=0, dtype='int'))")])
ok, why = compile_and_assemble(d)
results.append(("R5 index rank mismatch is a compile error", (not ok) and "indexed with 3 coordinates" in why, why))

for case, ok, why in results:
    print(f"{'ok  ' if ok else 'FAIL'} {case}\n       {why}")
bad = sum(not ok for _, ok, _ in results)
print(f"\n{len(results) - bad}/{len(results)} pass. Artifacts in {OUT}")
sys.exit(1 if bad else 0)
