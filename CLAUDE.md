# CLAUDE.md

Emmy Hackathon 2026: write a compiler from Emmy **Kernel IR** to **PTX**. Deadline: **23:59 AoE, Tue Oct 27** (submit at riftstack.ai/hackathon/submit; public GitHub repo + 3-5 min video in the author's own voice).

Sources: README.md, TASK.md, MVP.md, docs/API.md, problems/PROVENANCE.md, examples/NOTE.md. Read TASK.md for the full IR reference.

## Submission contract

- `compile.py` at the repo root (not yet written). Run as `python compile.py path/to/<kernel>.ir.json > out.ptx`. IR path is `argv[1]`, PTX goes to stdout only (debug output to stderr; stdout must start with a directive such as `.version`).
- Evaluated mechanically on a **hidden set** of IRs in the same 5 categories (different shapes/ops). Never hardcode the open problems. Keep compile time to seconds per kernel.
- **The IR -> PTX translation must be our own.** Do not lower through emmy's renderer (`render_kernelop`), `emmy compile`, `nvcc`, or any other compiler. Using `Graph.from_dict` to load the IR is fine. Judges read the code.
- AI assistance is allowed, but the author must be able to explain the compiler in the video. Explain design choices as we go, and keep the code readable (code review is the first tiebreak, speed the last).

## Priorities

1. MVP = all of categories 1 and 2 pass: `p00_scalar_add p01_vector_add p02_scale_shift p03_relu_mul` (pointwise) and `r01_array_sum r02_big_sum r03_mean r04_row_sum` (reductions).
2. Then matmul (`m01_matmul_512`, `m02_matmul_rect`: Loop, Init, CpAsync*, atomics + `zero_outputs`), fused (`f01_rmsnorm`, `f02_row_softmax`, `f03_l2norm`), attention (`a01_attention`, `a02_attention_128`).
3. A clean compiler for fewer categories beats a messy one covering more. No optimization is required.

## What the PTX must satisfy

- Entry name = the kernel's `name` in the IR (e.g. `k_sum_1_reduce`).
- Parameters follow `arg_order` from the problem's **`meta.json`**, not the IR node's `inputs`. Buffers are `.u64` pointers. `ConstantOp` values arrive as 1-element buffer args.
- Must be correct under the grid/block/smem in `meta.json` (e.g. 1024 elements over 128 threads means 8 per thread). Static `.shared` needs no launch `smem_bytes`.
- Match `expected.npz` within `rtol`/`atol` from `meta.json`. Exact equality is not required.
- Start with `.version 8.0`, `.target sm_89`, `.address_size 64`. `sm_89` needs version >= 7.8. The runner is an RTX 5090 (sm_120) and runs sm_89 PTX unchanged.
- **ASCII only** in PTX, comments included.
- Guard threads: the launch can have more threads than Tile cells (p00: 64 cells, 256 threads). The `tid < N` guard is not in the IR; derive it from the Tile's `N`.
- Vector ops: `a,b,c,d = load x[i]` is `ld.global.v4.f32`; a grouped Write is `st.global.v4.f32` (v2 for pairs); atomic Write is `red.global.add.f32`.
- Use fresh virtual registers and declare counts at the top; ptxas does real allocation. No liveness analysis.

## Compiler shape (per TASK.md)

Load (`Graph.from_dict`, find the KernelOp, entry name from the IR, params from `meta.json`) -> prologue (version/target, signature, `ld.param`, `cvta.to.global`, tid, bounds guard) -> walk body statements keeping an IR-name -> register map (index expressions are small `Var`/`Literal`/`BinaryExpr` trees, compiled recursively) -> emit per statement kind.

Needed statement kinds: `Tile, StridedLoop, Loop, Load, Assign, Let, Accum, Init, Write, Cond, Smem, Sync, WarpShuffle, TreeHalve, CpAsyncCopy/Commit/Wait`. Expressions: `Var, Literal, BinaryExpr, TernaryExpr`. Operations marked `*` in TASK.md (Pre, Carry, Select, ZeroPrologue, TMA, mbarrier, mma/wgmma fragments, Builtin, FuncCallExpr, CastExpr, FlatIndex, ...) appear in no open or hidden problem, so skip them. `KernelOp.serial` is empty in every problem.

## Repo layout

- `problems/<id>/`: `*.ir.json` (compiler input), `*.ir.txt` (readable), `*.ref.cu` (Emmy's CUDA reference, for understanding only), `meta.json`, `expected.npz`. Each has one or more kernels; multi-kernel problems chain intermediate buffers in `meta.json` order.
- `examples/k_sum_1_reduce.ptx`: hand-written PTX for r01 with comments mapping to IR statements. Best reference for output style.
- `submit.py`: test harness. `python submit.py <problem> file.ptx` (single-kernel) or `python submit.py <problem> --compiler compile.py` (runs the compiler per kernel, like the hidden eval). Exit 0 = all pass. Needs network access to the runner.
- `docs/API.md`: runner API (`/health`, `/compile`, `/run`). Runner: `http://runner.riftstack.ai:48701`. ptxas errors come back verbatim; failed compile is HTTP 200 with `ok:false`.
- `tools/`: how problems were generated. `env/`: local virtualenv (Python 3.14, emmy installed). It is not in `.gitignore`, so do not commit it.
- `hidden/`, `hidden_full/`, `expressions_hidden.py` are gitignored organizer-only material.

## Environment

- Python >= 3.12. `pip install emmy-ml==0.3.11 numpy requests` (no torch or GPU needed). Use `env/bin/python` or activate `env/`.
- Never parse the IR body string by hand; use `Graph.from_dict` typed objects, e.g. `g.nodes["sum_1"].op.body`. Graph-level json (`inputs`, `outputs`, `shape`/`dtype`) is plain.
- To generate extra IRs (needs torch): `EMMY_DUMP_DIR=out emmy compile -c "<torch expr>" --ir kernel`. `x.T` does not trace; pre-transpose.
- Problems were generated for sm_89 and checked on an RTX 5090; not run on a 4090.

## Workflow

- **Plan first, no code until approved.** When the user says "let's work on X" (or asks to implement, brainstorm, or change anything), do NOT edit code. Instead, split the idea into small numbered parts in the message, each with a short explanation of what it does and why. Reading files to prepare the plan is fine; editing or writing code is not. Wait for the user to say which parts they understand and approve (e.g. "I understand parts 1 and 2, go implement"), then implement only those parts. The author must be able to explain the compiler in the video, so this is a hard rule.
- After each change, run all open problems through `python submit.py <id> --compiler compile.py` and report failures with output.
- Debug with ptxas stderr from the runner; compare against `examples/*.ptx` and `*.ref.cu`.
