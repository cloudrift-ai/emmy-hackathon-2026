# Contributor guide

## Purpose and boundaries

This repository compiles Emmy Kernel IR directly to PTX. The required public
interface is unchanged: `python compile.py path/to/kernel.ir.json > out.ptx`.
PTX must be written only to stdout; diagnostics belong on stderr.

Do not replace the IR-to-PTX lowering with CUDA source, `nvcc`, or a reference
kernel. The compiler must generalize to hidden Kernel IR inputs.

## Prerequisites

- Python 3.12 or newer.
- Install runtime dependencies with `python -m pip install -r requirements.txt`.
- A local `ptxas` is optional but needed for assembly checks. Set `PTXAS` when
  it is not on `PATH`.
- GPU execution is optional locally; `submit.py` targets the hosted runner.

## Code map

- `compile.py`: stable CLI entry point.
- `kernelc/loader.py`: Emmy graph and launch-metadata loading.
- `kernelc/codegen.py`: Kernel IR statement and expression lowering.
- `kernelc/arith.py` and `kernelc/mathlib.py`: typed scalar lowering.
- `kernelc/ptx.py`: PTX registers, immediates, labels, and emission.
- `kernelc/peephole.py` and `kernelc/laneshare.py`: optional optimizations.
- `tools/ptxemu.py`: CPU safety/correctness emulator for emitted PTX.

## Implementation rules

- Keep changes scoped to the requested behavior. Do not modify problem IR,
  expected arrays, or reference CUDA merely to make a test pass.
- Preserve IR scoping: nested bodies receive copied environments; only carried
  accumulators deliberately outlive their loops or conditions.
- Maintain type and rounding behavior. In particular, do not silently narrow
  f64 or integer values in validation code.
- Treat global-buffer alignment and shared-memory barriers as correctness
  constraints when changing vectorization or address generation.
- New PTX syntax must be understood by `tools/ptxemu.py`, or that limitation
  must be explicit and covered by an assembly/GPU test.

## Validation

Run the narrowest relevant checks first:

```bash
python -m compileall -q compile.py submit.py kernelc tools tests
python -m unittest discover -s tests -p 'test_*.py'
python tests/kernelc_ir_cases.py       # requires Emmy and ptxas
./check_all.sh                         # hosted ptxas/GPU regression suite
```

For performance changes, first establish correctness, then use
`tools/bench.py` with at least a baseline and candidate variant. Record kernel
correctness alongside timing; a speedup that changes results is not valid.

`KERNELC_COMMENTS=0` suppresses source-oriented PTX comments for a smaller
production artifact. It must not change generated instructions.
