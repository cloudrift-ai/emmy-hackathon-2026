# kernelc: Kernel IR to PTX

```bash
python compile.py path/to/kernel.ir.json > out.ptx
./check_all.sh                       # the 15 open problems, through submit.py and the runner
./check_all.sh --dir tests/problems  # 39 extra problems: new shapes and ops in the same categories
python tests/op_accuracy.py          # ulp error of every transcendental op, measured on the GPU
```

Emmy is used only to load the IR (`Graph.from_dict`). The IR-to-PTX lowering is in `kernelc/`.

## Layout

| file | role |
|---|---|
| `compile.py` | CLI: one IR path in, PTX on stdout, errors on stderr with a non-zero exit |
| `kernelc/loader.py` | reads the KernelOp, buffer shapes and dtypes from the graph, constant values, and the parameter order from `meta.json` (`arg_order`) |
| `kernelc/ptx.py` | `Val` (a register or an immediate with its compile-time value), the `Emitter` (virtual registers, labels, `.reg` declarations) |
| `kernelc/arith.py` | typed scalar arithmetic with constant folding, algebraic identities and type conversions; the only place scalar opcodes are spelled |
| `kernelc/mathlib.py` | every elementwise op Emmy can emit (`add` through `gelu_tanh`), lowered onto `Arith` |
| `kernelc/codegen.py` | the statement walker: one `_lower_<Stmt>` method per Kernel IR statement |
| `kernelc/peephole.py` | after emission, merges adjacent aligned scalar loads into `.v2`/`.v4` loads |

## How a kernel is lowered

1. **Plan.** Every `Smem` slab is packed in body order. Up to 48 KiB the slabs become static
   `.shared` arrays. Above that they alias one `.extern .shared` pool, whose size `meta.json`
   passes at launch.
2. **Prologue.** Parameters follow `meta.json`'s `arg_order`. Each buffer goes through
   `ld.param` and `cvta.to.global`, and buffers the kernel never writes get `ld.global.nc`
   loads. `lane` and `warp` are derived from `%tid.x`.
3. **Tile.** `gid = ctaid * C + tid`. The `gid < N` guard is emitted only when `N % C != 0`.
   Because `0 <= tid < C`, an axis that lives inside one CTA decodes from `tid` alone, and one
   that is constant across the CTA decodes from `ctaid` alone. Grouped rasterization follows
   Emmy's threadblock swizzle.
4. **Walk.** `env` maps IR names to `Val`s. A nested body works on a copy of `env`, so its SSA
   names disappear when the body ends. Accumulators live in registers allocated before their
   loop and seeded with `op.identity`, so they outlive the loop. Like Emmy, a loop seeds only
   the `Accum`s in its immediate body. Index expressions are compiled recursively, folded, and
   memoized per scope.
4b. **Addressing.** An index becomes a base register plus an immediate offset, and the
   peephole merges neighbouring aligned loads into `.v2`/`.v4`.
4c. **Lane sharing.** In a key loop, statements that depend on the key but not on the
   innermost (lane) axes, such as attention's q.k dot products, are computed once per group of
   G adjacent lanes: lane i computes key chunk*G + i, and `shfl.sync.idx` hands the results to
   the group (`kernelc/laneshare.py`). `KERNELC_LANE_SHARE=0` turns it off.
5. **Loops.** If the trip count is the same for every thread (this is also proved when `start`
   is a Tile axis with a known range), the loop is unrolled fully or 2/4/8 copies per trip, with
   the induction variable as a constant or `i + k*step`. Otherwise it becomes a bottom-tested
   loop.

## Statement mapping

| IR | PTX |
|---|---|
| `Load` / `Write` | `ld/st.global` or `.shared`; 2/4-wide groups become `.v2/.v4`; constant offsets go into the address immediate |
| `Write(atomic)` | `red.global.add.f32` |
| `Assign`, `Let`, `Select`, `Init` | `mathlib` / `Arith`; `.rn` on every float op, so `ptxas` never contracts `mul`+`add` and results match numpy bit for bit (except under fast math, below) |
| `Accum` | updates the accumulator register in place; `acc += a*b` becomes one `fma.rn.f32` |
| `Cond` | `setp` + `@!p bra`; a condition that folds to a constant compiles only the branch taken |
| `WarpShuffle` | `activemask` + a `shfl.sync.bfly.b32` ladder, running `combine_states` after each step |
| `TreeHalve` | a halving loop over the shared slabs with a `bar.sync` per level, then a broadcast of slot 0 |
| `Sync` | `bar.sync 0`, `bar.sync id, count` or `bar.warp.sync` |
| `CpAsyncCopy` / `Commit` / `Wait` | `cp.async.cg` (16 B) or `.ca` (4/8 B), `cp.async.commit_group`, `cp.async.wait_group N` |
| `ZeroPrologue` | CTA 0 zeroes the delegated accumulator with a strided loop |

## Accuracy

The hardware `ex2/lg2/sin.approx` units are accurate only to about 2^-22. Each transcendental
op is therefore built from range reduction plus a polynomial:

- `exp`: Cody-Waite with ln2 split in two parts, then a degree-7 polynomial.
- `log`: musl's `logf`.
- `erf`: Juffa's `erff`.
- `tanh`: via `expm1`.
- `sin` and `cos`: 3-part pi/2 reduction.

`tests/op_accuracy.py` measures these on the GPU at 0.5 to 2 ulp against a float64 reference.
`remainder`, `fmod` and `floor_divide` are exact.

**Fast math.** When `meta.json` gives `rtol >= 1e-4`, `exp`, reciprocal, `sqrt` and `rsqrt` use
the hardware `.approx.ftz` units, division is `div.full`, and f32 `add/sub/mul` may fuse into
`fma`. `KERNELC_FAST_MATH=0/1` overrides.
