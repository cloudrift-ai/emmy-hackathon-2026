# Emmy Hackathon — write a PTX compiler for Kernel IR

Emmy (github.com/cloudrift-ai/emmy) compiles PyTorch to CUDA C++ and builds it with `nvcc`.
**Your task: write a compiler that takes Emmy's Kernel IR and emits PTX directly**, skipping the
CUDA C++ step. `ptxas` turns PTX into the same GPU binary ~19× faster than the full `nvcc`
pipeline — Emmy's autotuner compiles thousands of kernel variants, so that middle step matters.

You do **not** need a GPU. A hosted runner at `http://runner.riftstack.ai:48701` (the default in `submit.py`) compiles
your PTX with `ptxas`, executes it on real hardware, and tells you whether the output is correct. See [docs/API.md](docs/API.md).

## Requirements

- **Python 3.12 or newer.** The Kernel IR only deserializes through Emmy's own classes:

  ```bash
  python -m pip install -r requirements.txt  # no torch, no CUDA, no GPU needed
  ```

  This alone reads every problem IR. Torch is **not** needed — it is only required to
  *generate* new IRs from PyTorch expressions, which is optional.

- Your submission is a **program**, not PTX files. It must run as:

  ```bash
  python compile.py path/to/kernel.ir.json > out.ptx
  ```

  This contract matters: submissions are evaluated against a **hidden set** of Kernel IRs in the
  same categories as the open problems. Hardcoding the open set scores nothing.

## Five-minute start

```bash
git clone https://github.com/cloudrift-ai/emmy-hackathon-2026 && cd emmy-hackathon-2026
python -m pip install -r requirements.txt    # no torch needed
python submit.py r01_array_sum examples/k_sum_1_reduce.ptx    # see a pass
python submit.py p00_scalar_add examples/k_sum_1_reduce.ptx   # wrong entry: a clean failure
python submit.py p00_scalar_add --compiler compile.py         # your compiler, once you have one
```

Then open `problems/p00_scalar_add/` and look at three files:

| file | what it is |
|---|---|
| `*.ir.json` | the Kernel IR — **the input to your compiler** |
| `*.ir.txt` | the same IR, pretty-printed for reading |
| `*.ref.cu` | Emmy's generated CUDA C++ — the reference answer, in a different target language |

Load an IR in Python:

```python
import json
from emmy.compiler.graph import Graph
g = Graph.from_dict(json.load(open("problems/r01_array_sum/k_sum_1_reduce.ir.json")))
body = g.nodes["sum_1"].op.body     # a tree of typed statements: Tile, StridedLoop, Load, ...
```

Your compiler walks that tree and prints PTX. Start with the common node kinds in the short table —
see [TASK.md](TASK.md) for the categories and the node-by-node translation table.

## The problems

15 open problems in 5 categories, easy → hard: **pointwise → reductions → matmul → fused →
attention-like**. Each is a folder under `problems/` with the IR, the reference, a launch
descriptor (`meta.json` — you never guess grid/block), and expected outputs (`expected.npz`).

**MVP: your compiler passes categories 1 and 2.** See [MVP.md](MVP.md).

Problems are scheduled for `sm_89`, and the runner is an **RTX 5090 (`sm_120`)**, where PTX written for `sm_89`
runs unchanged; [problems/PROVENANCE.md](problems/PROVENANCE.md)
says how they were generated and checked. The hidden evaluation set contains more problems of the *same kinds* —
different shapes and ops, no surprise concepts.

## Submitting

Submit at [riftstack.ai/hackathon/submit](https://riftstack.ai/hackathon/submit) before
**23:59 AoE on Tuesday, October 27**. A submission is:

- a **public GitHub repository** with `compile.py` at its root, runnable as
  `python compile.py path/to/kernel.ir.json > out.ptx`;
- a **video of 3 to 5 minutes** in which you walk through your compiler in your own voice
  (YouTube, Vimeo or Google Drive, shared so that anyone with the link can watch it).

Teams of up to three submit once and list every member. You can submit again until the
deadline; we review your latest entry. AI coding assistants are allowed, but your video has to
show that you understand how your compiler works — an entry where the understanding seems to
have been handed to an AI is not eligible for a prize.

## How compilers are evaluated

We run your `compile.py` on every kernel of the **hidden set** — the same five categories as the
open problems, with different shapes and ops — then assemble the PTX with `ptxas`, run it on the
runner's GPU under each problem's `meta.json` launch, and compare the outputs with the expected
ones within the problem's tolerance. This is the pipeline `submit.py --compiler` runs, so passing
the open set that way is the best preparation. Keep compiling to seconds per kernel: a time
limit may apply.

1. **Correctness on the hidden set, by category.** Passing categories 1 and 2 is the bar
   ([MVP.md](MVP.md)); each further category is what sets submissions apart.
2. **Code review** of your compiler is the first tiebreak. The IR → PTX lowering has to be your
   own — see the rule in [TASK.md](TASK.md).
3. **Speed** of the generated kernels is the last tiebreak.

The video decides prize eligibility, as described above.

## Repo map

```
problems/     the open set (15 problems)
examples/     hand-written PTX with comments mapping each block back to the IR
submit.py     one-command test: compile + run + check against expected outputs
docs/API.md   the runner API, if you want to call it directly
tools/        how the problems were generated (emmy compile -c ... --ir cuda)
TASK.md       the task, the categories, the CLI contract
MVP.md        the minimum bar
```
