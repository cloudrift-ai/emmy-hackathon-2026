#!/usr/bin/env python3
"""Kernel IR -> PTX compiler.

    python compile.py path/to/kernel.ir.json > out.ptx

The IR is loaded through Emmy's classes; the lowering to PTX is ``kernelc``'s
own (see kernelc/codegen.py). Diagnostics go to stderr, PTX to stdout.
"""

import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from kernelc.codegen import compile_kernel  # noqa: E402
from kernelc.loader import load  # noqa: E402


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: python compile.py <kernel.ir.json> > out.ptx", file=sys.stderr)
        return 2
    # KERNELC_LOG=1 logs the compiler's decisions to stderr
    logging.basicConfig(stream=sys.stderr, format="kernelc: %(levelname)s: %(message)s",
                        level=logging.INFO if os.environ.get("KERNELC_LOG") == "1" else logging.WARNING)
    try:
        ptx = compile_kernel(load(argv[1]))
    except Exception as exc:  # report, never print partial PTX
        print(f"kernelc: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    sys.stdout.write(ptx)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
