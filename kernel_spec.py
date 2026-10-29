"""Load what we compile: the KernelOp from the IR plus its launch info from meta.json."""

import sys
import json
from dataclasses import dataclass
from pathlib import Path
from emmy.compiler.graph import Graph

@dataclass
class KernelSpec:
    # launch info + GPU kernel IR
    name: str          # kernel name
    body: tuple        # given emmy IR
    params: list       # arg_order, one .u64 param each
    grid: tuple
    block: tuple
    smem_bytes: int
    shapes: dict       # buffer name -> shape list, for flattening multi-dim indices

def load_kernel(path):
    with open(path) as f:
        graph = Graph.from_dict(json.load(f))
    # we compile KernelOp only because it is what compiles to PTX
    for node in graph.nodes.values():
        if type(node.op).__name__ == "KernelOp":
            return node.op
    sys.exit(f"{path}: no KernelOp found")

def load_spec(ir_path):
    #Load the KernelOp (GPU kernel IR)
    gpu_kernel_ir = load_kernel(ir_path)
    # load the Meta (Kernel Launch Code)
    meta = json.loads((Path(ir_path).parent / "meta.json").read_text())

    for meta_kernel_spec in meta["kernels"]:
        if meta_kernel_spec["entry"] == gpu_kernel_ir.name:
            return KernelSpec(gpu_kernel_ir.name,
                              gpu_kernel_ir.body,
                              meta_kernel_spec["arg_order"],
                              tuple(meta_kernel_spec["grid"]),
                              tuple(meta_kernel_spec["block"]),
                              meta_kernel_spec["smem_bytes"],
                              {arg["name"]: arg["shape"] for arg in meta_kernel_spec["args"]}
                             )
    sys.exit(f"{ir_path}: no kernel '{gpu_kernel_ir.name}' in meta.json")
