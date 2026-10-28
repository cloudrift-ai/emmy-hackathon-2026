"""Regression tests for PTX forms emitted by kernelc that do not need a GPU."""

from __future__ import annotations

import unittest

import numpy as np

from kernelc.arith import Arith
from kernelc.peephole import vectorize_loads
from kernelc.ptx import Emitter, imm_f32, imm_int
from kernelc.validation import result_array
from tools.ptxemu import Kernel, Machine

try:
    from kernelc.codegen import KernelCompiler
except ModuleNotFoundError:  # The lightweight emulator tests do not need Emmy.
    KernelCompiler = None


class PtxEmulatorRegressionTests(unittest.TestCase):
    def test_peephole_merges_aligned_f64_pairs(self) -> None:
        lines, merged = vectorize_loads([
            "\tld.global.nc.f64\t%fd1, [%rd1];",
            "\tld.global.nc.f64\t%fd2, [%rd1+8];",
        ], {"%rd1": 16})

        self.assertEqual(merged, 1)
        self.assertEqual(lines, ["\tld.global.nc.v2.f64\t{%fd1, %fd2}, [%rd1];"])

    @unittest.skipIf(KernelCompiler is None, "emmy-ml is not installed")
    def test_double_cast_stays_f64(self) -> None:
        compiler = object.__new__(KernelCompiler)
        compiler.ar = Arith(Emitter())

        value = compiler._cast(imm_f32(1.25), "double")

        self.assertEqual(value.ty, "f64")

    def test_emitter_can_suppress_comments(self) -> None:
        emitter = Emitter(comments=False)

        emitter.comment("omitted in a release artifact")
        emitter.ins("ret")

        self.assertEqual(emitter.lines, ["\tret;"])

    def test_unsigned_arithmetic_uses_unsigned_opcodes_and_comparisons(self) -> None:
        emitter = Emitter()
        arith = Arith(emitter)
        left, right = emitter.reg("u32"), emitter.reg("u32")

        quotient = arith.div(left, right)
        comparison = arith.cmp("<", imm_int(0xFFFFFFFF, "u32"), imm_int(0, "u32"))

        self.assertEqual(quotient.ty, "u32")
        self.assertIn("div.u32", emitter.lines[0])
        self.assertTrue(comparison.is_const)
        self.assertFalse(comparison.const)

    def test_result_array_preserves_f64_and_i64(self) -> None:
        f64_ref = np.array([1.23456789012345], dtype=np.float64)
        i64_ref = np.array([2**53 + 1], dtype=np.int64)

        f64 = result_array([float(f64_ref[0])], f64_ref)
        i64 = result_array([int(i64_ref[0])], i64_ref)

        self.assertEqual(f64.dtype, np.dtype(np.float64))
        self.assertEqual(i64.dtype, np.dtype(np.int64))
        np.testing.assert_array_equal(f64, f64_ref)
        np.testing.assert_array_equal(i64, i64_ref)

    def test_reads_f64_hex_immediate(self) -> None:
        """kernelc uses PTX ``0d`` literals for f64 constants and conversions."""
        ptx = """.version 8.0
.target sm_89
.address_size 64
.visible .entry k(
    .param .u64 k_param_0
)
{
    ld.param.u64 %rd1, [k_param_0];
    cvta.to.global.u64 %rd2, %rd1;
    mov.f64 %fd1, 0d3FF0000000000000;
    st.global.f64 [%rd2], %fd1;
    ret;
}
"""
        output = np.zeros(1, dtype=np.float64)
        machine = Machine(Kernel(ptx), 1, 1, 0, ["out"], {"out": output})

        machine.run()

        base = machine.bases[machine.names.index("out")]
        got = machine.gmem[base:base + output.nbytes].copy().view(np.float64)
        np.testing.assert_array_equal(got, np.array([1.0], dtype=np.float64))


if __name__ == "__main__":
    unittest.main()
