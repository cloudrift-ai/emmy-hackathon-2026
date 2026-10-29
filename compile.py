#!/usr/bin/env python3
"""Kernel IR -> PTX compiler.

    python compile.py path/to/<kernel>.ir.json > out.ptx

stdout is reserved for PTX only, so anything for humans goes to stderr.
"""

import sys
from pprint import pprint
from kernel_spec import load_spec
from prologue import PrologueMixin

class Emitter(PrologueMixin):
    """Collects PTX lines and hands out fresh virtual registers.

    Registers are never reused: ptxas does the real register allocation, so we
    only count how many of each kind we asked for. The .reg declarations need
    those final counts, so they are written last (see render).
    """

    # PTX register names emmy uses : PTX reg types
    REG_TYPES = {"r":".b32", 
                 "p":".pred",
                 "f":".f32",
                 "rd":".b64"
                }

    def __init__(self, spec):
        self.spec = spec
        self.ptx_code = []                                 
        self.reg_count = {k: 0 for k in self.REG_TYPES}     # create [r:0, p:0, f:0, rd:0]
        self.buffers = {}                                # buffer name -> pointer register
        self.vars = {}                                   # IR variable name -> register
        self.shapes = dict(spec.shapes)                  # buffer name -> shape (Smem adds its own later)
        self.label_count = 0                             # makes every branch label unique
        self.shared = set()                              # names of Smem buffers (live in .shared, not .global)
        self.smem_init_lines = []                             # saving smem initializations

    # _________________generic helpers_____________________

    def add_line(self, line):
        self.ptx_code.append("        " + line)

    def label(self, name):
        self.ptx_code.append(name + ":")

    def new_reg(self, reg_type):
        n = self.reg_count[reg_type]
        self.reg_count[reg_type] += 1
        return f"%{reg_type}{n}"

    def new_label_id(self):
        n = self.label_count
        self.label_count += 1
        return n

    # _________________WALK THE EMMY IR_________________

    # integer BinaryExpr op -> PTX instruction. Index values are never
    # negative, so unsigned div/rem matches the IR's floor division.
    INT_ARITH = {
        "+"  : "add.u32",
        "-"  : "sub.u32",
        "*"  : "mul.lo.u32",
        "/"  : "div.u32",
        "//" : "div.u32",
        "%"  : "rem.u32",
    }

    # comparison BinaryExpr op -> setp condition code
    COMPARE = {
        "==" : "eq",
        "!=" : "ne",
        "<"  : "lt",
        "<=" : "le",
        ">"  : "gt",
        ">=" : "ge",
    }

    def emit_expr(self, expr):
        # 1, Compile the ast expression tree
        # 2. add the ptx code to ptx_code
        # 3. return the newly assigned register to hold the expression value.

        #Every node gets a fresh register; no constant folding (ptxas cleans up).
        kind = type(expr).__name__

        # children node
        if kind == "Var":
            return self.vars[expr.name]

        # 1. mvoing values into registers because there is 
        # no limit for how many ptx registers we can use
        # 2. we also want to use check diffreent d-types as well

        # children node
        elif kind == "Literal":
            reg = self.new_reg("r")
            self.add_line(f"mov.u32 {reg}, {expr.value};")
            return reg

        # parent node - can't be a child node
        elif kind == "BinaryExpr":
            left = self.emit_expr(expr.left)
            right = self.emit_expr(expr.right)

            # comparisons produce a predicate (%p) instead of an integer
            if expr.op in self.COMPARE:
                pred = self.new_reg("p")
                self.add_line(f"setp.{self.COMPARE[expr.op]}.s32 {pred}, {left}, {right};")
                return pred

            ptx_op = self.INT_ARITH.get(expr.op)
            if ptx_op is None:
                sys.exit(f"unsupported BinaryExpr op '{expr.op}'")
            reg = self.new_reg("r")
            self.add_line(f"{ptx_op} {reg}, {left}, {right};")
            return reg

        else:
            sys.exit(f"emit_expr: unsupported expression {kind}")

    def flat_index(self, buffer, index):
        # 1. turn a multi-dim index tuple into one row-major element index
        # 2. return the reg that holds it
        # x[a0, a1] with shape (64, 512) -> a0*512 + a1, done innermost-last:
        # flat = ((i0 * d1) + i1) * d2 + i2 ...

        if len(index) == 0:
            # sum_1[] : a scalar output, always element 0
            reg = self.new_reg("r")
            self.add_line(f"mov.u32 {reg}, 0;")
            return reg

        shape = self.shapes[buffer]
        if len(index) != len(shape): # both need to 1-d, 2-d, 3-d, etc
            sys.exit(f"flat_index: {buffer} has shape {shape} but index has {len(index)} dims")

        # row major indexing and IR does (rows, column)
        flat = self.emit_expr(index[0])
        # zip 
        for dim, idx_expr in zip(shape[1:], index[1:]):
            scaled = self.new_reg("r")
            self.add_line(f"mul.lo.u32 {scaled}, {flat}, {dim};")
            idx_reg = self.emit_expr(idx_expr) # just a reg = #
            flat = self.new_reg("r")
            self.add_line(f"add.u32 {flat}, {scaled}, {idx_reg};")
        return flat

    def get_address(self, buffer, index_reg, nbytes=4):
        #1. calculate the address of uint8_t buffer[index]
        #2. return the reg that holds the address
        offset_reg = self.new_reg("rd")
        self.add_line(f"mul.wide.u32 {offset_reg}, {index_reg}, {nbytes};") # cell_size * index

        if buffer in self.shared:
            # smem has no reg that holds its base address
            # in PTX, the smem string = its base address 
            base_reg = self.new_reg("rd")
            self.add_line(f"mov.u64 {base_reg}, {buffer};")
        else:
            base_reg = self.buffers[buffer]

        final_byte_addr_reg = self.new_reg("rd")
        self.add_line(f"add.s64 {final_byte_addr_reg}, {base_reg}, {offset_reg};")

        return final_byte_addr_reg

    def get_mem_type(self, buffer):
        if buffer in self.shared:
            return "shared"
        return "global"

    # _________________Statements_________________________
    def emit_stmt(self, stmt):
        # type(stmt) returns it's class, and __name__ makes it a string
        kind = type(stmt).__name__ 
        # matches an attirubte in this Emitter Class named "emit_{kind}"
        handler = getattr(self, f"emit_{kind}", None)
        if handler is None:
            sys.exit(f"handler for this stmt doesn't exit: '{kind}'")

        # emit_Load
        # emit_Assign
        # emit_Write
        handler(stmt)

    def emit_Load(self, stmt):
        # one scalar load per name; vector (v2/v4) loads come later
        if len(stmt.names) != 1:
            sys.exit(f"emit_Load: vector load of {len(stmt.names)} not supported yet")

        idx = self.flat_index(stmt.input, stmt.index)
        addr = self.get_address(stmt.input, idx, stmt.dtype.nbytes)
        new_reg = self.new_reg("f")
        self.add_line(f"ld.{self.get_mem_type(stmt.input)}.f32 {new_reg}, [{addr}];")
        self.vars[stmt.names[0]] = new_reg

    # elementwise op name -> PTX template ({d} = result, {a}/{b} = arguments)
    ELEMENTWISE = {
        "add"       : "add.f32 {d}, {a}, {b};",
        "subtract"  : "sub.f32 {d}, {a}, {b};",
        "multiply"  : "mul.f32 {d}, {a}, {b};",
        "divide"    : "div.rn.f32 {d}, {a}, {b};",
        "maximum"   : "max.f32 {d}, {a}, {b};",
        "relu"      : "max.f32 {d}, {a}, 0f00000000;",   # max(x, 0.0)
        "copy"      : "mov.f32 {d}, {a};",
        "sqrt"      : "sqrt.rn.f32 {d}, {a};",
    }

    def emit_Assign(self, stmt):
        op_name = stmt.op.name
        ptx_template = self.ELEMENTWISE.get(op_name)
        if ptx_template is None:
            sys.exit(f"elementwise op '{op_name}' not supported")
        args = [self.vars[a] for a in stmt.args]
        dst_reg = self.new_reg("f")
        # unary ops have no {b} in their template, so the extra kwarg is ignored
        if len(args) > 1:
            b = args[1] 
        else:
            b = None
        self.add_line(ptx_template.format(d=dst_reg, a=args[0], b=b))
        self.vars[stmt.name] = dst_reg

    # reduce op name -> identity value as a PTX f32 hex literal
    ACCUM_IDENTITY = {
        "add"     : "0f00000000",   # 0.0
        "maximum" : "0fFF800000",   # -inf
    }

    def emit_StridedLoop(self, stmt):
        # for (int i = start; i<=extent; i+=step) { 
        #   body 
        # }
        #
        #
        #     mov.f32 acc, identity(0 or maximum);     (one per Accum in the body)
        #     mov.u32 i, start;
        # TOP_n:
        #     setp.ge.u32 p, i, extent;    i>=extend, then go to END of loop
        #     @p bra END_n;
        #
        #     { body }
        #
        #     add.u32 i, i, step;
        #     bra TOP_n;
        # END_n:
        #
        #
        if stmt.end is not None:
            sys.exit("emit_StridedLoop: loops with an explicit end are not supported")

        # first initialize the accumulator registers, one for each Accum
        for s in stmt.body:
            if type(s).__name__ == "Accum":
                identity = self.ACCUM_IDENTITY.get(s.op.name)
                if identity is None:
                    sys.exit(f"emit_StridedLoop: no identity for Accum op '{s.op.name}'")
                acc_reg = self.new_reg("f")
                self.add_line(f"mov.f32 {acc_reg}, {identity};")
                self.vars[s.name] = acc_reg

        
        # 1. init i
        # 2. start (e.g. a0_co) must not change for code after the loop
        i_reg = self.new_reg("r")
        self.add_line(f"mov.u32 {i_reg}, {self.emit_expr(stmt.start)};")
        self.vars[stmt.axis.name] = i_reg

        # init the step reg i+=step
        step_reg = self.emit_expr(stmt.step)

        n = self.new_label_id()
        self.label(f"TOP_{n}")

        done_reg = self.new_reg("p")

        # i>=extend, then go to END of loop
        self.add_line(f"setp.ge.u32 {done_reg}, {i_reg}, {stmt.axis.extent.value};")
        self.add_line(f"@{done_reg} bra END_{n};")

        for s in stmt.body:
            self.emit_stmt(s)

        self.add_line(f"add.u32 {i_reg}, {i_reg}, {step_reg};")
        self.add_line(f"bra TOP_{n};")

        self.label(f"END_{n}")

    def emit_Accum(self, stmt):
        # acc = op(acc, value), or op(base, value) when base is set (online
        # softmax rescales the old sum first). Written in place into the
        # register emit_StridedLoop gave the accumulator.
        ptx_template = self.ELEMENTWISE.get(stmt.op.name)
        if ptx_template is None:
            sys.exit(f"elementwise op '{stmt.op.name}' not supported")

        acc_reg = self.vars[stmt.name]

        #(base != None) means that acc will always be relacted based on the base + value used in softmax
        if stmt.base is not None:
            left = self.vars[stmt.base]
        else:
            left = acc_reg

        self.add_line(ptx_template.format(d=acc_reg, a=left, b=self.vars[stmt.value]))

    # aka: XOR butterfly
    # used to add up the same reg accros all 32 threads in a warp
    def emit_WarpShuffle(self, stmt):
        #  {float acc0__o = __shfl_xor_sync(__activemask(), acc0, 16);
        #     acc0 = acc0 + acc0__o; }
        #  {float acc0__o = __shfl_xor_sync(__activemask(), acc0, 8);
        #     acc0 = acc0 + acc0__o; }
        #  {float acc0__o = __shfl_xor_sync(__activemask(), acc0, 4);
        #     acc0 = acc0 + acc0__o; }
        #  {float acc0__o = __shfl_xor_sync(__activemask(), acc0, 2);
        #     acc0 = acc0 + acc0__o; }
        #  {float acc0__o = __shfl_xor_sync(__activemask(), acc0, 1);
        #     acc0 = acc0 + acc0__o; }

        offset = stmt.length // 2
        # 16 -> 8 -> 4 -> 2 -> 1 
        while offset >= 1:
            # shuffle every state before combining any: a combine for one state
            # may read another state's value from before this step
            for state, other in zip(stmt.state, stmt.state_b):

                # 1. get the same warp's target thread's acc0
                shuffled_target_reg = self.new_reg("f")
                self.add_line(f"shfl.sync.bfly.b32 {shuffled_target_reg}, {self.vars[state]}, {offset}, 0x1f, 0xffffffff;")

                self.vars[other] = shuffled_target_reg

            for s in stmt.combine_states:
                self.emit_stmt(s)
            offset //= 2

    # combines the per-warp partials that WarpShuffle left in smem, then
    # gives the result to every thread in the block
    def emit_TreeHalve(self, stmt):
        #  for (int s = 2; s > 0; s >>= 1) {
        #      if (warp < s) {
        #          float acc0 = acc0_smem[warp];
        #          float acc0__o = acc0_smem[warp + s];
        #          acc0 = acc0 + acc0__o;
        #          acc0_smem[warp] = acc0;
        #      }
        #      __syncthreads();
        #  }
        #  acc0 = acc0_smem[0];

        if stmt.inner is not None:
            sys.exit("emit_TreeHalve: inner tree not supported")

        nbytes = stmt.dtype.nbytes
        tid_reg = self.vars[stmt.tid_var]

        if stmt.barrier_count is None:
            barrier = f"bar.sync {stmt.barrier_id};"
        else:
            barrier = f"bar.sync {stmt.barrier_id}, {stmt.barrier_count};"

        # length is static, so the loop is unrolled here: s = length/2 ... 1
        s = stmt.length // 2
        while s >= 1:
            n = self.new_label_id()
            skip_reg = self.new_reg("p")
            self.add_line(f"setp.ge.u32 {skip_reg}, {tid_reg}, {s};")   # warp >= s skips
            self.add_line(f"@{skip_reg} bra TREE_SKIP_{n};")

            partner_reg = self.new_reg("r")
            self.add_line(f"add.u32 {partner_reg}, {tid_reg}, {s};")

            # load every state before combining: one combine may read several states
            my_addrs = []
            for buf, state, other in zip(stmt.bufs, stmt.state, stmt.state_b):
                my_addr = self.get_address(buf, tid_reg, nbytes)
                my_addrs.append(my_addr)
                state_reg = self.new_reg("f")
                self.add_line(f"ld.shared.f32 {state_reg}, [{my_addr}];")
                self.vars[state] = state_reg

                other_addr = self.get_address(buf, partner_reg, nbytes)
                other_reg = self.new_reg("f")
                self.add_line(f"ld.shared.f32 {other_reg}, [{other_addr}];")
                self.vars[other] = other_reg

            for c in stmt.combine_states:
                self.emit_stmt(c)

            for my_addr, state in zip(my_addrs, stmt.state):
                self.add_line(f"st.shared.f32 [{my_addr}], {self.vars[state]};")

            # every thread reaches the barrier, so it sits outside the if
            self.label(f"TREE_SKIP_{n}")
            self.add_line(barrier)
            s //= 2

        # broadcast: every thread reads the result from slot 0
        zero_reg = self.new_reg("r")
        self.add_line(f"mov.u32 {zero_reg}, 0;")
        for buf, state in zip(stmt.bufs, stmt.state):
            addr = self.get_address(buf, zero_reg, nbytes)
            result_reg = self.new_reg("f")
            self.add_line(f"ld.shared.f32 {result_reg}, [{addr}];")
            self.vars[state] = result_reg

    def emit_Write(self, stmt):

        reg_list_to_write = [self.vars[v] for v in stmt.values]
        reg_idx = self.flat_index(stmt.output, stmt.index)
        base_buffer = stmt.output
        addr_reg = self.get_address(base_buffer, reg_idx, stmt.value_dtype.nbytes)
        mem_type = self.get_mem_type(base_buffer)
        if stmt.atomic == True:
            # atomic adds go one value at a time: red.global.add has no vector form here
            for i, val in enumerate(reg_list_to_write):
                self.add_line(f"red.{mem_type}.add.f32 [{addr_reg}+{i * stmt.value_dtype.nbytes}], {val};")
        elif len(reg_list_to_write) == 1:
            self.add_line(f"st.{mem_type}.f32 [{addr_reg}], {reg_list_to_write[0]};")
        elif len(reg_list_to_write) in (2, 4):
            # EX: st.global.v2.f32 [%rd_addr], {%f_v0__u0, %f_v0__u1};
            self.add_line(f"st.{mem_type}.v{len(reg_list_to_write)}.f32 [{addr_reg}], {{{', '.join(reg_list_to_write)}}};")
        else:
            sys.exit(f"emit_Write: group of {len(reg_list_to_write)} values not supported")

    def emit_Smem(self, stmt):
        # .shared .align 4 .b8 acc0_smem[16];
        # Declared before .visible .entry so later smem Load/Write=ld.shared/st.shared
        if stmt.name in self.shared:
            sys.exit(f"emit_Smem: {stmt.name} declared twice")
        num_elems = 1
        for extent in stmt.extents:
            num_elems *= extent
        self.smem_init_lines.append(f".shared .align 4 .b8 {stmt.name}[{num_elems * 4}];")
        self.shared.add(stmt.name)
        self.shapes[stmt.name] = tuple(stmt.extents)

    def emit_Sync(self, stmt):
        if stmt.warp == True:
            sys.exit("warp-scope sync not supported yet")
        self.add_line(f"bar.sync {stmt.barrier_id};")

    def emit_Cond(self, cond_body_stmt):
        # if (cond) 
        #   { body } 
        # else 
        #   { else_body }
        #
        #
        #     @!p bra ELSE_n;
        #     {body}
        #     bra END_n;
        # ELSE_n:
        #     { else_body }
        # END_n:


        n = self.new_label_id()
        pred = self.emit_expr(cond_body_stmt.cond)  
        self.add_line(f"@!{pred} bra ELSE_{n};")
        
        for s in cond_body_stmt.body:
            self.emit_stmt(s)
        self.add_line(f"bra END_{n};")

        self.label(f"ELSE_{n}")
        for s in cond_body_stmt.else_body:
            self.emit_stmt(s)

        self.label(f"END_{n}")

    def render(self):
        # Build the body first: the .reg counts are only known once every
        # instruction has asked for its registers.

        # ___Prologue___

        self.load_input_output_pointers()
        self.init_thread_ids()
        tile = self.spec.body[0]
        # print(tile, file=sys.stderr)
        self.bounds_guard(tile)
        self.decode_axes(tile)

        # ___Walk____
        print("\n\n", file=sys.stderr)
        for stmt in tile.body:
            print(stmt, file=sys.stderr)
            self.ptx_code.append("\n")
            self.emit_stmt(stmt)

        # ___Emit___


        out = self.header()
        if self.smem_init_lines:
            out += self.smem_init_lines + [""]
        out += self.signature() + ["{"]
        out += self.init_gpu_regs() + [""]
        out += self.ptx_code + ["DONE:", "        ret;", "}"]

        return "\n".join(out) + "\n" # REMEBER TO keep it a string!

def main():
    if len(sys.argv) != 2:
        sys.exit("should only have 2 inputs: argv[0]=compile.py, argv[1]=IR")
    spec = load_spec(sys.argv[1])

    # ____debug graph IR on the terminal______
    print(f"---- IR body of {spec.name} ----", file=sys.stderr)
    pprint(spec.body, stream=sys.stderr, width=100)  # pprint breaks lines and indents nested statements

    em = Emitter(spec)
    sys.stdout.write(em.render()) # writes to -> out.ptx


if __name__ == "__main__":
    main()
