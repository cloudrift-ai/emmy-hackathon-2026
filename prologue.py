"""PTX prologue: header, entry signature, .reg declarations, param pointers, thread ids, bounds guard."""

import sys


class PrologueMixin:
    """Prologue methods for Emitter.

    Only works mixed into Emitter: relies on self.spec, self.new_reg,
    self.add_line, self.buffers and self.REG_TYPES.
    """

    def header(self):
        return [".version 8.0", ".target sm_89", ".address_size 64", ""]

    def signature(self):
        signature_ptx = [f".visible .entry {self.spec.name}("]
        for i, name in enumerate(self.spec.params):
            if i < (len(self.spec.params) - 1):
                signature_ptx.append(f"        .param .u64 param_{name},")
            else:
                signature_ptx.append(f"        .param .u64 param_{name}")
        signature_ptx.append(")")
        return signature_ptx

    def init_gpu_regs(self):
        lines = []
        for reg_name, ptx_type in self.REG_TYPES.items():
            num_reg = max(1, self.reg_count[reg_name])
            lines.append(f"        .reg {ptx_type}       %{reg_name}<{num_reg}>;")
        return lines

    def load_input_output_pointers(self):
        for name in self.spec.params:
            rd_reg_ptr = self.new_reg("rd")
            global_reg_ptr = self.new_reg("rd")
            self.add_line(f"ld.param.u64 {rd_reg_ptr}, [param_{name}];")
            self.add_line(f"cvta.to.global.u64 {global_reg_ptr}, {rd_reg_ptr}; \n")
            self.buffers[name] = global_reg_ptr

    def init_thread_ids(self):
        self.tid = self.new_reg("r")
        blockIdx_reg = self.new_reg("r")
        blockDimx_reg = self.new_reg("r")
        self.gid = self.new_reg("r")

        self.add_line(f"mov.u32 {self.tid}, %tid.x;")
        self.add_line(f"mov.u32 {blockIdx_reg}, %ctaid.x;")
        self.add_line(f"mov.u32 {blockDimx_reg}, %ntid.x;")
        #multiply-add: d = a * b + c.
        self.add_line(f"mad.lo.u32 {self.gid}, {blockDimx_reg}, {blockIdx_reg}, {self.tid}; \n")

        # lane and warp
        lane_reg = self.new_reg("r")
        self.add_line(f"and.b32 {lane_reg}, {self.tid}, 31;")   # lane = tid % 32
        self.vars["lane"] = lane_reg

        warp_reg = self.new_reg("r")
        self.add_line(f"shr.u32 {warp_reg}, {self.tid}, 5; \n")  # warp = tid / 32
        self.vars["warp"] = warp_reg


    #Threads with gid >= N (N = product of the Tile extents) skip to DONE.
    # Only emitted when the launch has more threads than cells: a thread that
    # skips to DONE never reaches bar.sync, and the rest of its block would hang.
    def bounds_guard(self, tile):
        n_cells = 1
        for axis in tile.axes:
            n_cells *= axis.extent.value
        n_threads = self.spec.grid[0] * self.spec.block[0]
        # this needs to be done so reduction kernels don't hang, the threads need to reach 
        # sync threads
        if n_cells >= n_threads:
            return
        skip_thread = self.new_reg("p")
        self.add_line(f"setp.ge.u32 {skip_thread}, {self.gid}, {n_cells};")
        self.add_line(f"@{skip_thread} bra DONE; \n")

    def decode_axes(self, tile):
        """Split gid into the Tile's axis variables, innermost axis first.

        Row-major: for axes (a0, a1) with a1 extent E, a1 = gid % E and a0 = gid / E.
        The outermost axis needs no modulo; the innermost needs no division
        before its own modulo. Results go into self.vars for index expressions.
        """
        rem = self.gid
        for i, axis in reversed(list(enumerate(tile.axes))):
            print("i: ", i, "axis: ", axis, file=sys.stderr)
            if i == 0:
                self.vars[axis.name] = rem        # outermost: gridDim
                break
            dim = axis.extent.value
            axis_reg = self.new_reg("r")
            self.add_line(f"rem.u32 {axis_reg}, {rem}, {dim};") # %
            self.vars[axis.name] = axis_reg #this thread's reg that holds this value

            quotient = self.new_reg("r")
            self.add_line(f"div.u32 {quotient}, {rem}, {dim};") #/

            rem = quotient
        self.add_line("")
