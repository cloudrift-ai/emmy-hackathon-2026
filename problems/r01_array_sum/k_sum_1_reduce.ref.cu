extern "C" __global__
__launch_bounds__(128) void k_sum_1_reduce(const float* x, float* sum_1) {
    int lane = threadIdx.x & 31; //thread.id within the warp (0..31)
    int warp = threadIdx.x >> 5; // 128 threads launched = 4 warps (0,1,2,3)
    int _gid = blockIdx.x * blockDim.x + threadIdx.x; //thread.id (0...127)
        int a0_co = _gid;
        float acc0 = 0.0f;

        // each thread sums up 8 elements in the 1024-element input array
        for (int a0 = a0_co; a0 < 1024; a0 += 128) {
            float in0 = x[a0];
            acc0 += in0;
        } 

        /*
        what is __activemask()
            __activemask() is a 32-bit bitmask. each bit represents a lane in a warp (0..31)
            __activemask()'s bit is set to 0 if its lane is running. 
                ex, if all lanes in the warp is running 16 is running, then __activemask() = 0xFFFFFFFF
                ex, if lanes 0->16 is only active, then __activemask() = 0x0x01FF

        threads in the same warp can read each other's private registers. 

        what is output = __shfl_xor_sync(mask,x,y)
            this is "warp-level shuffle intrinsic". a caller thread gets the callee's acc0 register value. the callee's threadid is 
            calculated by callee's lane = [caller's lane xor 16) 
                __shfl_xor_sync(mask,x,y)
                ex: [0 xor 16] = 16 = lane 16's x register = output
                ex: [1 xor 16] = 17
                ex: [2 xor 16] = 18
                ex: [14 xor 16] = 30
                ex: [15 xor 16] = 31
                ex: [16 xor 16] = 0
                ex: [31 xor 16] = 15

        */

        /* { } scope limitters in c++ are used when variabled declared in them only exit within the scope*/
        {
            float acc0__o = __shfl_xor_sync(__activemask(), acc0, 16);
            acc0 = acc0 + acc0__o;
        }

        {
            /*
                ex: [0 xor 8] = 8 = lane 16's x register = output
                ex: [1 xor 8] = 9
                ex: [2 xor 8] = 10
                ex: [3 xor 8] = 11
                ex: [7 xor 8] = 15
                ex: [8 xor 8] = 0
                ex: [14 xor 8] = 6
                ex: [15 xor 8] = 7
                ex: [16 xor 8] = 24
                ex: [24 xor 8] = 16
                ex: [25 xor 8] = 17
                ex: [31 xor 8] = 23
            */
            float acc0__o = __shfl_xor_sync(__activemask(), acc0, 8);
            acc0 = acc0 + acc0__o;
        }
        {
            float acc0__o = __shfl_xor_sync(__activemask(), acc0, 4);
            acc0 = acc0 + acc0__o;
        }
        {
            float acc0__o = __shfl_xor_sync(__activemask(), acc0, 2);
            acc0 = acc0 + acc0__o;
        }
        {
            float acc0__o = __shfl_xor_sync(__activemask(), acc0, 1);
            acc0 = acc0 + acc0__o;
        }

        // since only 1 block is launched, all threads write to the same shared mem
        __shared__ float acc0_smem[4];

        // lane 0 for all 4 warps write to smem
        if (lane == 0) {
            acc0_smem[warp] = acc0;
        }

        __syncthreads(); //waits for all warps in the same block to finish

        //s = b10 = 2
        //s = b01 = 1
        //s = b00 = 0
        for (int s = 2; s > 0; s >>= 1) {
            if (warp < s) {
                float acc0 = acc0_smem[warp];
                float acc0__o = acc0_smem[warp + s];
                acc0 = acc0 + acc0__o;
                acc0_smem[warp] = acc0;
            }
            __syncthreads();
        }

        acc0 = acc0_smem[0];

        if (a0_co == 0) { // thread.idx=0 out of 128 threads writes to the output
            sum_1[0] = acc0;
        }
}
