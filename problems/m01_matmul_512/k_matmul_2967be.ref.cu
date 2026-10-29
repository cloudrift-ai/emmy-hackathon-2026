//ca = cache all. global -> L2 -> L1 -> shared
//cg = cache global. global -> L2 -> shared 

static __device__ __forceinline__ void emmy_cp_async_cg(void* smem, const void* gmem) {
    unsigned addr = __cvta_generic_to_shared(smem);

    //memory means don't reorder loads/stores
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(addr), "l"(gmem) : "memory"); 
}

template <int Bytes>

static __device__ __forceinline__ void emmy_cp_async_ca(void* smem, const void* gmem) {
    unsigned addr = __cvta_generic_to_shared(smem);
    asm volatile("cp.async.ca.shared.global [%0], [%1], %2;\n" :: "r"(addr), "l"(gmem), "n"(Bytes) : "memory");
}

static __device__ __forceinline__ void emmy_cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" ::: "memory");
}

template <int N>
static __device__ __forceinline__ void emmy_cp_async_wait() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N) : "memory");
}

extern "C" __global__
__launch_bounds__(512) void k_matmul_2967be(const float* x1, const float* x0, float* matmul) {
    extern __shared__ __align__(16) unsigned char _smem_pool[];  // 73728 bytes = 18432 floats declared in meta.json
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
        int a0_b = _gid / 4096;
        int a1_b = (_gid / 512) % 8;
        int a0_u = (_gid / 32) % 16;
        int a1_u = (_gid) % 32;
        float acc0__c0_0 = 0.0f;
        float acc0__c0_1 = 0.0f;
        float acc0__c1_0 = 0.0f;
        float acc0__c1_1 = 0.0f;

        // a_smem and b_smem point to the shared memory, just different pointer
        float* _a_smem = reinterpret_cast<float*>(_smem_pool + 0);
        float* _b_smem = reinterpret_cast<float*>(_smem_pool + 24576);

        // a_smem <- x0 
        /*
        (a0_u * 32 + a1_u) = tid within a block
        
        
        (a0_u * 32 + a1_u) / 16 gets the warp id (0...15) within the block
        
        */
        emmy_cp_async_cg(&_a_smem[0 + ((a0_u * 32 + a1_u) * 4 / 64) * 64 + 
                            ((a0_u * 32 + a1_u) * 4 % 64)],
                         &x0[(a0_b * 32 + (a0_u * 32 + a1_u) / 16) * 512 + 4 * 
                            ((a0_u * 32 + a1_u) % 16)]);

        emmy_cp_async_cg(&_b_smem[0 + ((a0_u * 32 + a1_u) * 4 / 64) * 64 + 
                        ((a0_u * 32 + a1_u) * 4 % 64)],
                         &x1[(a0_u * 32 + a1_u) / 16 * 512 + (a1_b * 64 + 4 * 
                        ((a0_u * 32 + a1_u) % 16))]);

        emmy_cp_async_cg(&_b_smem[2048 + ((a0_u * 32 + a1_u) * 4 / 64) * 64 + 
                        ((a0_u * 32 + a1_u) * 4 % 64)],
                         &x1[((a0_u * 32 + a1_u) / 16 + 32) * 512 + (a1_b * 64 + 4 * 
                        ((a0_u * 32 + a1_u) % 16))]);

        emmy_cp_async_commit();

        emmy_cp_async_cg(&_a_smem[2048 + ((a0_u * 32 + a1_u) * 4 / 64) * 64 + 
                        ((a0_u * 32 + a1_u) * 4 % 64)],
                         &x0[(a0_b * 32 + (a0_u * 32 + a1_u) / 16) * 512 + (64 + 4 * 
                        ((a0_u * 32 + a1_u) % 16))]);

        emmy_cp_async_cg(&_b_smem[4096 + ((a0_u * 32 + a1_u) * 4 / 64) * 64 + ((a0_u * 32 + a1_u) * 4 % 64)],
                         &x1[(64 + (a0_u * 32 + a1_u) / 16) * 512 + (a1_b * 64 + 4 * ((a0_u * 32 + a1_u) % 16))]);

        emmy_cp_async_cg(&_b_smem[6144 + ((a0_u * 32 + a1_u) * 4 / 64) * 64 + ((a0_u * 32 + a1_u) * 4 % 64)],
                         &x1[(64 + ((a0_u * 32 + a1_u) / 16 + 32)) * 512 + (a1_b * 64 + 4 * ((a0_u * 32 + a1_u) % 16))]);

        emmy_cp_async_commit();

        for (int _ks = 0; _ks < 512; _ks += 64) {
            emmy_cp_async_cg(&_a_smem[(_ks / 64 + 2) % 3 * 32 * 64 + ((a0_u * 32 + a1_u) * 4 / 64) * 64 + 
                            ((a0_u * 32 + a1_u) * 4 % 64)],
                             &x0[(a0_b * 32 + (a0_u * 32 + a1_u) / 16) * 512 + (((_ks + 128 < 512) 
                                ? (_ks + 128) :
                                  (448)) + 4 * ((a0_u * 32 + a1_u) % 16))]
                            );

            emmy_cp_async_cg(&_b_smem[(_ks / 64 + 2) % 3 * 64 * 64 + ((a0_u * 32 + a1_u) * 4 / 64) * 64 + ((a0_u * 32 + a1_u) * 4 % 64)],
                             &x1[(((_ks + 128 < 512) ? (_ks + 128) : (448)) + (a0_u * 32 + a1_u) / 16) * 512 + (a1_b * 64 + 4 * ((a0_u * 32 + a1_u) % 16))]);
            emmy_cp_async_cg(&_b_smem[((_ks / 64 + 2) % 3 * 64 + 32) * 64 + ((a0_u * 32 + a1_u) * 4 / 64) * 64 + ((a0_u * 32 + a1_u) * 4 % 64)],
                             &x1[(((_ks + 128 < 512) ? (_ks + 128) : (448)) + ((a0_u * 32 + a1_u) / 16 + 32)) * 512 + (a1_b * 64 + 4 * ((a0_u * 32 + a1_u) % 16))]);

            emmy_cp_async_commit();
            emmy_cp_async_wait<2>();

            __syncthreads();

            #pragma unroll
            for (int _ki = 0; _ki < 64; _ki++) {
                float in1__c0_0 = _b_smem[(_ks / 64 % 3 * 64 + _ki) * 64 + (a1_b * 64 + a1_u * 2 - a1_b * 64)];
                float in0__c0_0 = _a_smem[(_ks / 64 % 3 * 32 + (a0_b * 32 + a0_u * 2 - a0_b * 32)) * 64 + _ki];
                float acc0__v__c0_0 = in1__c0_0 * in0__c0_0;
                acc0__c0_0 += acc0__v__c0_0;
                float in1__c0_1 = _b_smem[(_ks / 64 % 3 * 64 + _ki) * 64 + (a1_b * 64 + a1_u * 2 + 1 - a1_b * 64)];
                float acc0__v__c0_1 = in1__c0_1 * in0__c0_0;
                acc0__c0_1 += acc0__v__c0_1;
                float in0__c1_0 = _a_smem[(_ks / 64 % 3 * 32 + (a0_b * 32 + a0_u * 2 + 1 - a0_b * 32)) * 64 + _ki];
                float acc0__v__c1_0 = in1__c0_0 * in0__c1_0;
                acc0__c1_0 += acc0__v__c1_0;
                float acc0__v__c1_1 = in1__c0_1 * in0__c1_0;
                acc0__c1_1 += acc0__v__c1_1;
            }
            __syncthreads();
        }

        float2 _vs_acc0__c0_0_4f84ed4f = make_float2(acc0__c0_0, acc0__c0_1);
        *reinterpret_cast<float2*>(&matmul[(a0_b * 32 + a0_u * 2) * 512 + (a1_b * 64 + a1_u * 2)]) = _vs_acc0__c0_0_4f84ed4f;
        float2 _vs_acc0__c1_0_7fa8d53a = make_float2(acc0__c1_0, acc0__c1_1);
        *reinterpret_cast<float2*>(&matmul[(a0_b * 32 + a0_u * 2 + 1) * 512 + (a1_b * 64 + a1_u * 2)]) = _vs_acc0__c1_0_7fa8d53a;
}
