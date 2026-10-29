extern "C" __global__
__launch_bounds__(256) void k_mul_pointwise(const float* x, //arr[2048]
                                            const float* y, //arr[2048]
                                            float* mul)     //arr[2048]
                                            {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
    // blockDim=256, gridDim=2, _gid = 0...511
        int a0 = _gid;
        float in0__u0 = x[a0 * 4];
        float in1__u0 = y[a0 * 4];
        //fmaxf is same thing as max but for floating point
        float v1__u0 = in1__u0 * fmaxf(0.0f, in0__u0);
        float in0__u1 = x[a0 * 4 + 1];
        float in1__u1 = y[a0 * 4 + 1];
        float v1__u1 = in1__u1 * fmaxf(0.0f, in0__u1);
        float in0__u2 = x[a0 * 4 + 2];
        float in1__u2 = y[a0 * 4 + 2];
        float v1__u2 = in1__u2 * fmaxf(0.0f, in0__u2);
        float in0__u3 = x[a0 * 4 + 3];
        float in1__u3 = y[a0 * 4 + 3];
        float v1__u3 = in1__u3 * fmaxf(0.0f, in0__u3);
        float4 _vs_v1__u0_932c6ce9 = make_float4(v1__u0, v1__u1, v1__u2, v1__u3);
        *reinterpret_cast<float4*>(&mul[a0 * 4]) = _vs_v1__u0_932c6ce9;
}
