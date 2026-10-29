extern "C" __global__
__launch_bounds__(256) void k_add_pointwise(const float* mul_c1, //hardcoded to 2
                                            const float* add_c1, //hardcoded to 1
                                            const float* x,      //float arr[4096]
                                            float* add)          //float arr[4096]
                                            {   
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
    //blockdim=256, gridim=4
    // _gid 0..1023
        int a0 = _gid;
        float in1__u0 = mul_c1[0]; //2
        float in2__u0 = x[a0 * 4]; 
        float in0__u0 = add_c1[0]; //1
        float v1__u0 = in0__u0 + (in1__u0 * in2__u0);

        float in1__u1 = mul_c1[0];
        float in2__u1 = x[a0 * 4 + 1];
        float in0__u1 = add_c1[0];
        float v1__u1 = in0__u1 + (in1__u1 * in2__u1);

        float in1__u2 = mul_c1[0];
        float in2__u2 = x[a0 * 4 + 2];
        float in0__u2 = add_c1[0];
        float v1__u2 = in0__u2 + (in1__u2 * in2__u2);

        float in1__u3 = mul_c1[0];
        float in2__u3 = x[a0 * 4 + 3];
        float in0__u3 = add_c1[0];
        float v1__u3 = in0__u3 + (in1__u3 * in2__u3);

        float4 _vs_v1__u0_93c81c5a = make_float4(v1__u0, v1__u1, v1__u2, v1__u3);
        *reinterpret_cast<float4*>(&add[a0 * 4]) = _vs_v1__u0_93c81c5a;
}
