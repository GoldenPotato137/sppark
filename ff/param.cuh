#pragma once

namespace bignum_mult
{
    static  __device__ inline  constexpr int LEN_A = 8;
    static  __device__ inline  constexpr int LEN_B = 8;
    static  __device__ inline  constexpr int LEN_RESULT = LEN_A + LEN_B;

    namespace tensor_param
    {
        static  __device__ constexpr int K = LEN_A * 4;
        static  __device__ constexpr int N = K * 2;
    }
}