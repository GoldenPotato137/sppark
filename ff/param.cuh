#pragma once

namespace bignum_mult
{
    __device__ inline  constexpr int LEN_A = 8;
    __device__ inline  constexpr int LEN_B = 8;
    __device__ inline  constexpr int LEN_RESULT = LEN_A + LEN_B;

    namespace tensor_param
    {
        __device__ constexpr int K = LEN_A * 4;
        __device__ constexpr int N = K * 2;
    }
}