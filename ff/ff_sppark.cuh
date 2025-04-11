
#include <cuda_runtime_api.h>
#include "param.cuh"

namespace ff_sppark
{

    using namespace bignum_mult;

    __device__ __forceinline__ void ff_sppark_calc_kernel(
        const uint32_t *a_shared,
        const uint32_t *b_shared,
        uint32_t *result_shared,
        size_t n)
    {
        auto lane_id = threadIdx.x;
        auto block_dim = blockDim.x;

        for (size_t i = lane_id; i < n; i += block_dim)
        {
            const uint32_t *current_a = a_shared + i * LEN_A;
            uint32_t *current_result = result_shared + i * LEN_RESULT;

            uint32_t temp_a[LEN_A];
            uint32_t temp_result[LEN_RESULT] = {0};

#pragma unroll
            for (int j = 0; j < LEN_A; j += 4)
            {
                uint4 vec = *reinterpret_cast<const uint4 *>(current_a + j);
                temp_a[j] = vec.x;
                temp_a[j + 1] = vec.y;
                temp_a[j + 2] = vec.z;
                temp_a[j + 3] = vec.w;
            }

#pragma unroll
            for (int j = 0; j < LEN_A; j++)
            {
                uint32_t carry = 0;
#pragma unroll
                for (int k = 0; k < LEN_B; k++)
                {
                    uint64_t temp = (uint64_t)temp_a[j] * b_shared[k] + temp_result[j + k] + carry;
                    temp_result[j + k] = temp & 0xFFFFFFFF;
                    carry = temp >> 32;
                }
                if (j + LEN_B < LEN_RESULT)
                {
                    temp_result[j + LEN_B] = carry;
                }
            }

#pragma unroll
            for (int j = 0; j < LEN_RESULT; j += 4)
            {
                if (j + 3 < LEN_RESULT)
                {
                    uint4 vec;
                    vec.x = temp_result[j];
                    vec.y = temp_result[j + 1];
                    vec.z = temp_result[j + 2];
                    vec.w = temp_result[j + 3];
                    *reinterpret_cast<uint4 *>(current_result + j) = vec;
                }
                else
                {
                    for (int k = j; k < LEN_RESULT; k++)
                    {
                        current_result[k] = temp_result[k];
                    }
                }
            }
        }
    }

}
