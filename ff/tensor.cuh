
#include <mma.h>
#include <cute/tensor.hpp>
#include <cooperative_groups/memcpy_async.h>
#include <cuda/pipeline>
#include "param.cuh"

namespace tensor_mont
{
    using namespace bignum_mult;
    using namespace bignum_mult::tensor_param;
    using namespace cute;

    const int WARPS_PER_BLOCK = 4;
    const int WMMA_M = 16 * WARPS_PER_BLOCK, WMMA_N = 16, WMMA_K = 16;
    const int THREADS_PER_BLOCK = 32 * WARPS_PER_BLOCK;
    __device__ constexpr int C_OUTPUT_RATE = 16;
    __device__ constexpr int N_PER_OUTPUT = C_OUTPUT_RATE / (WMMA_N / 4);
    __device__ constexpr int K_CNT = K / WMMA_K;

    using MMA = decltype(make_tiled_mma(
        SM80_16x8x16_S32U8U8S32_TN{},
        make_layout(Shape<Int<WARPS_PER_BLOCK>, _1, _1>{}),
        Tile<Int<WMMA_M>,
             Layout<Shape<_2, _4, _2>,
                    Stride<_1, _4, _2>>,
             _16>{}));

    __device__ inline void big_num_mult(uint8_t *a, uint8_t *b, uint32_t *result, size_t M)
    {
        __syncthreads();
        using namespace cute;
        using namespace nvcuda;
        auto warp_id = threadIdx.x / 32;
        auto lane_id = threadIdx.x % 32;
        __shared__ uint32_t c_ptr[WMMA_M * WMMA_K / 4 * 2 + 16 * WARPS_PER_BLOCK];
        auto a_g = make_tensor(make_smem_ptr(a), make_shape(M, Int<K>{}), LayoutRight{});
        auto b2_g = make_tensor(make_smem_ptr(b), make_shape(Int<N>{}, Int<K>{}), make_stride(Int<K>{}, _1{}));
        auto b_s = make_tensor(make_smem_ptr(b), make_shape(Int<K>{}), make_stride(_1{}));
        auto c_g = make_tensor(make_smem_ptr(result), make_shape(M, Int<LEN_RESULT>{}), make_stride(Int<LEN_RESULT>{}, _1{}));
        auto c_s = make_tensor(make_smem_ptr(c_ptr), make_shape(_8{}, _8{}, _2{}, Int<WARPS_PER_BLOCK>{}), make_stride(_1{}, _8{}, Int<8 * 8 + 16>{}, Int<8 * 8 * 2 + 16>{}));
        auto a_tile_g = local_tile(a_g, make_shape(Int<WMMA_M>{}, Int<WMMA_K>{}), make_coord(_0{}, _));
        int32_t carry = 0;
        uint32_t temp_result[C_OUTPUT_RATE];
        MMA mma;
        auto thr_mma = mma.get_slice(threadIdx.x);

        auto c_fake_g = make_tensor(make_smem_ptr((int32_t *)result), make_shape(M, Int<N>{}), make_stride(_1{}, Int<N>{}));
        auto c_fake_tile_g = local_tile(c_fake_g, make_shape(Int<WMMA_M>{}, Int<WMMA_N>{}), make_coord(0, 0));

        auto a_thr_g = thr_mma.partition_A(a_g);
        auto b_thr_g = thr_mma.partition_B(b2_g);
        auto c_thr_g = thr_mma.partition_C(c_fake_tile_g);
        auto a_thr_r = thr_mma.partition_fragment_A(a_tile_g(_, _, 0));
        auto c_thr_r = make_tensor_like(c_thr_g);

        int32_t thread_b_row_offset = lane_id % 4 * 4;
        int32_t thread_b_col_offset = lane_id / 8 * 4 + ((lane_id / 4 * 4 % 8) != 0);

        // if (threadIdx.x == 0 && blockIdx.x == 1)
        // {
        //     print_tensor(a_g);
        //     printf("\n");
        //     print_tensor(b_s);
        // }
        auto calc_cnt = M / WMMA_M;
        for (auto calc_no = 0; calc_no < calc_cnt; calc_no++)
        {
            for (int32_t n = 0; n < N / WMMA_N; n++)
            {
                auto b2_tile_g = local_tile(b2_g, make_shape(Int<WMMA_N>{}, Int<WMMA_K>{}), make_coord(n, _));
                auto b_thr_r = thr_mma.partition_fragment_B(b2_tile_g(_, _, 0));
                __syncthreads();
                clear(c_thr_r);
                int k_begin = max(0, n - K_CNT), k_end = min(n, K_CNT - 1);
                for (uint32_t k = k_begin; k <= k_end; k++)
                {
                    copy(a_thr_g(_, calc_no, k), a_thr_r);
                    // copy(b_thr_g(_, make_coord(_, n), k), b_thr_r);
                    { // calc B
                        clear(b_thr_r);
                        int32_t k16 = k * 16, n16 = n * 16;
                        for (auto i = 0; i < 4; i++)
                        {
                            int32_t b_index = n16 + thread_b_col_offset - k16 - thread_b_row_offset - i;
                            if (b_index >= 0 && b_index < K)
                                b_thr_r(i) = b[b_index];
                            b_index += 2;
                            if (b_index >= 0 && b_index < K)
                                b_thr_r(i + 4) = b[b_index];
                        }
                        __syncthreads();
                    }
                    gemm(thr_mma, a_thr_r, b_thr_r, c_thr_r);
                }

                uint64_t num1 = c_thr_r(0) + ((uint64_t)c_thr_r(1) << 8) + ((uint64_t)c_thr_r(4) << 16) + ((uint64_t)c_thr_r(5) << 24);
                uint64_t num2 = c_thr_r(2) + ((uint64_t)c_thr_r(3) << 8) + ((uint64_t)c_thr_r(6) << 16) + ((uint64_t)c_thr_r(7) << 24);
                uint32_t carry1 = num1 >> 32, base1 = num1 & 0xFFFFFFFF;
                uint32_t carry2 = num2 >> 32, base2 = num2 & 0xFFFFFFFF;
                uint32_t div4 = lane_id / 4, mod4 = lane_id % 4;
                c_s(div4, mod4, 0, warp_id) = base1;
                c_s(div4, mod4 + 4, 0, warp_id) = carry1;
                c_s(div4, mod4, 1, warp_id) = base2;
                c_s(div4, mod4 + 4, 1, warp_id) = carry2;

                // if (threadIdx.x == 0 && blockIdx.x == 0 && n == 8)
                //     print_tensor(b2_tile_g);

                if (lane_id < WMMA_M / WARPS_PER_BLOCK)
                {
                    uint32_t is_lower = lane_id >= WMMA_M / WARPS_PER_BLOCK / 2;
                    uint32_t tmp_x = threadIdx.x % 8;
                    for (auto i = 0; i < 4; i++)
                    {
                        uint64_t temp = (uint64_t)c_s(tmp_x, i, is_lower, warp_id) + carry + (i > 0 ? c_s(tmp_x, i + 4 - 1, is_lower, warp_id) : 0);
                        temp_result[i + n % N_PER_OUTPUT * (WMMA_N / 4)] = temp;
                        carry = temp >> 32;
                    }
                    carry += c_s(tmp_x, 7, is_lower, warp_id);
                    // 搬运结果
                    if (n % N_PER_OUTPUT == N_PER_OUTPUT - 1)
                    {
                        auto c_tile_g = local_tile(c_g, make_shape(_16{}, Int<C_OUTPUT_RATE>{}), make_coord(calc_no * WARPS_PER_BLOCK + warp_id, n / N_PER_OUTPUT));
                        auto c_tmp_g = c_tile_g(lane_id, _);
                        for (auto i = 0; i < C_OUTPUT_RATE; i += 4)
                        {
                            uint4 vec;
                            vec.x = temp_result[i];
                            vec.y = temp_result[i + 1];
                            vec.z = temp_result[i + 2];
                            vec.w = temp_result[i + 3];
                            *reinterpret_cast<uint4 *>(c_tmp_g.data().get() + i) = vec;
                        }
                        // if (threadIdx.x == 0 && blockIdx.x == 0 && n == N_PER_OUTPUT - 1) print_tensor(c_tile_g);
                    }
                }
                __syncthreads();
            }
        }
    }

    __device__ inline void tensor_calc_kernel(uint32_t *a, uint32_t *b, uint32_t *result, size_t n)
    {
        __syncthreads();
        big_num_mult(reinterpret_cast<uint8_t *>(a), reinterpret_cast<uint8_t *>(b), result, n);
        __syncthreads();
    }
}
