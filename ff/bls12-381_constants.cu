// This TU provides the single definition of BLS12-381 device constants.

#include "bls12-381.hpp"

namespace device {

__device__ __constant__ __align__(16) const uint32_t BLS12_381_P[12] = {
    TO_CUDA_T(0xb9feffffffffaaab), TO_CUDA_T(0x1eabfffeb153ffff),
    TO_CUDA_T(0x6730d2a0f6b0f624), TO_CUDA_T(0x64774b84f38512bf),
    TO_CUDA_T(0x4b1ba7b6434bacd7), TO_CUDA_T(0x1a0111ea397fe69a)
};

__device__ __constant__ __align__(16) const uint32_t BLS12_381_RR[12] = {
    TO_CUDA_T(0xf4df1f341c341746), TO_CUDA_T(0x0a76e6a609d104f1),
    TO_CUDA_T(0x8de5476c4c95b6d5), TO_CUDA_T(0x67eb88a9939d83c0),
    TO_CUDA_T(0x9a793e85b519952d), TO_CUDA_T(0x11988fe592cae3aa)
};

__device__ __constant__ __align__(16) const uint32_t BLS12_381_one[12] = {
    TO_CUDA_T(0x760900000002fffd), TO_CUDA_T(0xebf4000bc40c0002),
    TO_CUDA_T(0x5f48985753c758ba), TO_CUDA_T(0x77ce585370525745),
    TO_CUDA_T(0x5c071a97a256ec6d), TO_CUDA_T(0x15f65ec3fa80e493)
};

__device__ __constant__ __align__(16) const uint32_t BLS12_381_Px8[12] = {
    TO_CUDA_T(0xcff7fffffffd5558), TO_CUDA_T(0xf55ffff58a9ffffd),
    TO_CUDA_T(0x39869507b587b120), TO_CUDA_T(0x23ba5c279c2895fb),
    TO_CUDA_T(0x58dd3db21a5d66bb), TO_CUDA_T(0xd0088f51cbff34d2)
};

__device__ __constant__ const uint32_t BLS12_381_M0 = 0xfffcfffd;

__device__ __constant__ __align__(16) const uint32_t BLS12_381_r[8] = {
    TO_CUDA_T(0xffffffff00000001), TO_CUDA_T(0x53bda402fffe5bfe),
    TO_CUDA_T(0x3339d80809a1d805), TO_CUDA_T(0x73eda753299d7d48)
};

__device__ __constant__ __align__(16) const uint32_t BLS12_381_rRR[8] = {
    TO_CUDA_T(0xc999e990f3f29c6d), TO_CUDA_T(0x2b6cedcb87925c23),
    TO_CUDA_T(0x05d314967254398f), TO_CUDA_T(0x0748d9d99f59ff11)
};

__device__ __constant__ __align__(16) const uint32_t BLS12_381_rone[8] = {
    TO_CUDA_T(0x00000001fffffffe), TO_CUDA_T(0x5884b7fa00034802),
    TO_CUDA_T(0x998c4fefecbc4ff5), TO_CUDA_T(0x1824b159acc5056f)
};

__device__ __constant__ __align__(16) const uint32_t BLS12_381_rx2[8] = {
    TO_CUDA_T(0xfffffffe00000002), TO_CUDA_T(0xa77b4805fffcb7fd),
    TO_CUDA_T(0x6673b0101343b00a), TO_CUDA_T(0xe7db4ea6533afa90),
};

__device__ __constant__ uint32_t BLS12_381_m0 = 0xffffffff;

} // namespace device
