// This TU provides the single definition of BLS12-377 device constants.

#include "bls12-377.hpp"

namespace device {

__device__ __constant__ __align__(16) const uint32_t BLS12_377_P[12] = {
    TO_CUDA_T(0x8508c00000000001), TO_CUDA_T(0x170b5d4430000000),
    TO_CUDA_T(0x1ef3622fba094800), TO_CUDA_T(0x1a22d9f300f5138f),
    TO_CUDA_T(0xc63b05c06ca1493b), TO_CUDA_T(0x01ae3a4617c510ea)
};

__device__ __constant__ __align__(16) const uint32_t BLS12_377_RR[12] = {
    TO_CUDA_T(0xb786686c9400cd22), TO_CUDA_T(0x0329fcaab00431b1),
    TO_CUDA_T(0x22a5f11162d6b46d), TO_CUDA_T(0xbfdf7d03827dc3ac),
    TO_CUDA_T(0x837e92f041790bf9), TO_CUDA_T(0x006dfccb1e914b88)
};

__device__ __constant__ __align__(16) const uint32_t BLS12_377_one[12] = {
    TO_CUDA_T(0x02cdffffffffff68), TO_CUDA_T(0x51409f837fffffb1),
    TO_CUDA_T(0x9f7db3a98a7d3ff2), TO_CUDA_T(0x7b4e97b76e7c6305),
    TO_CUDA_T(0x4cf495bf803c84e8), TO_CUDA_T(0x008d6661e2fdf49a)
};

__device__ __constant__ __align__(16) const uint32_t BLS12_381_Px128[12] = {
    TO_CUDA_T(0x8460000000000080), TO_CUDA_T(0x85aea21800000042),
    TO_CUDA_T(0x79b117dd04a4000b), TO_CUDA_T(0x116cf9807a89c78f),
    TO_CUDA_T(0x1d82e03650a49d8d), TO_CUDA_T(0xd71d230be2887563)
};

__device__ __constant__ uint32_t BLS12_377_M0 = 0xffffffff;

__device__ __constant__ __align__(16) const uint32_t BLS12_377_r[8] = {
    TO_CUDA_T(0x0a11800000000001), TO_CUDA_T(0x59aa76fed0000001),
    TO_CUDA_T(0x60b44d1e5c37b001), TO_CUDA_T(0x12ab655e9a2ca556)
};

__device__ __constant__ __align__(16) const uint32_t BLS12_377_rRR[8] = {
    TO_CUDA_T(0x25d577bab861857b), TO_CUDA_T(0xcc2c27b58860591f),
    TO_CUDA_T(0xa7cc008fe5dc8593), TO_CUDA_T(0x011fdae7eff1c939)
};

__device__ __constant__ __align__(16) const uint32_t BLS12_377_rone[8] = {
    TO_CUDA_T(0x7d1c7ffffffffff3), TO_CUDA_T(0x7257f50f6ffffff2),
    TO_CUDA_T(0x16d81575512c0fee), TO_CUDA_T(0x0d4bda322bbb9a9d)
};

__device__ __constant__ __align__(16) const uint32_t BLS12_377_rx8[8] = {
    TO_CUDA_T(0x508c000000000008), TO_CUDA_T(0xcd53b7f680000008),
    TO_CUDA_T(0x05a268f2e1bd800a), TO_CUDA_T(0x955b2af4d1652ab3)
};

__device__ __constant__ uint32_t BLS12_377_m0 = 0xffffffff;

} // namespace device
