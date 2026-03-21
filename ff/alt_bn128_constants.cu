// This TU provides the single definition of ALT_BN128 device constants.

#include "alt_bn128.hpp"

namespace device {

__device__ __constant__ __align__(16) const uint32_t ALT_BN128_P[8] = {
    TO_CUDA_T(0x3c208c16d87cfd47), TO_CUDA_T(0x97816a916871ca8d),
    TO_CUDA_T(0xb85045b68181585d), TO_CUDA_T(0x30644e72e131a029)
};

__device__ __constant__ __align__(16) const uint32_t ALT_BN128_RR[8] = {
    TO_CUDA_T(0xf32cfc5b538afa89), TO_CUDA_T(0xb5e71911d44501fb),
    TO_CUDA_T(0x47ab1eff0a417ff6), TO_CUDA_T(0x06d89f71cab8351f),
};

__device__ __constant__ __align__(16) const uint32_t ALT_BN128_one[8] = {
    TO_CUDA_T(0xd35d438dc58f0d9d), TO_CUDA_T(0x0a78eb28f5c70b3d),
    TO_CUDA_T(0x666ea36f7879462c), TO_CUDA_T(0x0e0a77c19a07df2f)
};

__device__ __constant__ __align__(16) const uint32_t ALT_BN128_Px4[8] = {
    TO_CUDA_T(0xf082305b61f3f51c), TO_CUDA_T(0x5e05aa45a1c72a34),
    TO_CUDA_T(0xe14116da06056176), TO_CUDA_T(0xc19139cb84c680a6)
};

__device__ __constant__ __align__(16) const uint32_t ALT_BN128_MODn[8] = {
    TO_CUDA_T(0x87d20782e4866389), TO_CUDA_T(0x9ede7d651eca6ac9),
    TO_CUDA_T(0xd8afcbd01833da80), TO_CUDA_T(0xf57a22b791888c6b)
};

__device__ __constant__ const uint32_t ALT_BN128_M0 = 0xe4866389;

__device__ __constant__ __align__(16) const uint32_t ALT_BN128_r[8] = {
    TO_CUDA_T(0x43e1f593f0000001), TO_CUDA_T(0x2833e84879b97091),
    TO_CUDA_T(0xb85045b68181585d), TO_CUDA_T(0x30644e72e131a029)
};

__device__ __constant__ __align__(16) const uint32_t ALT_BN128_rRR[8] = {
    TO_CUDA_T(0x1bb8e645ae216da7), TO_CUDA_T(0x53fe3ab1e35c59e3),
    TO_CUDA_T(0x8c49833d53bb8085), TO_CUDA_T(0x0216d0b17f4e44a5)
};

__device__ __constant__ __align__(16) const uint32_t ALT_BN128_rone[8] = {
    TO_CUDA_T(0xac96341c4ffffffb), TO_CUDA_T(0x36fc76959f60cd29),
    TO_CUDA_T(0x666ea36f7879462e), TO_CUDA_T(0x0e0a77c19a07df2f)
};

__device__ __constant__ __align__(16) const uint32_t ALT_BN128_rx4[8] = {
    TO_CUDA_T(0x0f87d64fc0000004), TO_CUDA_T(0xa0cfa121e6e5c245),
    TO_CUDA_T(0xe14116da06056174), TO_CUDA_T(0xc19139cb84c680a6)
};

__device__ __constant__ __align__(16) const uint32_t ALT_BN128_MODN[8] = {
    TO_CUDA_T(0xc2e1f593efffffff), TO_CUDA_T(0x6586864b4c6911b3),
    TO_CUDA_T(0xe39a982899062391), TO_CUDA_T(0x73f82f1d0d8341b2)
};

__device__ __constant__ const uint32_t ALT_BN128_m0 = 0xefffffff;

} // namespace device
