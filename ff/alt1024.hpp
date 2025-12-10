// Copyright Supranational LLC
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#ifndef __SPPARK_FF_ALT1022_HPP__
#define __SPPARK_FF_ALT1022_HPP__

# include <cstdint>
# include "mont_t.cuh"

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsubobject-linkage"
#endif

# pragma nv_diag_suppress 20012

namespace device
{
#define TO_CUDA_T(limb64) (uint32_t)(limb64), (uint32_t)(limb64>>32)
    static __device__ __constant__ __align__(16) const uint32_t ALT_BN128_P[32] = {
                TO_CUDA_T(0xc381353af3ca1b25),
                TO_CUDA_T(0xf879374b6cb8c786),
                TO_CUDA_T(0x2c07e62e04f2050a),
                TO_CUDA_T(0x43336a7fc8ba450f),
                TO_CUDA_T(0xba0408ad9642b1dc),
                TO_CUDA_T(0x6e510015ce648011),
                TO_CUDA_T(0x158029249e28032b),
                TO_CUDA_T(0x8aecbe63dd05e394),
                TO_CUDA_T(0x5e736f0a3c868f69),
                TO_CUDA_T(0x036c7d58fe946c6d),
                TO_CUDA_T(0x99f797c782418c64),
                TO_CUDA_T(0x1f8fc83d85208ad8),
                TO_CUDA_T(0xf01413b6bd00ff6f),
                TO_CUDA_T(0x2bbc442d75dfe4e8),
                TO_CUDA_T(0x980c59ef9be0e4b9),
                TO_CUDA_T(0x38a725f7b5dcacf8)
    };
    static __device__ __constant__ __align__(16) const uint32_t ALT_BN128_RR[32] = { /* (1<<512)%P */
                TO_CUDA_T(0x5077e3988fb21140),
                TO_CUDA_T(0x6799152746790811),
                TO_CUDA_T(0x18526b0df2c6bf6a),
                TO_CUDA_T(0x6520873bb9a8ab0f),
                TO_CUDA_T(0x999aea1c811b46e2),
                TO_CUDA_T(0xd90781e7dad6978c),
                TO_CUDA_T(0x09da8b64fa210f4d),
                TO_CUDA_T(0xbf591ecf678fffa7),
                TO_CUDA_T(0x5e8e1a0764ba1ebd),
                TO_CUDA_T(0x80e49e3551ffe65a),
                TO_CUDA_T(0xe79e82695edc6bd7),
                TO_CUDA_T(0x759d298be4f9e206),
                TO_CUDA_T(0xc7381c9083236c50),
                TO_CUDA_T(0xbd2eb2e8e5ea8481),
                TO_CUDA_T(0xc68a17ef6d565279),
                TO_CUDA_T(0x1592a509b821dd3e)
    };
    static __device__ __constant__ __align__(16) const uint32_t ALT_BN128_one[32] = { /* (1<<256)%P */
                TO_CUDA_T(0xf1fb2b1430d7936c),
                TO_CUDA_T(0x1e1b22d24d1ce1e4),
                TO_CUDA_T(0x4fe06747ec37ebd4),
                TO_CUDA_T(0xf3325600dd16ebc3),
                TO_CUDA_T(0x17efdd49a6f5388e),
                TO_CUDA_T(0x46bbffa8c66dffb9),
                TO_CUDA_T(0xa9ff5b6d875ff352),
                TO_CUDA_T(0xd44d06708be871af),
                TO_CUDA_T(0x863243d70de5c259),
                TO_CUDA_T(0xf24e0a9c05ae4e4a),
                TO_CUDA_T(0x9821a0e1f6f9ce6f),
                TO_CUDA_T(0x81c0df09eb7dd49d),
                TO_CUDA_T(0x3fafb1250bfc0243),
                TO_CUDA_T(0x510eef4a28806c5c),
                TO_CUDA_T(0x9fce9841907c6d1b),
                TO_CUDA_T(0x1d636821288d4c1d)
    };
    static __device__ __constant__ __align__(16) const uint32_t ALT_BN128_Px4[32] = { /* left-aligned value of the modulus */
                TO_CUDA_T(0x0e04d4ebcf286c94),
                TO_CUDA_T(0xe1e4dd2db2e31e1b),
                TO_CUDA_T(0xb01f98b813c8142b),
                TO_CUDA_T(0x0ccda9ff22e9143c),
                TO_CUDA_T(0xe81022b6590ac771),
                TO_CUDA_T(0xb944005739920046),
                TO_CUDA_T(0x5600a49278a00cad),
                TO_CUDA_T(0x2bb2f98f74178e50),
                TO_CUDA_T(0x79cdbc28f21a3da6),
                TO_CUDA_T(0x0db1f563fa51b1b5),
                TO_CUDA_T(0x67de5f1e09063190),
                TO_CUDA_T(0x7e3f20f614822b62),
                TO_CUDA_T(0xc0504edaf403fdbc),
                TO_CUDA_T(0xaef110b5d77f93a3),
                TO_CUDA_T(0x603167be6f8392e4),
                TO_CUDA_T(0xe29c97ded772b3e2)
    };
    static __device__ __constant__ __align__(16) const uint32_t ALT_BN128_MODn[32] = {
                TO_CUDA_T(0xb0361ed645ff7753),
                TO_CUDA_T(0x78a7df5061da728c),
                TO_CUDA_T(0x374db78e60ee3e6f),
                TO_CUDA_T(0x097317596c784987),
                TO_CUDA_T(0xd45e7436810a111d),
                TO_CUDA_T(0x12e1c9ae7f75d997),
                TO_CUDA_T(0x2bdb47e00dac28ed),
                TO_CUDA_T(0x1fbb12f4c61a50e7),
                TO_CUDA_T(0x083433616d76f442),
                TO_CUDA_T(0x1bb6204f25c9ba97),
                TO_CUDA_T(0xd450ae22cf004343),
                TO_CUDA_T(0x58c11f8964462679),
                TO_CUDA_T(0xd784b90423c28ff2),
                TO_CUDA_T(0xdde62774d09cd3a5),
                TO_CUDA_T(0x4cb47ba75ef26b16),
                TO_CUDA_T(0x9606e21248c0b9d9)
    };
    static __device__ __constant__ const uint32_t ALT_BN128_M0 = 0x45ff7753;

    static __device__ __constant__ __align__(16) const uint32_t ALT_BN128_r[32] = {
            TO_CUDA_T(0x47e4fc46461db611),
            TO_CUDA_T(0x9cfe1ebb5734e6ac),
            TO_CUDA_T(0xbe4966ed7a8a5d71),
            TO_CUDA_T(0xad6e6caabb972709),
            TO_CUDA_T(0x395bd0259959b38b),
            TO_CUDA_T(0x32b95c2e34d84153),
            TO_CUDA_T(0x03bdcb769f912d12),
            TO_CUDA_T(0x3980d3b7dc1ab70c),
            TO_CUDA_T(0xcb4841f87c1b4f6a),
            TO_CUDA_T(0x2e7678b3d18b73bd),
            TO_CUDA_T(0x49c059c363847a9c),
            TO_CUDA_T(0x7aad7f55d0170ee6),
            TO_CUDA_T(0x3db604b815076f5a),
            TO_CUDA_T(0xcbf08e442e2557a0),
            TO_CUDA_T(0x325212768f9d218f),
            TO_CUDA_T(0x237cdc3504e2e9f7)
    };
    static __device__ __constant__ __align__(16) const uint32_t ALT_BN128_rRR[32] = { /* (1<<512)%P */
            TO_CUDA_T(0x00588eb548837425),
            TO_CUDA_T(0xb7bbf82845bede17),
            TO_CUDA_T(0xe04eb526b8bbb1c0),
            TO_CUDA_T(0x7a2679b9e108900f),
            TO_CUDA_T(0xba2d4c4dbdc501dc),
            TO_CUDA_T(0x7f2f8c029e4a65df),
            TO_CUDA_T(0xe33996ee13be1115),
            TO_CUDA_T(0x4cdd1c8449bdb6b0),
            TO_CUDA_T(0xaf259f16ca9c3994),
            TO_CUDA_T(0x8b3620e43d626562),
            TO_CUDA_T(0xc868d761cdbda303),
            TO_CUDA_T(0x8fb87128363a1e6a),
            TO_CUDA_T(0x3ccf745a76326434),
            TO_CUDA_T(0xf8df334f23ed7888),
            TO_CUDA_T(0x3e7387f3fd359353),
            TO_CUDA_T(0x14c14c9bc6d741f5)
    };
    static __device__ __constant__ __align__(16) const uint32_t ALT_BN128_rone[32] = { /* (1<<256)%P */
            TO_CUDA_T(0x08bd1a1415300589),
            TO_CUDA_T(0xb50d28e09d8db14a),
            TO_CUDA_T(0xcbfe2f81a63771e4),
            TO_CUDA_T(0x41fb0754deddeebb),
            TO_CUDA_T(0x6e7d4ef8ce8c172e),
            TO_CUDA_T(0x9cee7abc8e1636b9),
            TO_CUDA_T(0xe5cf6fc1a307c480),
            TO_CUDA_T(0x6d7a35f8fb44feab),
            TO_CUDA_T(0x710632349b40d418),
            TO_CUDA_T(0xbac2b315452fd5cf),
            TO_CUDA_T(0xfbbd8ba84760a5ba),
            TO_CUDA_T(0xa54184a74f5e97b3),
            TO_CUDA_T(0x5005def76ccbf486),
            TO_CUDA_T(0x6c6c1c22bcfa9a9e),
            TO_CUDA_T(0x9fc17ec212b41511),
            TO_CUDA_T(0x0795fa8cddcb9a3d)
    };
    static __device__ __constant__ __align__(
            16) const uint32_t ALT_BN128_rx4[32] = { /* left-aligned value of the modulus */
            TO_CUDA_T(0x1f93f1191876d844),
            TO_CUDA_T(0x73f87aed5cd39ab1),
            TO_CUDA_T(0xf9259bb5ea2975c6),
            TO_CUDA_T(0xb5b9b2aaee5c9c26),
            TO_CUDA_T(0xe56f40966566ce2e),
            TO_CUDA_T(0xcae570b8d361054c),
            TO_CUDA_T(0x0ef72dda7e44b448),
            TO_CUDA_T(0xe6034edf706adc30),
            TO_CUDA_T(0x2d2107e1f06d3da8),
            TO_CUDA_T(0xb9d9e2cf462dcef7),
            TO_CUDA_T(0x2701670d8e11ea70),
            TO_CUDA_T(0xeab5fd57405c3b99),
            TO_CUDA_T(0xf6d812e0541dbd69),
            TO_CUDA_T(0x2fc23910b8955e80),
            TO_CUDA_T(0xc94849da3e74863f),
            TO_CUDA_T(0x8df370d4138ba7dc)
    };
    static __device__ __constant__ __align__(
        16) const uint32_t ALT_BN128_MODN[32] = {
            TO_CUDA_T(0xa8a1622fd074050f),
            TO_CUDA_T(0x5b6fe1db2987841c),
            TO_CUDA_T(0x76eab6a10a789c41),
            TO_CUDA_T(0xc6e70b8bbe5a44b4),
            TO_CUDA_T(0x623ebf7c2faddf29),
            TO_CUDA_T(0x25df6651b6f758f1),
            TO_CUDA_T(0x61173cd409fb6f18),
            TO_CUDA_T(0x36025782d1af0594),
            TO_CUDA_T(0xc86eeb6450b73afd),
            TO_CUDA_T(0x02e55d2293fd4eef),
            TO_CUDA_T(0x0c38fb8d7e9d0863),
            TO_CUDA_T(0xeefd09b82fe5701c),
            TO_CUDA_T(0xaa9b7214b6d95823),
            TO_CUDA_T(0x2fd9c539d57e0614),
            TO_CUDA_T(0xe8c1e1f490bd2f8a),
            TO_CUDA_T(0xc67cd3831200a39c)
    };

    static __device__ __constant__ const uint32_t ALT_BN128_m0 = 0xd074050f;

    // device-side field types
    typedef mont_t<1022, device::ALT_BN128_P, 
                device::ALT_BN128_M0,
                device::ALT_BN128_RR, 
                device::ALT_BN128_one,
                device::ALT_BN128_Px4,
                device::ALT_BN128_MODn> fp_mont;

    struct fp_t : public fp_mont
    {
        using mem_t = fp_t;

        __host__ __device__  fp_t() = default;

        __host__ __device__  fp_t(const fp_mont &a) : fp_mont(a) {}

        __device__ fp_t(uint32_t a) : fp_mont(a){}
    };

    typedef mont_t<1022, device::ALT_BN128_r,            // 模数 N = r
                device::ALT_BN128_m0,                   // m0 ≈ N'
                device::ALT_BN128_rRR,                  // R^2 mod r
                device::ALT_BN128_rone,                 // 1 的 Montgomery 表示 = R mod r
                device::ALT_BN128_rx4,                  // 4 的 Montgomery 表示 = 4R mod r
                device::ALT_BN128_MODN> fr_mont;

    struct fr_t : public fr_mont
    {
        using mem_t = fr_t;

        __host__ __device__ __forceinline__ fr_t() {}

        __device__ __forceinline__ fr_t(const uint32_t &a) : fr_mont(a){}

        __device__ __forceinline__ fr_t(const fr_mont &a) : fr_mont(a) {}
    };

    static constexpr int LAMBDA = 1022;
}

// host-side field types
// # include <blst_t.hpp>

// namespace host
// {
//     static const vec256 ALT_BN128_P = {
//             TO_CUDA_T(0xc381353af3ca1b25),
//             TO_CUDA_T(0xf879374b6cb8c786),
//             TO_CUDA_T(0x2c07e62e04f2050a),
//             TO_CUDA_T(0x43336a7fc8ba450f),
//             TO_CUDA_T(0xba0408ad9642b1dc),
//             TO_CUDA_T(0x6e510015ce648011),
//             TO_CUDA_T(0x158029249e28032b),
//             TO_CUDA_T(0x8aecbe63dd05e394),
//             TO_CUDA_T(0x5e736f0a3c868f69),
//             TO_CUDA_T(0x036c7d58fe946c6d),
//             TO_CUDA_T(0x99f797c782418c64),
//             TO_CUDA_T(0x1f8fc83d85208ad8),
//             TO_CUDA_T(0xf01413b6bd00ff6f),
//             TO_CUDA_T(0x2bbc442d75dfe4e8),
//             TO_CUDA_T(0x980c59ef9be0e4b9),
//             TO_CUDA_T(0x38a725f7b5dcacf8)
//     };
//     static const vec256 ALT_BN128_RR = {    /* (1<<512)%P */
//             TO_CUDA_T(0x5077e3988fb21140),
//             TO_CUDA_T(0x6799152746790811),
//             TO_CUDA_T(0x18526b0df2c6bf6a),
//             TO_CUDA_T(0x6520873bb9a8ab0f),
//             TO_CUDA_T(0x999aea1c811b46e2),
//             TO_CUDA_T(0xd90781e7dad6978c),
//             TO_CUDA_T(0x09da8b64fa210f4d),
//             TO_CUDA_T(0xbf591ecf678fffa7),
//             TO_CUDA_T(0x5e8e1a0764ba1ebd),
//             TO_CUDA_T(0x80e49e3551ffe65a),
//             TO_CUDA_T(0xe79e82695edc6bd7),
//             TO_CUDA_T(0x759d298be4f9e206),
//             TO_CUDA_T(0xc7381c9083236c50),
//             TO_CUDA_T(0xbd2eb2e8e5ea8481),
//             TO_CUDA_T(0xc68a17ef6d565279),
//             TO_CUDA_T(0x1592a509b821dd3e)
//     };
//     static const vec256 ALT_BN128_ONE = {   /* (1<<256)%P */
//             TO_CUDA_T(0xf1fb2b1430d7936c),
//             TO_CUDA_T(0x1e1b22d24d1ce1e4),
//             TO_CUDA_T(0x4fe06747ec37ebd4),
//             TO_CUDA_T(0xf3325600dd16ebc3),
//             TO_CUDA_T(0x17efdd49a6f5388e),
//             TO_CUDA_T(0x46bbffa8c66dffb9),
//             TO_CUDA_T(0xa9ff5b6d875ff352),
//             TO_CUDA_T(0xd44d06708be871af),
//             TO_CUDA_T(0x863243d70de5c259),
//             TO_CUDA_T(0xf24e0a9c05ae4e4a),
//             TO_CUDA_T(0x9821a0e1f6f9ce6f),
//             TO_CUDA_T(0x81c0df09eb7dd49d),
//             TO_CUDA_T(0x3fafb1250bfc0243),
//             TO_CUDA_T(0x510eef4a28806c5c),
//             TO_CUDA_T(0x9fce9841907c6d1b),
//             TO_CUDA_T(0x1d636821288d4c1d)
//     };
//     typedef blst_256_t<1022, ALT_BN128_P, 0xb0361ed645ff7753u,
//             ALT_BN128_RR, ALT_BN128_ONE> fp_mont;

//     struct fp_t : public fp_mont
//     {
//         using mem_t = fp_t;

//         inline fp_t() = default;

//         inline fp_t(const fp_mont &a) : fp_mont(a) {}
//     };

//     static const vec256 ALT_BN128_r = {
//             TO_CUDA_T(0x47e4fc46461db611),
//             TO_CUDA_T(0x9cfe1ebb5734e6ac),
//             TO_CUDA_T(0xbe4966ed7a8a5d71),
//             TO_CUDA_T(0xad6e6caabb972709),
//             TO_CUDA_T(0x395bd0259959b38b),
//             TO_CUDA_T(0x32b95c2e34d84153),
//             TO_CUDA_T(0x03bdcb769f912d12),
//             TO_CUDA_T(0x3980d3b7dc1ab70c),
//             TO_CUDA_T(0xcb4841f87c1b4f6a),
//             TO_CUDA_T(0x2e7678b3d18b73bd),
//             TO_CUDA_T(0x49c059c363847a9c),
//             TO_CUDA_T(0x7aad7f55d0170ee6),
//             TO_CUDA_T(0x3db604b815076f5a),
//             TO_CUDA_T(0xcbf08e442e2557a0),
//             TO_CUDA_T(0x325212768f9d218f),
//             TO_CUDA_T(0x237cdc3504e2e9f7)
//     };
//     static const vec256 ALT_BN128_rRR = {   /* (1<<512)%r */
//             TO_CUDA_T(0x00588eb548837425),
//             TO_CUDA_T(0xb7bbf82845bede17),
//             TO_CUDA_T(0xe04eb526b8bbb1c0),
//             TO_CUDA_T(0x7a2679b9e108900f),
//             TO_CUDA_T(0xba2d4c4dbdc501dc),
//             TO_CUDA_T(0x7f2f8c029e4a65df),
//             TO_CUDA_T(0xe33996ee13be1115),
//             TO_CUDA_T(0x4cdd1c8449bdb6b0),
//             TO_CUDA_T(0xaf259f16ca9c3994),
//             TO_CUDA_T(0x8b3620e43d626562),
//             TO_CUDA_T(0xc868d761cdbda303),
//             TO_CUDA_T(0x8fb87128363a1e6a),
//             TO_CUDA_T(0x3ccf745a76326434),
//             TO_CUDA_T(0xf8df334f23ed7888),
//             TO_CUDA_T(0x3e7387f3fd359353),
//             TO_CUDA_T(0x14c14c9bc6d741f5)
//     };
//     static const vec256 ALT_BN128_rONE = {  /* (1<<256)%r */
//             TO_CUDA_T(0x08bd1a1415300589),
//             TO_CUDA_T(0xb50d28e09d8db14a),
//             TO_CUDA_T(0xcbfe2f81a63771e4),
//             TO_CUDA_T(0x41fb0754deddeebb),
//             TO_CUDA_T(0x6e7d4ef8ce8c172e),
//             TO_CUDA_T(0x9cee7abc8e1636b9),
//             TO_CUDA_T(0xe5cf6fc1a307c480),
//             TO_CUDA_T(0x6d7a35f8fb44feab),
//             TO_CUDA_T(0x710632349b40d418),
//             TO_CUDA_T(0xbac2b315452fd5cf),
//             TO_CUDA_T(0xfbbd8ba84760a5ba),
//             TO_CUDA_T(0xa54184a74f5e97b3),
//             TO_CUDA_T(0x5005def76ccbf486),
//             TO_CUDA_T(0x6c6c1c22bcfa9a9e),
//             TO_CUDA_T(0x9fc17ec212b41511),
//             TO_CUDA_T(0x0795fa8cddcb9a3d)
//     };
//     typedef blst_256_t<1022, ALT_BN128_r, 0xa8a1622fd074050fu,
//             ALT_BN128_rRR, ALT_BN128_rONE> fr_mont;

//     struct fr_t : public fr_mont
//     {
//         using mem_t = fr_t;

//         inline fr_t() = default;

//         inline fr_t(const fr_mont &a) : fr_mont(a) {}
//     };
// }

# include <blst_t.hpp>

namespace host
{
    static const vec256 ALT_BN128_P = {
            TO_LIMB_T(0x3c208c16d87cfd47), TO_LIMB_T(0x97816a916871ca8d),
            TO_LIMB_T(0xb85045b68181585d), TO_LIMB_T(0x30644e72e131a029)
    };
    static const vec256 ALT_BN128_RR = {    /* (1<<512)%P */
            TO_LIMB_T(0xf32cfc5b538afa89), TO_LIMB_T(0xb5e71911d44501fb),
            TO_LIMB_T(0x47ab1eff0a417ff6), TO_LIMB_T(0x06d89f71cab8351f),
    };
    static const vec256 ALT_BN128_ONE = {   /* (1<<256)%P */
            TO_LIMB_T(0xd35d438dc58f0d9d), TO_LIMB_T(0x0a78eb28f5c70b3d),
            TO_LIMB_T(0x666ea36f7879462c), TO_LIMB_T(0x0e0a77c19a07df2f)
    };
    typedef blst_256_t<254, ALT_BN128_P, 0x87d20782e4866389u,
            ALT_BN128_RR, ALT_BN128_ONE> fp_mont;

    struct fp_t : public fp_mont
    {
        using mem_t = fp_t;

        inline fp_t() = default;

        inline fp_t(const fp_mont &a) : fp_mont(a) {}
    };

    static const vec256 ALT_BN128_r = {
            TO_LIMB_T(0x43e1f593f0000001), TO_LIMB_T(0x2833e84879b97091),
            TO_LIMB_T(0xb85045b68181585d), TO_LIMB_T(0x30644e72e131a029)
    };
    static const vec256 ALT_BN128_rRR = {   /* (1<<512)%r */
            TO_LIMB_T(0x1bb8e645ae216da7), TO_LIMB_T(0x53fe3ab1e35c59e3),
            TO_LIMB_T(0x8c49833d53bb8085), TO_LIMB_T(0x0216d0b17f4e44a5)
    };
    static const vec256 ALT_BN128_rONE = {  /* (1<<256)%r */
            TO_LIMB_T(0xac96341c4ffffffb), TO_LIMB_T(0x36fc76959f60cd29),
            TO_LIMB_T(0x666ea36f7879462e), TO_LIMB_T(0x0e0a77c19a07df2f)
    };
    typedef blst_256_t<254, ALT_BN128_r, 0xc2e1f593efffffffu,
            ALT_BN128_rRR, ALT_BN128_rONE> fr_mont;

    struct fr_t : public fr_mont
    {
        using mem_t = fr_t;

        inline fr_t() = default;

        inline fr_t(const fr_mont &a) : fr_mont(a) {}
    };
}

# if defined(__GNUC__) && !defined(__clang__)
#  pragma GCC diagnostic pop
# endif
# pragma nv_diag_default 20012

#endif
