// Copyright Supranational LLC
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#ifndef __SPPARK_FF_ALT2046_HPP__
#define __SPPARK_FF_ALT2046_HPP__

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
    static __device__ __constant__ __align__(16) const uint32_t ALT_BN128_P[64] = {
                TO_CUDA_T(0x07c3927a25caf63d),
                TO_CUDA_T(0x51dc3850fa90bb20),
                TO_CUDA_T(0x5771a94f28dc86fc),
                TO_CUDA_T(0x68cc903d038c49b1),
                TO_CUDA_T(0xa71362e6d2b16df0),
                TO_CUDA_T(0x1264097050e24079),
                TO_CUDA_T(0xd90f8aafd1243fe3),
                TO_CUDA_T(0x3d1fc2195f6050ed),
                TO_CUDA_T(0x4fb0d63ca6013376),
                TO_CUDA_T(0xb036122e701ae06c),
                TO_CUDA_T(0x17b5deb285cf879f),
                TO_CUDA_T(0xd6ea9307cd136d79),
                TO_CUDA_T(0x75060c4f16ec1747),
                TO_CUDA_T(0x34f77a92b6393d89),
                TO_CUDA_T(0x172e9bad2f5d88b4),
                TO_CUDA_T(0x8cbbabfc8dd98632),
                TO_CUDA_T(0x8b727dd1c1656aff),
                TO_CUDA_T(0x256882d6cb7af044),
                TO_CUDA_T(0x9f3c8676e52ef42b),
                TO_CUDA_T(0x8b706eb7bf32b3c0),
                TO_CUDA_T(0x2d56091f29a13ea6),
                TO_CUDA_T(0xd1a0f20a1b3b0895),
                TO_CUDA_T(0x99de162d825c3a57),
                TO_CUDA_T(0x603852b23094b629),
                TO_CUDA_T(0x57b40651078ce4a8),
                TO_CUDA_T(0x0e28bc5abe601f25),
                TO_CUDA_T(0xb00a93acddc3d4ac),
                TO_CUDA_T(0x94b3f5c643f48acc),
                TO_CUDA_T(0x3fe5d9838f483a57),
                TO_CUDA_T(0x164ed89df01684e6),
                TO_CUDA_T(0x3f701ea60f45b27e),
                TO_CUDA_T(0x2dbbf8d40391561c)
    };
    static __device__ __constant__ __align__(16) const uint32_t ALT_BN128_RR[64] = { /* (1<<512)%P */
                TO_CUDA_T(0x190e9df27847d60c),
                TO_CUDA_T(0xacae0900a81bffc2),
                TO_CUDA_T(0x1add11bbe198ffb0),
                TO_CUDA_T(0x140915d967fec275),
                TO_CUDA_T(0x566fbb184cb06ece),
                TO_CUDA_T(0x6c2fdd776e1adf10),
                TO_CUDA_T(0xf1e12ff1e7a3be99),
                TO_CUDA_T(0xeeb0445d6906b0f4),
                TO_CUDA_T(0xd90b748bd32a6cd1),
                TO_CUDA_T(0xa54f85c95c7eea05),
                TO_CUDA_T(0x4ea25d156c45bf06),
                TO_CUDA_T(0x59ed8bec02ab2c1f),
                TO_CUDA_T(0x48a517e0278a52f0),
                TO_CUDA_T(0x604044ab8a6bb2b8),
                TO_CUDA_T(0x2a056490dce96938),
                TO_CUDA_T(0x71b18d0b6ec747f9),
                TO_CUDA_T(0x694cf20db43827f1),
                TO_CUDA_T(0x72b086897f89af57),
                TO_CUDA_T(0x99869f088edead70),
                TO_CUDA_T(0xc254d06b96b275d0),
                TO_CUDA_T(0xcf9e547e0070f9ff),
                TO_CUDA_T(0x87e95244952c3595),
                TO_CUDA_T(0x1662852cd7137c32),
                TO_CUDA_T(0x6ea9efcf648671c8),
                TO_CUDA_T(0x9d212bb5d0cbaba5),
                TO_CUDA_T(0xfa3f4d3e549f0197),
                TO_CUDA_T(0x8450de5a5702a542),
                TO_CUDA_T(0xb6e1f97c55c27fcf),
                TO_CUDA_T(0x50306b8ea358eda4),
                TO_CUDA_T(0x440e83a2c984bebc),
                TO_CUDA_T(0xf5541c22fc315717),
                TO_CUDA_T(0x2da37266a9fb90ea)
    };
    static __device__ __constant__ __align__(16) const uint32_t ALT_BN128_one[64] = { /* (1<<256)%P */
                TO_CUDA_T(0xd92e239d430930cf),
                TO_CUDA_T(0x66b2e66b1b2c585f),
                TO_CUDA_T(0x4ac7b17433b15d12),
                TO_CUDA_T(0xf4012eceee428f89),
                TO_CUDA_T(0xbc9f117de288da4d),
                TO_CUDA_T(0xa40bd0ce6b94bd9f),
                TO_CUDA_T(0xc2b24a90ea4ac090),
                TO_CUDA_T(0xce613581231e6b5a),
                TO_CUDA_T(0x718bd0d0c1f9feb0),
                TO_CUDA_T(0x8ef1a517cf799de2),
                TO_CUDA_T(0x8972a68362f259e1),
                TO_CUDA_T(0xcd6b20d8fe9edca2),
                TO_CUDA_T(0xb6e1c2748d638b98),
                TO_CUDA_T(0xf72a9b2270e1cc50),
                TO_CUDA_T(0x8c16f59e132c547a),
                TO_CUDA_T(0x4055a4113ac06105),
                TO_CUDA_T(0x46c38ae73904e902),
                TO_CUDA_T(0x44f571ce06994ea9),
                TO_CUDA_T(0xe3d15fad86153b28),
                TO_CUDA_T(0x46cdd66944027d3c),
                TO_CUDA_T(0x1d51d2642fd9c6bf),
                TO_CUDA_T(0xe7db45cd77d8d516),
                TO_CUDA_T(0xfea9911c7432dc48),
                TO_CUDA_T(0x1ee662850d18712f),
                TO_CUDA_T(0x497be06ada3f88b6),
                TO_CUDA_T(0xb934523a481f6445),
                TO_CUDA_T(0x8fcb1d9fab2cd8a3),
                TO_CUDA_T(0x187c3320ac394a00),
                TO_CUDA_T(0xc082c06e3396dc4a),
                TO_CUDA_T(0x9075c4ea4f8f6780),
                TO_CUDA_T(0xc2cf66c1b3a38389),
                TO_CUDA_T(0x1b5423dbee295172)
    };
    static __device__ __constant__ __align__(16) const uint32_t ALT_BN128_Px4[64] = { /* left-aligned value of the modulus */
                TO_CUDA_T(0x1f0e49e8972bd8f4),
                TO_CUDA_T(0x4770e143ea42ec80),
                TO_CUDA_T(0x5dc6a53ca3721bf1),
                TO_CUDA_T(0xa33240f40e3126c5),
                TO_CUDA_T(0x9c4d8b9b4ac5b7c1),
                TO_CUDA_T(0x499025c1438901e6),
                TO_CUDA_T(0x643e2abf4490ff8c),
                TO_CUDA_T(0xf47f08657d8143b7),
                TO_CUDA_T(0x3ec358f29804cdd8),
                TO_CUDA_T(0xc0d848b9c06b81b1),
                TO_CUDA_T(0x5ed77aca173e1e7e),
                TO_CUDA_T(0x5baa4c1f344db5e4),
                TO_CUDA_T(0xd418313c5bb05d1f),
                TO_CUDA_T(0xd3ddea4ad8e4f625),
                TO_CUDA_T(0x5cba6eb4bd7622d0),
                TO_CUDA_T(0x32eeaff2376618c8),
                TO_CUDA_T(0x2dc9f7470595abfe),
                TO_CUDA_T(0x95a20b5b2debc112),
                TO_CUDA_T(0x7cf219db94bbd0ac),
                TO_CUDA_T(0x2dc1badefccacf02),
                TO_CUDA_T(0xb558247ca684fa9a),
                TO_CUDA_T(0x4683c8286cec2254),
                TO_CUDA_T(0x677858b60970e95f),
                TO_CUDA_T(0x80e14ac8c252d8a6),
                TO_CUDA_T(0x5ed019441e3392a1),
                TO_CUDA_T(0x38a2f16af9807c95),
                TO_CUDA_T(0xc02a4eb3770f52b0),
                TO_CUDA_T(0x52cfd7190fd22b32),
                TO_CUDA_T(0xff97660e3d20e95e),
                TO_CUDA_T(0x593b6277c05a1398),
                TO_CUDA_T(0xfdc07a983d16c9f8),
                TO_CUDA_T(0xb6efe3500e455870)
    };
    static __device__ __constant__ __align__(16) const uint32_t ALT_BN128_MODn[64] = {
                TO_CUDA_T(0xd18f6ce1d8452eeb),
                TO_CUDA_T(0x5623b514721abe1a),
                TO_CUDA_T(0x9a9c2082417c32ae),
                TO_CUDA_T(0xfe67712e7ac8608f),
                TO_CUDA_T(0x5d6e48d522033308),
                TO_CUDA_T(0x1b0ba11c13ac6c40),
                TO_CUDA_T(0x38a7d3f40e408cc6),
                TO_CUDA_T(0xc95ae74c6809d919),
                TO_CUDA_T(0x51848b8a9486a2c3),
                TO_CUDA_T(0x478a987aefb5553a),
                TO_CUDA_T(0x6391e3aa826f4119),
                TO_CUDA_T(0x34d05de9ed241d28),
                TO_CUDA_T(0xa8069037f8c9ed78),
                TO_CUDA_T(0x1eed0f2e9009e976),
                TO_CUDA_T(0xaa42d5d47dbf3170),
                TO_CUDA_T(0x66db631498762946),
                TO_CUDA_T(0xdadd78b23d82bf47),
                TO_CUDA_T(0x0f1969e3291ce781),
                TO_CUDA_T(0x088e248ca08c535f),
                TO_CUDA_T(0x5c13b48f030fd867),
                TO_CUDA_T(0x070d13f7e6411373),
                TO_CUDA_T(0xee67c620b6746b0d),
                TO_CUDA_T(0x073dc268505f802c),
                TO_CUDA_T(0x8defed22f0176215),
                TO_CUDA_T(0x02a5e97992ea7dd7),
                TO_CUDA_T(0x87bf9aadde9786ec),
                TO_CUDA_T(0x6adee019174e35b0),
                TO_CUDA_T(0x33d9a2a525173498),
                TO_CUDA_T(0x722f4ba08c3a1743),
                TO_CUDA_T(0x556faab40d2d5086),
                TO_CUDA_T(0xcd55e1ef82e0dbd7),
                TO_CUDA_T(0x7ffe215100c2d925)
    };
    static __device__ __constant__ const uint32_t ALT_BN128_M0 = 0xd8452eeb;

    static __device__ __constant__ __align__(16) const uint32_t ALT_BN128_r[64] = {
            TO_CUDA_T(0x6b7b3e104163638f),
            TO_CUDA_T(0x40aebbb5d31d1b6e),
            TO_CUDA_T(0xb6fb9603cc8ce77d),
            TO_CUDA_T(0x684b278f8457a39d),
            TO_CUDA_T(0xa926f44cf2b0e2c1),
            TO_CUDA_T(0x790eec6bb028833b),
            TO_CUDA_T(0x22ba649508c958a4),
            TO_CUDA_T(0x78681d1cfcd85d7b),
            TO_CUDA_T(0x1f399a2c2cd19236),
            TO_CUDA_T(0x158f42ed66dcaee6),
            TO_CUDA_T(0x6016ecd7fcc854a2),
            TO_CUDA_T(0xb66b3a49cd1a2324),
            TO_CUDA_T(0xc474a50393596eaa),
            TO_CUDA_T(0x643bb21329380593),
            TO_CUDA_T(0x3e0c4b39bf076438),
            TO_CUDA_T(0x1a61bddbf589d3e8),
            TO_CUDA_T(0x9ad3fc1cca11656b),
            TO_CUDA_T(0x42c1326cf8ca4470),
            TO_CUDA_T(0x276d6d2fc4402343),
            TO_CUDA_T(0x3b32214698d42047),
            TO_CUDA_T(0x8013877e11c26fd8),
            TO_CUDA_T(0x73c0fe2b966fe3f5),
            TO_CUDA_T(0xd7ad4c1c993ab762),
            TO_CUDA_T(0x6983d98a71578bde),
            TO_CUDA_T(0x744ce7469ed065a5),
            TO_CUDA_T(0xd68e0b9921172dc1),
            TO_CUDA_T(0x9e838ad5e3ce58b3),
            TO_CUDA_T(0x650b17ae5fe1615f),
            TO_CUDA_T(0x1941f64beb9c04ca),
            TO_CUDA_T(0xbbc699edc28f0ffb),
            TO_CUDA_T(0x93b4a613fa9a2d94),
            TO_CUDA_T(0x341e31681c60006b)
    };
    static __device__ __constant__ __align__(16) const uint32_t ALT_BN128_rRR[64] = { /* (1<<512)%P */
            TO_CUDA_T(0x62eb7a2f73ec7909),
            TO_CUDA_T(0x39c2bb48146f68df),
            TO_CUDA_T(0x2a03a8afe8c6beee),
            TO_CUDA_T(0x7b5a7362ad0835bd),
            TO_CUDA_T(0xa99646b21faa1e5c),
            TO_CUDA_T(0x6028284812c5f39a),
            TO_CUDA_T(0xcfdb1f03323305b7),
            TO_CUDA_T(0xfa7ad4f6b80e1401),
            TO_CUDA_T(0xcd3eaaf2e0f55182),
            TO_CUDA_T(0xf788209ff089ec4e),
            TO_CUDA_T(0x2495245e56bf62f8),
            TO_CUDA_T(0x44533c6fd8aefc80),
            TO_CUDA_T(0x56f0227268e90c14),
            TO_CUDA_T(0x450f13aed4d73f28),
            TO_CUDA_T(0x4431a8841453262f),
            TO_CUDA_T(0x509aee5f752236e7),
            TO_CUDA_T(0x6794a2a50030b9c5),
            TO_CUDA_T(0xd61de6c79a77ba4c),
            TO_CUDA_T(0x510c3504da8be9d2),
            TO_CUDA_T(0x213f2118cd428f8b),
            TO_CUDA_T(0xe1788dc786c65a94),
            TO_CUDA_T(0x099ae4dd2e429318),
            TO_CUDA_T(0x45297285926d740f),
            TO_CUDA_T(0x2efccdd071f1a608),
            TO_CUDA_T(0xb416e1606152f22b),
            TO_CUDA_T(0xda2918e495feb836),
            TO_CUDA_T(0xafe65df78744940b),
            TO_CUDA_T(0xb2fd7aa97101c0e0),
            TO_CUDA_T(0x84e944224ad2e704),
            TO_CUDA_T(0x8573244ab258e33c),
            TO_CUDA_T(0x61155167168f5473),
            TO_CUDA_T(0x1f20fcbdd12192e8)
    };
    static __device__ __constant__ __align__(16) const uint32_t ALT_BN128_rone[64] = { /* (1<<256)%P */
            TO_CUDA_T(0x521307befa7271c4),
            TO_CUDA_T(0xfd451128b38b9246),
            TO_CUDA_T(0x2411a7f0cdcc620a),
            TO_CUDA_T(0x5ed361c1eea17189),
            TO_CUDA_T(0x5b642ecc353c74fa),
            TO_CUDA_T(0x1bc44e513f5df311),
            TO_CUDA_T(0x75166dabdcda9d6e),
            TO_CUDA_T(0x1e5f8b8c0c9e8a13),
            TO_CUDA_T(0x8319974f4cb9b726),
            TO_CUDA_T(0xa9c2f44a648d4467),
            TO_CUDA_T(0x7fa44ca00cdead77),
            TO_CUDA_T(0x265316d8cb97736e),
            TO_CUDA_T(0xee2d6bf1b29a4555),
            TO_CUDA_T(0x6f1137b35b1fe9b0),
            TO_CUDA_T(0x07ced31903e26f1e),
            TO_CUDA_T(0x9679089029d8b05f),
            TO_CUDA_T(0x94b00f8cd7ba6a53),
            TO_CUDA_T(0xf4fb364c1cd6ee3d),
            TO_CUDA_T(0x624a4b40eeff72f2),
            TO_CUDA_T(0x13377ae59caf7ee3),
            TO_CUDA_T(0xffb1e207b8f6409f),
            TO_CUDA_T(0x30fc0751a6407029),
            TO_CUDA_T(0xa14acf8d9b152276),
            TO_CUDA_T(0x59f099d63aa1d084),
            TO_CUDA_T(0x2ecc62e584be696a),
            TO_CUDA_T(0xa5c7d19b7ba348fa),
            TO_CUDA_T(0x85f1d4a870c69d30),
            TO_CUDA_T(0x6bd3a146807a7a81),
            TO_CUDA_T(0x9af826d0518fecd6),
            TO_CUDA_T(0x10e59848f5c3c013),
            TO_CUDA_T(0xb12d67b0159749ad),
            TO_CUDA_T(0x2f873a5f8e7ffe51)
    };
    static __device__ __constant__ __align__(
            16) const uint32_t ALT_BN128_rx4[64] = { /* left-aligned value of the modulus */
            TO_CUDA_T(0xadecf841058d8e3c),
            TO_CUDA_T(0x02baeed74c746db9),
            TO_CUDA_T(0xdbee580f32339df5),
            TO_CUDA_T(0xa12c9e3e115e8e76),
            TO_CUDA_T(0xa49bd133cac38b05),
            TO_CUDA_T(0xe43bb1aec0a20cee),
            TO_CUDA_T(0x8ae9925423256291),
            TO_CUDA_T(0xe1a07473f36175ec),
            TO_CUDA_T(0x7ce668b0b34648d9),
            TO_CUDA_T(0x563d0bb59b72bb98),
            TO_CUDA_T(0x805bb35ff3215288),
            TO_CUDA_T(0xd9ace92734688c91),
            TO_CUDA_T(0x11d2940e4d65baaa),
            TO_CUDA_T(0x90eec84ca4e0164f),
            TO_CUDA_T(0xf8312ce6fc1d90e1),
            TO_CUDA_T(0x6986f76fd6274fa0),
            TO_CUDA_T(0x6b4ff073284595ac),
            TO_CUDA_T(0x0b04c9b3e32911c2),
            TO_CUDA_T(0x9db5b4bf11008d0d),
            TO_CUDA_T(0xecc8851a6350811c),
            TO_CUDA_T(0x004e1df84709bf60),
            TO_CUDA_T(0xcf03f8ae59bf8fd6),
            TO_CUDA_T(0x5eb5307264eadd89),
            TO_CUDA_T(0xa60f6629c55e2f7b),
            TO_CUDA_T(0xd1339d1a7b419695),
            TO_CUDA_T(0x5a382e64845cb705),
            TO_CUDA_T(0x7a0e2b578f3962cf),
            TO_CUDA_T(0x942c5eb97f85857e),
            TO_CUDA_T(0x6507d92fae701329),
            TO_CUDA_T(0xef1a67b70a3c3fec),
            TO_CUDA_T(0x4ed2984fea68b652),
            TO_CUDA_T(0xd078c5a0718001ae)
    };
    static __device__ __constant__ __align__(
        16) const uint32_t ALT_BN128_MODN[64] = {
            TO_CUDA_T(0x43c1d177bbc3a491),
            TO_CUDA_T(0x859fd70cd39eba99),
            TO_CUDA_T(0xb8d27bca6e26f5f7),
            TO_CUDA_T(0x162154df45a0555c),
            TO_CUDA_T(0x113929863eefe7bc),
            TO_CUDA_T(0xe5f0a29d82644bc7),
            TO_CUDA_T(0x52987bcce3fb9ae0),
            TO_CUDA_T(0x955f342bac03abcb),
            TO_CUDA_T(0x9ad44a743bd5359e),
            TO_CUDA_T(0xa766d7bef9f2e5a3),
            TO_CUDA_T(0x0ce477943c691300),
            TO_CUDA_T(0x632739cfba6c8ed7),
            TO_CUDA_T(0x621b9e68d155f41c),
            TO_CUDA_T(0xf19ddaec45284f64),
            TO_CUDA_T(0xf4b6b0b5810c3e63),
            TO_CUDA_T(0x237b90cb48807913),
            TO_CUDA_T(0xebb8e8893922657b),
            TO_CUDA_T(0xc1c9e0c82cad1992),
            TO_CUDA_T(0x914ded65b1cde372),
            TO_CUDA_T(0xbda79286bbaf8c27),
            TO_CUDA_T(0x0021562e38c7776d),
            TO_CUDA_T(0x8e1415c2cdeb8fce),
            TO_CUDA_T(0x90a13c791bc16328),
            TO_CUDA_T(0x88682096cd4ec54a),
            TO_CUDA_T(0x2016dd5068602a93),
            TO_CUDA_T(0x701e1b559c9a2ee7),
            TO_CUDA_T(0x3fef6a1f6ad77c89),
            TO_CUDA_T(0x5b65742a28763bcc),
            TO_CUDA_T(0x3244fa1dca17e2ca),
            TO_CUDA_T(0x1e4ca2a970c2dcc4),
            TO_CUDA_T(0x0e0972675a626564),
            TO_CUDA_T(0x031c2d1c7f2a3f2f)
    };

    static __device__ __constant__ const uint32_t ALT_BN128_m0 = 0xbbc3a491;

    // device-side field types
    typedef mont_t<2046, device::ALT_BN128_P, 
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

    typedef mont_t<2046, device::ALT_BN128_r,            // 模数 N = r
                device::ALT_BN128_m0,                   // m0 �? N'
                device::ALT_BN128_rRR,                  // R^2 mod r
                device::ALT_BN128_rone,                 // 1 �? Montgomery 表示 = R mod r
                device::ALT_BN128_rx4,                  // 4 �? Montgomery 表示 = 4R mod r
                device::ALT_BN128_MODN> fr_mont;

    struct fr_t : public fr_mont
    {
        using mem_t = fr_t;

        __host__ __device__ __forceinline__ fr_t() {}

        __device__ __forceinline__ fr_t(const uint32_t &a) : fr_mont(a){}

        __device__ __forceinline__ fr_t(const fr_mont &a) : fr_mont(a) {}
    };

    static constexpr int LAMBDA = 2046;
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
