//
// Created by xmd on 24-9-11.
//
#include <cuda.h>
#include <iostream>

#include <ff/alt_bn128.hpp>

#include <ec/jacobian_t.hpp>
#include <ec/xyzz_t.hpp>

typedef jacobian_t<fp_t> point_t;
typedef xyzz_t<fp_t> bucket_t;
typedef bucket_t::affine_t affine_t;
typedef fr_t scalar_t;

//#include <msm/pippenger.cuh>
//
//RustError mult_pippenger(point_t* out, const affine_t points[], size_t npoints,
//                         const scalar_t scalars[])
//{
//    return mult_pippenger<bucket_t>(out, points, npoints, scalars, false);
//}

#include <vect.h>

__global__ void test()
{
    auto tmp = fp_mont::one();
    tmp = tmp + fp_mont(5);
    tmp.from();
    for (auto i = 0; i < tmp.n; i++)
        printf("%u ", tmp[i]);
}

#include "depends/blst/src/blst_t.hpp"
int main()
{
//    auto tmp = fp_mont::one();
//    printf("%lu\n", sizeof(tmp));
//    std::cerr << tmp;

    test<<<1,1>>>();
    cudaDeviceSynchronize();
    return 0;
}