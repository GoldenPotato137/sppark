//
// Created by xmd on 24-9-11.
//
#include <cuda.h>
#include <iostream>

#include <ff/alt_bn128.hpp>

#include <ec/jacobian_t.hpp>
#include <ec/xyzz_t.hpp>
#include "util/exception.cuh"

typedef jacobian_t<device::fp_t> point_t;
typedef device::xyzz_t<device::fp_t> bucket_t;
typedef host::xyzz_t<host::fp_t> bucket_t_host;
typedef bucket_t::affine_t affine_t;
typedef bucket_t_host::affine_t affine_t_host;
typedef device::fr_t scalar_t;

//#include <msm/pippenger.cuh>
//
//RustError mult_pippenger(point_t* out, const affine_t points[], size_t npoints,
//                         const scalar_t scalars[])
//{
//    return mult_pippenger<bucket_t>(out, points, npoints, scalars, false);
//}

__global__ void test(device::fp_t &num, bucket_t &point)
{
    using namespace device;
    auto tmp = fp_t::one();
    tmp = tmp + fp_t(5);
//    tmp = tmp ^ 3;
    print_num(tmp);

    num += tmp;
    print_num(num);
    num.from();

//    affine_t t_point = affine_t (fp_t(1), fp_t(2));
//    point.uadd(t_point);
//    point.uadd(t_point);
    point = take(point, tmp);
}

const int S = 10;

int main()
{
    auto tmp = host::fr_t(10);
    tmp <<= S;
    affine_t_host tt;
//    std::cout << tmp << std::endl;

    auto g = affine_t_host(host::fp_t(1), host::fp_t(2));
//    std::cout << g.X << std::endl << g.Y << std::endl;
    auto g_xyz = bucket_t_host(g);
//    g_xyz = take(g_xyz, tmp);
//    std:: cout << g_xyz.X << std::endl << g_xyz.Y << std::endl << g_xyz.ZZ << std::endl << g_xyz.ZZZ << std::endl;
//    g_xyz.add(g);
    auto g2 = (affine_t_host)g_xyz;
//    std::cout << g2.X << std::endl << g2.Y << std::endl;

//    printf("%lu %lu\n", sizeof(device::fp_t), sizeof(host::fp_t));
    device::fp_t* tmp2;
    bucket_t* t_point;
    CUDA_OK(cudaMallocManaged(&tmp2, sizeof (device::fp_t)));
    CUDA_OK(cudaMallocManaged(&t_point, sizeof (bucket_t)));
    CUDA_OK(cudaMemcpy(tmp2, &tmp, sizeof(tmp), cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(t_point, &g_xyz, sizeof (g_xyz), cudaMemcpyHostToDevice));
    test<<<1,1>>>(*tmp2, *t_point);
    cudaDeviceSynchronize();

    memcpy(&g_xyz, t_point, sizeof (g_xyz));
    g2 = (affine_t_host)g_xyz;
    std::cout << g2.X << std::endl << g2.Y << std::endl;

    for(auto i = 0; i < tmp2->n; i++)
        printf("%u ", (*tmp2)[i]);

    return 0;
}