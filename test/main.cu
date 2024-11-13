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

__global__ void test(device::fp_t &num, bucket_t &point)
{
    using namespace device;
    auto tmp = fp_t::one();
    tmp = tmp + fp_t(5);
//    tmp = tmp ^ 3;

    num += tmp;
    print_num(num);

    affine_t t_point = affine_t (fp_t(1), fp_t(2));
    point.add(t_point);
    point.add(t_point);
}

__global__ void test_div(int n, scalar_t *output)
{
    auto id = (threadIdx.x + blockIdx.x * blockDim.x) / 2;
    if (id >= n)
        return;

    auto tmp = scalar_t((id + 1) * 3) / scalar_t(3);
    if (threadIdx.x % 2 == 0)
        output[id] = tmp;
}

__global__ void test_xyzz_to_affine(int n, affine_t* result)
{
    auto id = (threadIdx.x + blockIdx.x * blockDim.x) / 2;
    if (id >= n)
        return;

    auto base = affine_t(device::fp_t(1), device::fp_t(2));
    auto point = bucket_t(base);
    point = take(point, scalar_t(id + 1));

    auto tmp = affine_t(point);
    if (threadIdx.x % 2 == 0)
        result[id] = tmp;
}

void check_affine(affine_t_host affine, uint64_t x_small)
{
    affine.X.from();
    assert(x_small == affine.X[0]);
    affine.X.to();
}

template<typename HOST_T, typename DEVICE_T>
static HOST_T DtoH(const DEVICE_T &device)
{
    HOST_T result;
    memcpy(&result, &device, sizeof(HOST_T));
    return result;
}

int main()
{
    auto tmp = host::fp_t(10);
//    tmp <<= S;
//    std::cout << tmp << std::endl;

    auto g = affine_t_host(host::fp_t(1), host::fp_t(2));
    auto g_xyz = bucket_t_host(g);
    g_xyz = host::take(g_xyz, host::fr_t(2));
    auto g2 = (affine_t_host)g_xyz;
    check_affine(g2, 0xd3c208c16d87cfd3);

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
//    std::cout << g2.X << std::endl << g2.Y << std::endl;
    host::print_num(g2.X);
    host::print_num(g2.Y);
    host::print_num(DtoH<host::fp_t>(*tmp2));


    // host::xyzz::affine_t operator == test
    {
        affine_t_host point1 = affine_t_host (host::fp_t(1), host::fp_t(2));
        affine_t_host point2 = affine_t_host (host::fp_t(1114514), host::fp_t(2));
        assert(!(point1 == point2));
        assert(point1 == point1);
    }

    // host::xyzz::affine_t operator + / += / * / *= test
    {
        auto g5_xyzz = bucket_t_host(g) * host::fr_t(5);
        check_affine(affine_t_host(g5_xyzz), 0xe849a8a7fa163fa9);
        auto g5 = g * host::fr_t(5);
        assert(g5 == affine_t_host(g5_xyzz));

        auto g10 = g5 + g5;
        check_affine(g10, 0xc6951d924b4045b4);
        check_affine(g10 + g5 * host::fr_t(114514), 0x9bed052c7cf50040); // 572580 * g

        auto g15 = g10;
        g15 += g5;
        check_affine(g15, 0xb05dcd507457f63c);
    }

    // div test
    {
        scalar_t *output;
        int n = 79;
        static constexpr int block_size = 32;
        int block_num = (n * 2 + block_size - 1) / block_size;
        CUDA_OK(cudaMallocManaged(&output, n * sizeof (scalar_t)));
        test_div<<<block_num, block_size>>>(n, output);
        cudaDeviceSynchronize();
        for(auto i = 0; i < n; i++)
            assert(DtoH<host::fr_t>(output[i]) == host::fr_t(i+1));
        // std::cout << std::endl;
        // for (auto i = 0; i < n; i++)
        //     std::cout << DtoH<host::fr_t>(output[i]) << std::endl;
        // std::cout << std::endl;
        CUDA_OK(cudaFree(output));
    }

    // xyzz and affine convention test
    {
        affine_t *output;
        int n = 10;
        static constexpr int block_size = 32;
        int block_num = (n * 2 + block_size - 1) / block_size;
        CUDA_OK(cudaMallocManaged(&output, n * sizeof (affine_t)));
        test_xyzz_to_affine<<<block_num, block_size>>>(n, output);
        cudaDeviceSynchronize();

        for (auto i = 0; i < n; i++)
        {
            auto target = affine_t_host(host::fp_t(1), host::fp_t(2)) * host::fp_t(i + 1);
            assert(DtoH<affine_t_host>(output[i]).X == target.X);
        }
        // for (auto i = 0; i < n; i++)
        // {
        //     printf("point: %dg\n", i+1);
        //     auto tmp = DtoH<affine_t_host>(output[i]);
        //     print_num(tmp.X);
        //     print_num(tmp.Y);
        // }

        CUDA_OK(cudaFree(output));
    }

    return 0;
}