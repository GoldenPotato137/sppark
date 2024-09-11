//
// Created by xmd on 24-9-11.
//
#include <msm/pippenger.hpp>
#include <cuda.h>

#include <ff/bls12-381.hpp>

#include <ec/jacobian_t.hpp>
#include <ec/xyzz_t.hpp>

typedef jacobian_t<fp_t> point_t;
typedef xyzz_t<fp_t> bucket_t;
typedef bucket_t::affine_t affine_t;
typedef fr_t scalar_t;

#include <msm/pippenger.cuh>

RustError mult_pippenger(point_t* out, const affine_t points[], size_t npoints,
                         const scalar_t scalars[])
{
    return mult_pippenger<bucket_t>(out, points, npoints, scalars, false);
}


int main()
{
    return 0;
}