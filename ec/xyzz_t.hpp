// Copyright Supranational LLC
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#ifndef __SPPARK_EC_XYZZ_T_HPP__
#define __SPPARK_EC_XYZZ_T_HPP__

#ifndef __CUDACC__
# undef  __host__
# define __host__
# undef  __device__
# define __device__
# undef  __noinline__
# define __noinline__
#endif
# pragma nv_diag_suppress 284   // NULL reference is not allowed

#include "gmp.h"
#include "../ff/mont_t.cuh"
#include "../../cgbn/cgbn.h"
#define TPI 1
typedef cgbn_context_t<TPI> context_t;
typedef cgbn_env_t<context_t, 256> env256_t;

namespace device
{

    template<class field_t, class field_h = typename field_t::mem_t,
            const field_h *a4 = nullptr>
    class xyzz_t
    {
    public:
        field_t X, Y, ZZZ, ZZ;
        static const unsigned int degree = field_t::degree;

        class mem_t
        {
            friend class xyzz_t;

            field_h X, Y, ZZZ, ZZ;

        public:
            inline __device__ operator xyzz_t() const
            {
                xyzz_t p;
                p.X = X;
                p.Y = Y;
                p.ZZZ = ZZZ;
                p.ZZ = ZZ;
                return p;
            }

            inline __device__ mem_t &operator=(const xyzz_t &p)
            {
                X = p.X;
                Y = p.Y;
                ZZZ = p.ZZZ;
                ZZ = p.ZZ;
                return *this;
            }

            inline __device__ void inf()
            {
                ZZZ.zero();
                ZZ.zero();
            }
        };

        inline __device__ xyzz_t &operator=(const mem_t &p)
        {
            X = p.X;
            Y = p.Y;
            ZZZ = p.ZZZ;
            ZZ = p.ZZ;
            return *this;
        }

        __device__ bool is_equal(const xyzz_t &p2)
        {
#ifdef __CUDA_ARCH__
            xyzz_t p31 = *this;
#else
            xyzz_t &p31 = *this;
#endif
            if (p2.is_inf())
            {
                return (bool)is_inf();
            }
            else if (p31.is_inf())
            {
                return (bool)p2.is_inf();
            }

            field_t U, S, P, R;

            U = p31.X * p2.ZZ;  /* U1 = X1*ZZ2 */
            S = p31.Y * p2.ZZZ; /* S1 = Y1*ZZZ2 */
            P = p2.X * p31.ZZ;  /* U2 = X2*ZZ1 */
            R = p2.Y * p31.ZZZ; /* S2 = Y2*ZZZ1 */
            P -= U;             /* P = U2-U1 */
            R -= S;             /* R = S2-S1 */

            return (bool)(P.is_zero() & R.is_zero());
        }

        class affine_t
        {
            friend class xyzz_t;

        public:
            field_t X, Y;

            __device__ affine_t(const field_t &x, const field_t &y) : X(x), Y(y) {}

            inline __host__ __device__ affine_t() {}

            inline __device__ affine_t(const xyzz_t &a)
            {
                // printf("\n123 %u\n", threadIdx.x);
                // print_num(a.X);
                // print_num(a.Y);
                // print_num(a.ZZ);
                // print_num(a.ZZZ);
                Y = 1 / a.ZZZ;
                X = Y * a.ZZ;   // 1/Z
                X = X ^ 2;        // 1/Z^2
                X *= a.X;       // X/Z^2
                Y *= a.Y;       // Y/Z^3
            }

            inline __device__ bool is_inf() const { return (bool) (X.is_zero(Y)); }

            inline  __device__ operator xyzz_t() const
            {
                xyzz_t p;
                p.X = X;
                p.Y = Y;
                p.ZZZ = p.ZZ = field_t::one(is_inf());
                return p;
            }

            class mem_t
            {
                field_h X, Y;

            public:
                inline __device__ operator affine_t() const
                {
                    affine_t p;
                    p.X = X;
                    p.Y = Y;
                    return p;
                }
            };
        };

        class affine_inf_t
        {
            field_t X, Y;
            bool inf;

            inline __device__ bool is_inf() const { return inf; }

        public:
            inline __device__ operator affine_t() const
            {
                bool inf = is_inf();
                affine_t p;
                p.X = czero(X, inf);
                p.Y = czero(Y, inf);
                return p;
            }

            class mem_t
            {
                field_h X, Y;
                int inf[sizeof(field_t) % 16 ? 2 : 4];

                inline __device__ bool is_inf() const { return inf[0] & 1 != 0; }

            public:
                inline __device__ operator affine_t() const
                {
                    bool inf = is_inf();
                    affine_t p;
                    p.X = czero((field_t) X, inf);
                    p.Y = czero((field_t) Y, inf);
                    return p;
                }

                inline __device__ operator affine_inf_t() const
                {
                    bool inf = is_inf();
                    affine_inf_t p;
                    p.X = czero((field_t) X, inf);
                    p.Y = czero((field_t) Y, inf);
                    p.inf = inf;
                    return p;
                }
            };
        };

        template<class affine_t>
        inline  __device__ xyzz_t &operator=(const affine_t &a)
        {
            X = a.X;
            Y = a.Y;
            ZZZ = ZZ = field_t::one(a.is_inf());
            return *this;
        }

        inline __device__ operator affine_t() const
        {
            return affine_t(*this);
        }

        inline __device__ bool is_inf() const { return (bool) (ZZZ.is_zero(ZZ)); }

        inline  __device__ void inf()
        {
            ZZZ.zero();
            ZZ.zero();
        }

        inline  __device__ void cneg(bool neg) { ZZZ.cneg(neg); }

        static inline __device__ void prefetch(const xyzz_t *p_)
        {
            const unsigned char *p = (const unsigned char *) p_;
            for (size_t i = 0; i < sizeof(*p_); i += 128)
                asm("prefetch.global.L2 [%0];"::"l"(p + i));
        }


        __device__ void cadd_a(const affine_t &p2)
        {
            field_t P, R;

            R = p2.Y * ZZZ; /* S2 = Y2*ZZZ1 */
            R -= Y;         /* R = S2-Y1 */
            P = p2.X * ZZ;  /* U2 = X2*ZZ1 */
            P -= X;         /* P = U2-X1 */

            field_t PP; /* add |p2| to |p1| */

            PP = P ^ 2; /* PP = P^2 */
#define PPP P
            PPP = P * PP; /* PPP = P*PP */
            ZZ *= PP;     /* ZZ3 = ZZ1*PP */
            ZZZ *= PPP;   /* ZZZ3 = ZZZ1*PPP */
#define Q PP
            Q = PP * X; /* Q = X1*PP */
            X = R ^ 2;  /* R^2 */
            X -= PPP;   /* R^2-PPP */
            X -= Q;
            X -= Q; /* X3 = R^2-PPP-2*Q */
            Q -= X;
            Q *= R;    /* R*(Q-X3) */
            Y *= PPP;  /* Y1*PPP */
            Y = Q - Y; /* Y3 = R*(Q-X3)-Y1*PPP */
        }

        template <typename env256_t, typename cgbn_t>
        __device__ void cgbn_mont(env256_t bn256_env, const field_t *aa, const field_t *bb, field_t *cc)
        {
            cgbn_mem_t<256> ins_a;
            cgbn_mem_t<256> ins_b;
            cgbn_mem_t<256> ins_r;
            cgbn_mem_t<256> ins_m;

            const uint32_t *MOD = device::fp_t::get_MOD();
            const uint32_t M0 = device::fp_t::get_M0();
            cgbn_t a, b, r, m; // three 1024-bit values (spread across a warp)

            for (int i = 0; i < 8; i++)
            {
                ins_a._limbs[i] = (*aa)[i];
            }

            for (int i = 0; i < 8; i++)
            {
                ins_b._limbs[i] = (*bb)[i];
            }
            for (int i = 0; i < 8; i++)
            {
                ins_m._limbs[i] = MOD[i];
            }

            // load a and b from global memory
            bn256_env.load(a, &ins_a);
            bn256_env.load(b, &ins_b);
            bn256_env.load(m, &ins_m);

            cgbn_mont_mul(bn256_env, r, a, b, m, M0);

            bn256_env.store(&ins_r, r);

            for (int i = 0; i < 8; i++)
            {
                cc[i] = ins_r._limbs[i];
            }
        }

        __device__ void cgbn_add_a(const affine_t &p2)
        {
            context_t bn_context;
            env256_t bn256_env(bn_context);

            field_t P, R;

            // R = p2.Y * ZZZ; /* S2 = Y2*ZZZ1 */
            cgbn_mont<env256_t, env256_t::cgbn_t>(bn256_env, &ZZZ, &p2.Y, &R);
            R -= Y;        /* R = S2-Y1 */
            P = p2.X * ZZ; /* U2 = X2*ZZ1 */
            // cgbn_mont<env256_t,env256_t::cgbn_t>(bn256_env,bn256_env,&ZZ, &p2.X,  &P);
            P -= X; /* P = U2-X1 */

            field_t PP; /* add |p2| to |p1| */

            // PP = P ^ 2; /* PP = P^2 */
            cgbn_mont<env256_t, env256_t::cgbn_t>(bn256_env, &P, &P, &PP);

#define PPP P
            // PPP = P * PP; /* PPP = P*PP */
            cgbn_mont<env256_t, env256_t::cgbn_t>(bn256_env, &P, &PP, &PPP);
            ZZ *= PP; /* ZZ3 = ZZ1*PP */
            // cgbn_mont<env256_t, env256_t::cgbn_t>(bn256_env, &ZZ, &PP, &ZZ);
            ZZZ *= PPP; /* ZZZ3 = ZZZ1*PPP */
            // cgbn_mont<env256_t, env256_t::cgbn_t>(bn256_env, &ZZZ, &PPP, &ZZZ);
#define Q PP
            Q = PP * X; /* Q = X1*PP */
            // cgbn_mont<env256_t,env256_t::cgbn_t>(bn256_env,&PP, &X, &Q);
            X = R ^ 2; /* R^2 */
            // cgbn_mont<env256_t,env256_t::cgbn_t>(bn256_env,&R, &R, &X);
            X -= PPP; /* R^2-PPP */
            X -= Q;
            X -= Q; /* X3 = R^2-PPP-2*Q */
            Q -= X;
            // Q *= R;    /* R*(Q-X3) */
            // cgbn_mont<env256_t,env256_t::cgbn_t>(bn256_env,&R, &Q, &Q);
            Y *= PPP; /* Y1*PPP */
            // cgbn_mont<env256_t,env256_t::cgbn_t>(bn256_env,&Y, &PPP, &Y);
            Y = Q - Y; /* Y3 = R*(Q-X3)-Y1*PPP */
        }

        __device__ void mmadd_a(const affine_t &p2)
        {
            if (p2.is_inf())
            {
                return;
            }
            if (is_inf())
            {
                *this = p2;
                return;
            }
            field_t P, R;
            uint32_t tid = threadIdx.x;
            uint32_t block_size = 256;
            __shared__ field_t mult_a[256];
            __shared__ field_t mult_b[256];
            __shared__ field_t mult_results[256];
            int mult_count = 0;
            mult_a[tid + mult_count] = p2.Y; // p2.Y * ZZZ
            mult_b[tid + mult_count] = ZZZ;
            mult_count++;
            // __syncthreads();
            montgomery_mult_coop_kernel(mult_a, mult_b, mult_results, mult_count * block_size);
            // __syncthreads();
            field_t temp1 = mult_results[tid]; // p2.Y * ZZZ
            R = temp1 - Y;                     // delta_y = p2.Y * ZZZ - Y
            // if(tid == 0 && blockIdx.x == 0){
            //     field_t tttt = p2.Y * ZZZ; /* S2 = Y2*ZZZ1 */
            //     printf("======\n");
            //     print_num(tttt);
            //     printf("======\n");
            //     print_num(temp1);
            // }
            // mult_count = 0;
            // mult_a[tid+ mult_count] = p2.X; // p2.X * ZZ
            // mult_b[tid + mult_count] = ZZ;
            // mult_count++;
            // __syncthreads();
            // montgomery_mult_coop_kernel(mult_a, mult_b, mult_results, mult_count * block_size);
            // __syncthreads();
            // field_t temp2 = mult_results[tid]; // p2.X * ZZ
            // P = temp2 - X;                   // delta_x = p2.X * ZZ - X

            // R = p2.Y * ZZZ; /* S2 = Y2*ZZZ1 */
            // R -= Y;         /* R = S2-Y1 */
            P = p2.X * ZZ; /* U2 = X2*ZZ1 */
            P -= X;        /* P = U2-X1 */

            if (!P.is_zero())
            {               /* X1!=X2 */
                field_t PP; /* add |p2| to |p1| */

                PP = P ^ 2; /* PP = P^2 */
#define PPP P
                PPP = P * PP; /* PPP = P*PP */
                ZZ *= PP;     /* ZZ3 = ZZ1*PP */
                ZZZ *= PPP;   /* ZZZ3 = ZZZ1*PPP */
#define Q PP
                Q = PP * X; /* Q = X1*PP */
                X = R ^ 2;  /* R^2 */
                X -= PPP;   /* R^2-PPP */
                X -= Q;
                X -= Q; /* X3 = R^2-PPP-2*Q */
                Q -= X;
                Q *= R;    /* R*(Q-X3) */
                Y *= PPP;  /* Y1*PPP */
                Y = Q - Y; /* Y3 = R*(Q-X3)-Y1*PPP */
#undef Q
#undef PPP
            }
            else if (R.is_zero())
            {              /* X1==X2 && Y1==Y2 */
                field_t M; /* double |p2| */

#define U P
                U = p2.Y + p2.Y; /* U = 2*Y1 */
                ZZ = U ^ 2;      /* [ZZ3 =] V = U^2 */
                ZZZ = ZZ * U;    /* [ZZZ3 =] W = U*V */
#define S R
                S = p2.X * ZZ; /* S = X1*V */
                M = p2.X ^ 2;
                M = M + M + M; /* M = 3*X1^2[+a] */
                X = M ^ 2;
                X -= S;
                X -= S;         /* X3 = M^2-2*S */
                Y = ZZZ * p2.Y; /* W*Y1 */
                S -= X;
                S *= M;    /* M*(S-X3) */
                Y = S - Y; /* Y3 = M*(S-X3)-W*Y1 */
#undef S
#undef U
            }
            else
            {          /* X1==X2 && Y1==-Y2 */
                inf(); /* set |p3| to infinity */
            }
        }

        __device__ void mmadd(const xyzz_t &p2)
        {
            if (p2.is_inf())
            {
                return;
            }
            else if (is_inf())
            {
                *this = p2;
                return;
            }

            xyzz_t p31 = *this;
            field_t U, S, P, R;

            uint32_t tid = threadIdx.x;
            uint32_t block_size = 256;
            __shared__ field_t mult_a[256];
            __shared__ field_t mult_b[256];
            __shared__ field_t mult_results[256];
            int mult_count = 0;
            mult_a[tid + mult_count] = p31.X;
            mult_b[tid + mult_count] = p2.ZZ;
            mult_count++;
            montgomery_mult_coop_kernel(mult_a, mult_b, mult_results, mult_count * block_size);
            U = mult_results[tid];

            mult_count = 0;
            mult_a[tid + mult_count] = p31.Y; // p2.X * ZZ
            mult_b[tid + mult_count] = p2.ZZZ;
            mult_count++;
            montgomery_mult_coop_kernel(mult_a, mult_b, mult_results, mult_count * block_size);
            S = mult_results[tid]; // p2.X * ZZ

            mult_count = 0;
            mult_a[tid + mult_count] = p2.X; // p2.X * ZZ
            mult_b[tid + mult_count] = p31.ZZ;
            mult_count++;
            montgomery_mult_coop_kernel(mult_a, mult_b, mult_results, mult_count * block_size);
            P = mult_results[tid]; // p2.X * ZZ

            mult_count = 0;
            mult_a[tid + mult_count] = p2.Y; // p2.X * ZZ
            mult_b[tid + mult_count] = p31.ZZZ;
            mult_count++;
            montgomery_mult_coop_kernel(mult_a, mult_b, mult_results, mult_count * block_size);
            R = mult_results[tid]; // p2.X * ZZ

            // U = p31.X * p2.ZZ;          /* U1 = X1*ZZ2 */
            // S = p31.Y * p2.ZZZ;         /* S1 = Y1*ZZZ2 */
            // P = p2.X * p31.ZZ;          /* U2 = X2*ZZ1 */
            // R = p2.Y * p31.ZZZ;         /* S2 = Y2*ZZZ1 */
            P -= U; /* P = U2-U1 */
            R -= S; /* R = S2-S1 */

            if (!P.is_zero())
            {               /* X1!=X2 */
                field_t PP; /* add |p1| and |p2| */

                PP = P ^ 2; /* PP = P^2 */
#define PPP P
                PPP = P * PP;   /* PPP = P*PP */
                p31.ZZ *= PP;   /* ZZ3 = ZZ1*ZZ2*PP */
                p31.ZZZ *= PPP; /* ZZZ3 = ZZZ1*ZZZ2*PPP */
#define Q PP
                Q = U * PP;    /* Q = U1*PP */
                p31.X = R ^ 2; /* R^2 */
                p31.X -= PPP;  /* R^2-PPP */
                p31.X -= Q;
                p31.X -= Q; /* X3 = R^2-PPP-2*Q */
                Q -= p31.X;
                Q *= R;            /* R*(Q-X3) */
                p31.Y = S * PPP;   /* S1*PPP */
                p31.Y = Q - p31.Y; /* Y3 = R*(Q-X3)-S1*PPP */
                p31.ZZ *= p2.ZZ;   /* ZZ1*ZZ2 */
                p31.ZZZ *= p2.ZZZ; /* ZZZ1*ZZZ2 */
#undef PPP
#undef Q
            }
            else if (R.is_zero())
            {              /* X1==X2 && Y1==Y2 */
                field_t M; /* double |p1| */

                U = p31.Y + p31.Y; /* U = 2*Y1 */
#define V P
#define W R
                V = U ^ 2;     /* V = U^2 */
                W = U * V;     /* W = U*V */
                S = p31.X * V; /* S = X1*V */
                M = p31.X ^ 2;
                M = M + M + M; /* M = 3*X1^2[+a*ZZ1^2] */
                if (a4 != nullptr)
                {
                    U = *a4;
                    U *= p31.ZZ ^ 2;
                    M += U;
                }
                p31.X = M ^ 2;
                p31.X -= S;
                p31.X -= S; /* X3 = M^2-2*S */
                p31.Y *= W; /* W*Y1 */
                S -= p31.X;
                S *= M;            /* M*(S-X3) */
                p31.Y = S - p31.Y; /* Y3 = M*(S-X3)-W*Y1 */
                p31.ZZ *= V;       /* ZZ3 = V*ZZ1 */
                p31.ZZZ *= W;      /* ZZZ3 = W*ZZZ1 */
#undef V
#undef W
            }
            else
            {              /* X1==X2 && Y1==-Y2 */
                p31.inf(); /* set |p3| to infinity */
            }
            *this = p31;
        }

        

        __device__ void add_a(const affine_t &p2)
        {
            if (p2.is_inf())
            {
                return;
            }
            else if (is_inf())
            {
                *this = p2;
            }
            else
            {
                field_t P, R;

                R = p2.Y * ZZZ; /* S2 = Y2*ZZZ1 */
                R -= Y;         /* R = S2-Y1 */
                P = p2.X * ZZ;  /* U2 = X2*ZZ1 */
                P -= X;         /* P = U2-X1 */

                if (!P.is_zero())
                {               /* X1!=X2 */
                    field_t PP; /* add |p2| to |p1| */

                    PP = P ^ 2; /* PP = P^2 */
#define PPP P
                    PPP = P * PP; /* PPP = P*PP */
                    ZZ *= PP;     /* ZZ3 = ZZ1*PP */
                    ZZZ *= PPP;   /* ZZZ3 = ZZZ1*PPP */
#define Q PP
                    Q = PP * X; /* Q = X1*PP */
                    X = R ^ 2;  /* R^2 */
                    X -= PPP;   /* R^2-PPP */
                    X -= Q;
                    X -= Q; /* X3 = R^2-PPP-2*Q */
                    Q -= X;
                    Q *= R;    /* R*(Q-X3) */
                    Y *= PPP;  /* Y1*PPP */
                    Y = Q - Y; /* Y3 = R*(Q-X3)-Y1*PPP */
#undef Q
#undef PPP
                }
                else if (R.is_zero())
                {              /* X1==X2 && Y1==Y2 */
                    field_t M; /* double |p2| */

#define U P
                    U = p2.Y + p2.Y; /* U = 2*Y1 */
                    ZZ = U ^ 2;      /* [ZZ3 =] V = U^2 */
                    ZZZ = ZZ * U;    /* [ZZZ3 =] W = U*V */
#define S R
                    S = p2.X * ZZ; /* S = X1*V */
                    M = p2.X ^ 2;
                    M = M + M + M; /* M = 3*X1^2[+a] */
                    X = M ^ 2;
                    X -= S;
                    X -= S;         /* X3 = M^2-2*S */
                    Y = ZZZ * p2.Y; /* W*Y1 */
                    S -= X;
                    S *= M;    /* M*(S-X3) */
                    Y = S - Y; /* Y3 = M*(S-X3)-W*Y1 */
#undef S
#undef U
                }
                else
                {          /* X1==X2 && Y1==-Y2 */
                    inf(); /* set |p3| to infinity */
                }
            }
        }


        __device__ void add(const xyzz_t &p2)
        {
            if (p2.is_inf())
            {
                return;
            } else if (is_inf())
            {
                *this = p2;
                return;
            }

            xyzz_t p31 = *this;
            field_t U, S, P, R;

            U = p31.X * p2.ZZ;          /* U1 = X1*ZZ2 */
            S = p31.Y * p2.ZZZ;         /* S1 = Y1*ZZZ2 */
            P = p2.X * p31.ZZ;          /* U2 = X2*ZZ1 */
            R = p2.Y * p31.ZZZ;         /* S2 = Y2*ZZZ1 */
            P -= U;                     /* P = U2-U1 */
            R -= S;                     /* R = S2-S1 */

            if (!P.is_zero())
            {         /* X1!=X2 */
                field_t PP;             /* add |p1| and |p2| */

                PP = P ^ 2;               /* PP = P^2 */
#define PPP P
                PPP = P * PP;           /* PPP = P*PP */
                p31.ZZ *= PP;           /* ZZ3 = ZZ1*ZZ2*PP */
                p31.ZZZ *= PPP;         /* ZZZ3 = ZZZ1*ZZZ2*PPP */
#define Q PP
                Q = U * PP;             /* Q = U1*PP */
                p31.X = R ^ 2;            /* R^2 */
                p31.X -= PPP;           /* R^2-PPP */
                p31.X -= Q;
                p31.X -= Q;             /* X3 = R^2-PPP-2*Q */
                Q -= p31.X;
                Q *= R;                 /* R*(Q-X3) */
                p31.Y = S * PPP;        /* S1*PPP */
                p31.Y = Q - p31.Y;      /* Y3 = R*(Q-X3)-S1*PPP */
                p31.ZZ *= p2.ZZ;        /* ZZ1*ZZ2 */
                p31.ZZZ *= p2.ZZZ;      /* ZZZ1*ZZZ2 */
#undef PPP
#undef Q
            } else if (R.is_zero())
            {   /* X1==X2 && Y1==Y2 */
                field_t M;              /* double |p1| */

                U = p31.Y + p31.Y;      /* U = 2*Y1 */
#define V P
#define W R
                V = U ^ 2;                /* V = U^2 */
                W = U * V;              /* W = U*V */
                S = p31.X * V;          /* S = X1*V */
                M = p31.X ^ 2;
                M = M + M + M;          /* M = 3*X1^2[+a*ZZ1^2] */
                if (a4 != nullptr)
                {
                    U = *a4;
                    U *= p31.ZZ ^ 2;
                    M += U;
                }
                p31.X = M ^ 2;
                p31.X -= S;
                p31.X -= S;             /* X3 = M^2-2*S */
                p31.Y *= W;             /* W*Y1 */
                S -= p31.X;
                S *= M;                 /* M*(S-X3) */
                p31.Y = S - p31.Y;      /* Y3 = M*(S-X3)-W*Y1 */
                p31.ZZ *= V;            /* ZZ3 = V*ZZ1 */
                p31.ZZZ *= W;           /* ZZZ3 = W*ZZZ1 */
#undef V
#undef W
            } else
            {                    /* X1==X2 && Y1==-Y2 */\
            p31.inf();              /* set |p3| to infinity */\

            }
            *this = p31;
        }

        __device__ void uadd(const xyzz_t &p2)
        {
            xyzz_t p31 = *this;

            if (p2.is_inf())
            {
                return;
            } else if (p31.is_inf())
            {
                *this = p2;
                return;
            }

            field_t A, B, U, S, P, R, PP;
            int pc = -1;
            bool done = false, dbl = false, inf = false;

            A = p31.Y;
            B = p2.ZZZ;
#pragma unroll 1
            do
            {
                A = A * B;
                switch (++pc)
                {
                    case 0:
                        S = A;                  /* S1 = Y1*ZZZ2 */
                        A = p2.Y;
                        B = p31.ZZZ;
                        break;
                    case 1:                     /* S2 = Y2*ZZZ1 */
                        R = A - S;              /* R = S2-S1 */
                        A = p31.X;
                        B = p2.ZZ;
                        break;
                    case 2:
                        U = A;                  /* U1 = X1*ZZ2 */
                        A = p2.X;
                        B = p31.ZZ;
                        break;
                    case 3:                     /* U2 = X2*ZZ1 */
                        A = A - U;              /* P = U2-U1 */
                        inf = A.is_zero();      /* X1==X2, not add |p1| and |p2| */
                        dbl = R.is_zero() & inf;
                        if (dbl)
                        {              /* X1==X2 && Y1==Y2, double |p2| */
                            if (a4 != nullptr)
                            {
                                A = p2.ZZ;
                                pc = 16;
                            } else
                            {
                                A = p2.Y << 1;    /* U = 2*Y1 */
                            }
                            inf = false;        /* don't set |p3| to infinity */
                        }
                        B = A;
                        break;
                    case 4:
                        PP = A;                 /* PP = P^2 */
                        break;
                    case 5:
#define PPP P
                        PPP = A;                /* PPP = P*PP */
                        B = field_t::csel(field_t::one(), p31.ZZZ, dbl);
                        break;
                    case 6:                     /* ZZZ1*PPP */
                        B = czero(p2.ZZZ, inf);
                        break;
                    case 7:
                        p31.ZZZ = A;            /* ZZZ3 = ZZZ1*ZZZ2*PPP */
                        A = field_t::csel(field_t::one(), p31.ZZ, dbl);
                        B = czero(p2.ZZ, inf);
                        break;
                    case 8:                     /* ZZ1*ZZ2 */
                        B = PP;
                        break;
                    case 9:
                        p31.ZZ = A;             /* ZZ3 = ZZ1*ZZ2*PP */
                        A = field_t::csel(p2.X, U, dbl);
                        break;
                    case 10:
#define Q PP
                        Q = A;                  /* Q = U1*PP */
                        A = field_t::csel(p2.Y, S, dbl);
                        B = PPP;
                        break;
                    case 11:
                        p31.Y = A;              /* S1*PPP */
                        A = R;
                        B = R;
                        break;
                    case 12:                    /* R^2 */
                        p31.X = A - PPP;        /* R^2-PPP */
                        p31.X -= Q;
                        p31.X -= Q;             /* X3 = R^2-PPP-2*Q */
                        A = Q - p31.X;
                        break;
                    case 13:                    /* R*(Q-X3) */
                        /* p31.Y = A - p31.Y; */    /* Y3 = R*(Q-X3)-S1*PPP */
                        if (dbl)
                        {
                            A = p2.X;
                            B = p2.X;
                        } else
                        {
                            done = true;
                        }
                        break;
#undef PPP
#undef Q
                        /*** double |p2|, V*X1, W*Y1, ZZZ3 and ZZ3 are already calculated ***/
#define S PP
                    case 14:
                        A = A + A + A;          /* M = 3*X1^2[+a*ZZ1^2] */
                        if (a4 != nullptr)
                            A += U;
                        B = A;
                        break;
                    case 15:
                        p31.X = A - S - S;      /* X3 = M^2-2*S */
                        A = S - p31.X;
                        break;
                    case 16:                    /* M*(S-X3) */
                        /* p31.Y = A - p31.Y; */    /* Y3 = M*(S-X3)-W*Y1 */
                        done = true;
                        break;
#undef S
                        /*** account for a4 != nullptr when doubling ***/
                    case 17:                    /* ZZ1^2 */
                        if (a4 != nullptr)
                            B = *a4;
                        break;
                    case 18:                    /* ZZ1^2*a4 */
                        if (a4 != nullptr)
                        {
                            U = A;
                            A = p2.Y << 1;        /* U = 2*Y1 */
                            B = A;
                        }
                        pc = 3;
                        break;
                }
            } while (!done);
            p31.Y = A - p31.Y;              /* Y3 = R*(Q-X3)-S1*PPP */

            *this = p31;
        }

        inline __device__ xyzz_t shfl_down(uint32_t off) const
        {
            xyzz_t ret;

            ret.X = X.shfl_down(off);
            ret.Y = Y.shfl_down(off);
            ret.ZZZ = ZZZ.shfl_down(off);
            ret.ZZ = ZZ.shfl_down(off);

            return ret;
        }
    };

} // namespace device

namespace host
{

    template<class field_t, class field_h = typename field_t::mem_t,
            const field_h *a4 = nullptr>
    class xyzz_t
    {
    public:
        field_t X, Y, ZZZ, ZZ;
        static const unsigned int degree = field_t::degree;

        using mem_t = xyzz_t;

        class affine_t
        {
            friend class xyzz_t;

        public:
            field_t X, Y;

            affine_t(const field_t &x, const field_t &y) : X(x), Y(y) {}

            inline __host__ affine_t() {}

            inline __host__   bool is_inf() const { return (bool) ((int) X.is_zero() & (int) Y.is_zero()); }

            inline __host__ affine_t &operator=(const xyzz_t &a)
            {
                Y = 1 / a.ZZZ;
                X = Y * a.ZZ;   // 1/Z
                X = X ^ 2;        // 1/Z^2
                X *= a.X;       // X/Z^2
                Y *= a.Y;       // Y/Z^3
                return *this;
            }

            inline __host__ affine_t(const xyzz_t &a) { *this = a; }

            inline __host__ operator xyzz_t() const
            {
                xyzz_t p;
                p.X = X;
                p.Y = Y;
                p.ZZZ = p.ZZ = field_t::one(is_inf());
                return p;
            }

            friend bool operator == (const affine_t &lhs,const affine_t &rhs)
            {
                for(auto i = 0; i < lhs.X.n; i++)
                    if (lhs.X[i] != rhs.X[i])
                        return false;
                for(auto i = 0; i < lhs.X.n; i++)
                    if (lhs.Y[i] != rhs.Y[i])
                        return false;
                return true;
            }

            inline affine_t &operator+=(const affine_t &other)
            {
                auto tmp = xyzz_t(*this);
                tmp.uadd(other);
                *this = affine_t(tmp);
                return *this;
            }

            inline affine_t operator+(const affine_t& other) const
            {
                affine_t result = *this;
                result += other;
                return result;
            }

            template <typename field>
            inline affine_t &operator*=(const field &other)
            {
                *this = take(xyzz_t(*this), other);
                return *this;
            }

            template <typename field>
            inline affine_t operator*(const field &other) const
            {
                affine_t result = *this;
                result *= other;
                return result;
            }

            using mem_t = affine_t;
        };

        class affine_inf_t
        {
            field_t X, Y;
            bool inf;

            inline __host__  bool is_inf() const { return inf; }

        public:
            inline __host__ operator affine_t() const
            {
                bool inf = is_inf();
                affine_t p;
                p.X = czero(X, inf);
                p.Y = czero(Y, inf);
                return p;
            }

            using mem_t = affine_inf_t;
        };

        template<class affine_t>
        inline __host__  xyzz_t &operator=(const affine_t &a)
        {
            X = a.X;
            Y = a.Y;
            ZZZ = ZZ = field_t::one(a.is_inf());
            return *this;
        }

        inline __host__ operator affine_t() const { return affine_t(*this); }

        inline __host__   bool is_inf() const { return (bool) ((int) ZZZ.is_zero() & (int) ZZ.is_zero()); }

        inline __host__  void inf()
        {
            ZZZ.zero();
            ZZ.zero();
        }

        inline __host__  void cneg(bool neg) { ZZZ.cneg(neg); }

        __host__  void add(const xyzz_t &p2)
        {
            if (p2.is_inf())
            {
                return;
            } else if (is_inf())
            {
                *this = p2;
                return;
            }

            xyzz_t &p31 = *this;
            field_t U, S, P, R;

            U = p31.X * p2.ZZ;          /* U1 = X1*ZZ2 */
            S = p31.Y * p2.ZZZ;         /* S1 = Y1*ZZZ2 */
            P = p2.X * p31.ZZ;          /* U2 = X2*ZZ1 */
            R = p2.Y * p31.ZZZ;         /* S2 = Y2*ZZZ1 */
            P -= U;                     /* P = U2-U1 */
            R -= S;                     /* R = S2-S1 */

            if (!P.is_zero())
            {         /* X1!=X2 */
                field_t PP;             /* add |p1| and |p2| */

                PP = P ^ 2;               /* PP = P^2 */
#define PPP P
                PPP = P * PP;           /* PPP = P*PP */
                p31.ZZ *= PP;           /* ZZ3 = ZZ1*ZZ2*PP */
                p31.ZZZ *= PPP;         /* ZZZ3 = ZZZ1*ZZZ2*PPP */
#define Q PP
                Q = U * PP;             /* Q = U1*PP */
                p31.X = R ^ 2;            /* R^2 */
                p31.X -= PPP;           /* R^2-PPP */
                p31.X -= Q;
                p31.X -= Q;             /* X3 = R^2-PPP-2*Q */
                Q -= p31.X;
                Q *= R;                 /* R*(Q-X3) */
                p31.Y = S * PPP;        /* S1*PPP */
                p31.Y = Q - p31.Y;      /* Y3 = R*(Q-X3)-S1*PPP */
                p31.ZZ *= p2.ZZ;        /* ZZ1*ZZ2 */
                p31.ZZZ *= p2.ZZZ;      /* ZZZ1*ZZZ2 */
#undef PPP
#undef Q
            } else if (R.is_zero())
            {   /* X1==X2 && Y1==Y2 */
                field_t M;              /* double |p1| */

                U = p31.Y + p31.Y;      /* U = 2*Y1 */
#define V P
#define W R
                V = U ^ 2;                /* V = U^2 */
                W = U * V;              /* W = U*V */
                S = p31.X * V;          /* S = X1*V */
                M = p31.X ^ 2;
                M = M + M + M;          /* M = 3*X1^2[+a*ZZ1^2] */
                if (a4 != nullptr)
                {
                    U = p31.ZZ ^ 2;
                    U *= *a4;
                    M += U;
                }
                p31.X = M ^ 2;
                p31.X -= S;
                p31.X -= S;             /* X3 = M^2-2*S */
                p31.Y *= W;             /* W*Y1 */
                S -= p31.X;
                S *= M;                 /* M*(S-X3) */
                p31.Y = S - p31.Y;      /* Y3 = M*(S-X3)-W*Y1 */
                p31.ZZ *= V;            /* ZZ3 = V*ZZ1 */
                p31.ZZZ *= W;           /* ZZZ3 = W*ZZZ1 */
#undef V
#undef W
            } else
            {                    /* X1==X2 && Y1==-Y2 */\
            p31.inf();              /* set |p3| to infinity */\

            }
        }

        inline void uadd(const xyzz_t &p2) { add(p2); }

        template<class affine_t>
        __host__  void add(const affine_t &p2, bool subtract = false)
        {
            if (p2.is_inf())
            {
                return;
            } else if (is_inf())
            {
                *this = p2;
                ZZZ.cneg(subtract);
            } else
            {
                field_t P, R;

                R = p2.Y * ZZZ;         /* S2 = Y2*ZZZ1 */
                R.cneg(subtract);
                R -= Y;                 /* R = S2-Y1 */
                P = p2.X * ZZ;          /* U2 = X2*ZZ1 */
                P -= X;                 /* P = U2-X1 */

                if (!P.is_zero())
                {     /* X1!=X2 */
                    field_t PP;             /* add |p2| to |p1| */

                    PP = P ^ 2;               /* PP = P^2 */
#define PPP P
                    PPP = P * PP;           /* PPP = P*PP */
                    ZZ *= PP;               /* ZZ3 = ZZ1*PP */
                    ZZZ *= PPP;             /* ZZZ3 = ZZZ1*PPP */
#define Q PP
                    Q = PP * X;             /* Q = X1*PP */
                    X = R ^ 2;                /* R^2 */
                    X -= PPP;               /* R^2-PPP */
                    X -= Q;
                    X -= Q;                 /* X3 = R^2-PPP-2*Q */
                    Q -= X;
                    Q *= R;                 /* R*(Q-X3) */
                    Y *= PPP;               /* Y1*PPP */
                    Y = Q - Y;              /* Y3 = R*(Q-X3)-Y1*PPP */
#undef Q
#undef PPP
                } else if (R.is_zero())
                {   /* X1==X2 && Y1==Y2 */
                    field_t M;              /* double |p2| */

#define U P
                    U = p2.Y + p2.Y;        /* U = 2*Y1 */
                    ZZ = U ^ 2;               /* [ZZ3 =] V = U^2 */
                    ZZZ = ZZ * U;           /* [ZZZ3 =] W = U*V */
#define S R
                    S = p2.X * ZZ;          /* S = X1*V */
                    M = p2.X ^ 2;
                    M = M + M + M;          /* M = 3*X1^2[+a] */
                    if (a4 != nullptr)
                    {
                        M += *a4;
                    }
                    X = M ^ 2;
                    X -= S;
                    X -= S;                 /* X3 = M^2-2*S */
                    Y = ZZZ * p2.Y;         /* W*Y1 */
                    S -= X;
                    S *= M;                 /* M*(S-X3) */
                    Y = S - Y;              /* Y3 = M*(S-X3)-W*Y1 */
#undef S
#undef U
                    ZZZ.cneg(subtract);
                } else
                {                    /* X1==X2 && Y1==-Y2 */
                    inf();                  /* set |p3| to infinity */
                }
            }
        }

        inline void uadd(const affine_t &p2, bool subtract = false) { add(p2, subtract); }

        inline xyzz_t &operator+=(const xyzz_t &other)
        {
            this->uadd(other);
            return *this;
        }

        inline xyzz_t operator+(const xyzz_t& other) const
        {
            xyzz_t result = *this;
            result += other;
            return result;
        }

        template <typename field>
        inline xyzz_t &operator*=(const field &other)
        {
            *this = take(*this, other);
            return *this;
        }

        template <typename field>
        inline xyzz_t operator*(const field &other) const
        {
            xyzz_t result = *this;
            result *= other;
            return result;
        }
    };

} // namespace host


namespace host
{
    template<typename group, typename field>
    group take(group point, field scalar)
    {
        scalar.from();
        group result;
        result.inf();
        for (int i = scalar.n - 1; i >= 0; i--)
        {
            uint64_t tmp = scalar[i];
            for (auto j = 63; j >= 0; j--)
            {
                result.uadd(result);
                if (tmp & (1ull << j))
                {
//                printf("add at int.no: %d, bit: %d\n", i, j);
                    result.uadd(point);
                }
            }
        }
        return result;
    }
}

namespace device
{
    template<typename group, typename field>
    __device__ group take(group point, field scalar)
    {
        scalar.from();
        group result;
        result.inf();
        for (int i = scalar.n - 1; i >= 0; i--)
        {
            uint32_t tmp = scalar[i];
            for (auto j = 31; j >= 0; j--)
            {
                result.uadd(result);
                if (tmp & (1ull << j))
                {
//                printf("add at int.no: %d, bit: %d\n", i, j);
                    result.uadd(point);
                }
            }
        }
        return result;
    }
}

# pragma nv_diag_default 284

#endif