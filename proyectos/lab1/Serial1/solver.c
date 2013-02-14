#include <stddef.h>

#include <nmmintrin.h> // soporte para sse4.2

#include "solver.h"

#define IX(i,j) ((i)+(n+2)*(j))
#define SWAP(x0,x) {float * tmp=x0;x0=x;x=tmp;}

typedef enum { NEITHER = 0, HORIZONTAL = 1, VERTICAL = 2 } boundary;

static void add_source(unsigned int n, float * x, const float * s, float dt)
{
    unsigned int size = (n + 2) * (n + 2);
    for (unsigned int i = 0; i < size; i++) {
        x[i] += dt * s[i];
    }
}

static void set_bnd(unsigned int n, boundary b, float * x)
{
    for (unsigned int i = 1; i <= n; i++) {
        x[IX(0, i)]     = b == 1 ? -x[IX(1, i)] : x[IX(1, i)];
        x[IX(n + 1, i)] = b == 1 ? -x[IX(n, i)] : x[IX(n, i)];
        x[IX(i, 0)]     = b == 2 ? -x[IX(i, 1)] : x[IX(i, 1)];
        x[IX(i, n + 1)] = b == 2 ? -x[IX(i, n)] : x[IX(i, n)];
    }
    x[IX(0, 0)]         = 0.5f * (x[IX(1, 0)]     + x[IX(0, 1)]);
    x[IX(0, n + 1)]     = 0.5f * (x[IX(1, n + 1)] + x[IX(0, n)]);
    x[IX(n + 1, 0)]     = 0.5f * (x[IX(n, 0)]     + x[IX(n + 1, 1)]);
    x[IX(n + 1, n + 1)] = 0.5f * (x[IX(n, n + 1)] + x[IX(n + 1, n)]);
}

static void lin_solve(unsigned int n, boundary b, float * __restrict__ x, const float * __restrict__ x0, float a, float c)
{    
    __m128 inv_c_s   = _mm_set1_ps(1. / c);  // (1/c, 1/c, 1/c, 1/c)
    __m128 a_div_c_s = _mm_set1_ps(a / c);   // (a/c, a/c, a/c, a/c)
    __m128 zeros     = _mm_setzero_ps();     // (0, 0, 0, 0)
    
    for (unsigned int k = 0; k < 20; k++) {
        for (unsigned int i = 1; i <= n; i++) {
            for (unsigned int j = 1; j <= n; j += 2) {
   /* original: x[IX(i, j)] = (x0[IX(i, j)] + a * (x[IX(i - 1, j)] + x[IX(i + 1, j)] +
                                                     x[IX(i, j - 1)] + x[IX(i, j + 1)]) ) / c; */             
                // leo de la memoria (desalineada)
                __m128 _x0 = _mm_loadu_ps((float*) &x0[IX(i + 0, j - 1)]);
                __m128 r0  = _mm_loadu_ps((float*) &x[IX(i - 1, j - 1)]);
                __m128 r1  = _mm_loadu_ps((float*) &x[IX(i + 0, j - 1)]);
                __m128 r2  = _mm_loadu_ps((float*) &x[IX(i + 1, j - 1)]);

                // add1 = ( _ , x[i-1][j] + x[i+1][j] , x[i-1][j+1] + x[i+1][j+1] , _ )
                __m128 up_down_x2 = _mm_add_ps(r0, r2);
                // ( x[i][j+1] , x[i][j+1] , x[i][j+2] , x[i][j+2] )
                __m128 right_x2   = _mm_shuffle_ps(r1, r1, _MM_SHUFFLE(3, 3, 2, 2));
                // add2 = ( _ , add1[1] + x[i][j+1] , add1[2] + x[i][j+2], _ )               
                __m128 up_down_right_x2 = _mm_add_ps(up_down_x2, right_x2);
                // ( x[i][j-1] , x[i][j-1] , 0, 0)
                __m128 left_1 = _mm_shuffle_ps(r1, zeros, _MM_SHUFFLE(0, 0, 0, 0));
                // add3 = ( _ , add2[1] + x[i][j-1], add2[2], _)                
                __m128 up_down_right_x2_left_1 = _mm_add_ps(up_down_right_x2, left_1);
                // (0, 0, add3[1], add3[1])
                __m128 left_2 = _mm_shuffle_ps(zeros, up_down_right_x2_left_1, _MM_SHUFFLE(1, 1, 0, 0));
                // add4 = ( _ , add3[1] , add3[1] + add3[2] , _ ) ;
                __m128 up_down_right_x2_left_x2 = _mm_add_ps(left_2, up_down_right_x2_left_1);
                // add4 * a / c               
                __m128 x_mul_a_div_c = _mm_mul_ps(up_down_right_x2_left_x2, a_div_c_s);
                // (_ , x0[i][j+1] / c , x0[i][j+1] / c , _) 
                __m128 x0_div_c = _mm_mul_ps(_x0, inv_c_s);
                // new = add4 * a / c + x0 / c
                __m128 x_mul_a_plus_x0_div_c = _mm_add_ps(x_mul_a_div_c, x0_div_c);
                // res = (x[i][j-1] , new[1] , new[2] , x[i][j+2] )
                __m128 res = _mm_blend_ps(r1, x_mul_a_plus_x0_div_c, 6); //0110
                // escribo en la memoria                
                _mm_storeu_ps((float*) &x[IX(i + 0, j - 1)], res);
            }            
        }
        set_bnd(n, b, x);
    }
}

static void diffuse(unsigned int n, boundary b, float * x, const float * x0, float diff, float dt)
{
    float a = dt * diff * n * n;
    lin_solve(n, b, x, x0, a, 1 + 4 * a);
}

static void advect(unsigned int n, boundary b, float * d, const float * d0, const float * u, const float * v, float dt)
{
    int i0, i1, j0, j1;
    float x, y, s0, t0, s1, t1;

    float dt0 = dt * n;
    for (unsigned int i = 1; i <= n; i++) {
        for (unsigned int j = 1; j <= n; j++) {
            x = i - dt0 * u[IX(i, j)];
            y = j - dt0 * v[IX(i, j)];
            if (x < 0.5f) {
                x = 0.5f;
            } else if (x > n + 0.5f) {
                x = n + 0.5f;
            }
            i0 = (int) x;
            i1 = i0 + 1;
            if (y < 0.5f) {
                y = 0.5f;
            } else if (y > n + 0.5f) {
                y = n + 0.5f;
            }
            j0 = (int) y;
            j1 = j0 + 1;
            s1 = x - i0;
            s0 = 1 - s1;
            t1 = y - j0;
            t0 = 1 - t1;
            d[IX(i, j)] = s0 * (t0 * d0[IX(i0, j0)] + t1 * d0[IX(i0, j1)]) +
                          s1 * (t0 * d0[IX(i1, j0)] + t1 * d0[IX(i1, j1)]);
        }
    }
    set_bnd(n, b, d);
}

static void project(unsigned int n, float *u, float *v, float *p, float *div)
{
    for (unsigned int i = 1; i <= n; i++) {
        for (unsigned int j = 1; j <= n; j++) {
            div[IX(i, j)] = -0.5f * (u[IX(i + 1, j)] - u[IX(i - 1, j)] +
                                     v[IX(i, j + 1)] - v[IX(i, j - 1)]) / n;
            p[IX(i, j)] = 0;
        }
    }
    set_bnd(n, 0, div);
    set_bnd(n, 0, p);

    lin_solve(n, 0, p, div, 1, 4);

    for (unsigned int i = 1; i <= n; i++) {
        for (unsigned int j = 1; j <= n; j++) {
            u[IX(i, j)] -= 0.5f * n * (p[IX(i + 1, j)] - p[IX(i - 1, j)]);
            v[IX(i, j)] -= 0.5f * n * (p[IX(i, j + 1)] - p[IX(i, j - 1)]);
        }
    }
    set_bnd(n, 1, u);
    set_bnd(n, 2, v);
}

void dens_step(unsigned int n, float *x, float *x0, float *u, float *v, float diff, float dt)
{
    add_source(n, x, x0, dt);
    SWAP(x0, x);
    diffuse(n, 0, x, x0, diff, dt);
    SWAP(x0, x);
    advect(n, 0, x, x0, u, v, dt);
}

void vel_step(unsigned int n, float *u, float *v, float *u0, float *v0, float visc, float dt)
{
    add_source(n, u, u0, dt);
    add_source(n, v, v0, dt);
    SWAP(u0, u);
    diffuse(n, 1, u, u0, visc, dt);
    SWAP(v0, v);
    diffuse(n, 2, v, v0, visc, dt);
    project(n, u, v, u0, v0);
    SWAP(u0, u);
    SWAP(v0, v);
    advect(n, 1, u, u0, u0, v0, dt);
    advect(n, 2, v, v0, u0, v0, dt);
    project(n, u, v, u0, v0);
}
