#include <stddef.h>
#include <assert.h>
#include <x86intrin.h>  // soporte para intrisics

#include "solver.h"

#define IX(i,j) ((i)+(n+2)*(j))
#define SWAP(x0,x) {float * tmp=x0;x0=x;x=tmp;}
#define N_MAX_DUMMY (2 << 30)

#define SHIFT_LEFT(_mm_value) _mm_shuffle_ps((_mm_value), (_mm_value), _MM_SHUFFLE(0, 3, 2, 1))  
#define SHIFT_RIGHT(_mm_value) _mm_shuffle_ps((_mm_value), (_mm_value), _MM_SHUFFLE(2, 1, 0, 3))  

typedef enum { NEITHER = 0, HORIZONTAL = 1, VERTICAL = 2 } boundary;

static void add_source(unsigned int n, float * x, const float * s, float dt)
{
    unsigned int size = (n + 2) * (n + 2);
    for (unsigned int i = 0; i < size; i++) { //vectorizado automatico con -ffast-math 
        x[i] += dt * s[i];
    }
}

static void set_bnd(unsigned int n, boundary b, float * x)
{   
    for (unsigned int i = 1; i <= n; i += 2) { //loop unrolling manual
        x[IX(0, i)]     = b == 1 ? -x[IX(1, i)] : x[IX(1, i)];
        x[IX(0, i+1)]   = b == 1 ? -x[IX(1, i+1)] : x[IX(1, i+1)];

        x[IX(n + 1, i)] = b == 1 ? -x[IX(n, i)] : x[IX(n, i)];
        x[IX(n + 1, i+1)] = b == 1 ? -x[IX(n, i+1)] : x[IX(n, i+1)];
    }
    for (unsigned int i = 1; i <= n; i++) { //vectorizado automatico
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
        for (unsigned int i = 1; i <= n; i += 2) {
            __m128 r0, r1, r2;
            r0  = _mm_loadu_ps((float*) &x[IX(i - 1, 0)]);
            r1  = _mm_loadu_ps((float*) &x[IX(i - 1, 1)]);
            for (unsigned int j = 1; j <= n; j++) {
   /* original: x[IX(i, j)] = (x0[IX(i, j)] + a * (x[IX(i - 1, j)] + x[IX(i + 1, j)] +
                                                     x[IX(i, j - 1)] + x[IX(i, j + 1)]) ) / c; */             
                __m128 _x0 = _mm_loadu_ps((float*) &x0[IX(i - 1, j + 0)]);
                r2  = _mm_loadu_ps((float*) &x[IX(i - 1, j + 1)]);
                                                
                // add1 = ( _ , x[i-1][j] + x[i+1][j] , x[i-1][j+1] + x[i+1][j+1] , _ )
                __m128 add1 = _mm_add_ps(r0, r2);
                // ( _ , x[i][j+1] , x[i][j+2] , _ )
                __m128 right_x2   = SHIFT_LEFT(r1);
                // add2 = ( _ , add1[1] + x[i][j+1] , add1[2] + x[i][j+2], _ )               
                __m128 add2 = _mm_add_ps(add1, right_x2);
                // ( x[i][j-1] , x[i][j-1] , 0, 0)
                __m128 left_1 = _mm_shuffle_ps(r1, zeros, _MM_SHUFFLE(0, 0, 0, 0));
                // add3 = ( _ , add2[1] + x[i][j-1], add2[2], _)                
                __m128 add3 = _mm_add_ps(add2, left_1);
                // add3 * a / c               
                __m128 add3_mul_a_div_c = _mm_mul_ps(add3, a_div_c_s);
                // x0_div_c = (_ , x0[i][j+1] / c , x0[i][j+1] / c , _) 
                __m128 x0_div_c = _mm_mul_ps(_x0, inv_c_s);
                // add4 = x0_div_c + add3_mul_a_div_c 
                __m128 add4 = _mm_add_ps(x0_div_c, add3_mul_a_div_c);
                // (0, 0, add4[1], add4[1])
                __m128 left_2 = _mm_shuffle_ps(zeros, add4, _MM_SHUFFLE(1, 1, 0, 0));
                // (0, 0, add4[1] * a / c, add4[1] * a / c)
                __m128 left_2_mul_a_div_c = _mm_mul_ps(left_2, a_div_c_s);
                // add5 = (_, add4[1], add4[1] * a / c + add4[2], _)
                __m128 add5 = _mm_add_ps(add4, left_2_mul_a_div_c);               
                // res = (x[i][j-1] , add5[1] , add5[2] , x[i][j+2] )
                
                //__m128 res = _mm_blend_ps(r1, add5, 6); //0110 ; arch >= sse4.1
                // arch < mss4.1
                __m128 res_aux = _mm_shuffle_ps(add5, r1, _MM_SHUFFLE(3, 0, 1, 2));
                __m128 res = _mm_shuffle_ps(res_aux, res_aux, _MM_SHUFFLE(3, 0, 1, 2));
                
                _mm_storeu_ps((float*) &x[IX(i - 1, j + 0)], res);
                
                r0 = res; r1 = r2; //reutilizo valores - importante optimizacion
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
    // se agrega la cota N_MAX_DUMMY para que vectorice automaticamente el loop
    // gcc no queria vectorizarlo por que las cantidad de iteraciones son no computable
    assert(n < N_MAX_DUMMY);

    // recorrido row major (inverti el orden del recorrido)
    for (unsigned int i = 1; i <= n; i++) {
        for (unsigned int j = 1; j <= n; j++) { // vectorizado automatico con -ffast-math        
            div[IX(j, i)] = -0.5f * (u[IX(j + 1, i)] - u[IX(j - 1, i)] +
                                     v[IX(j, i + 1)] - v[IX(j, i - 1)]) / n;
            p[IX(j, i)] = 0;
        }
    }
    
    set_bnd(n, 0, div);
    set_bnd(n, 0, p);
    lin_solve(n, 0, p, div, 1, 4);

    // recorrido row major (inverti el orden del recorrido)
    for (unsigned int i = 1; i <= n; i++) {
        for (unsigned int j = 1; j <= n; j++) { //vectorizado automatico con --ffast-math
            u[IX(j  , i)] -= 0.5f * n * (p[IX(j + 1, i)] - p[IX(j - 1, i)]);
            v[IX(j, i)] -= 0.5f * n * (p[IX(j, i + 1)] - p[IX(j, i - 1)]);
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
