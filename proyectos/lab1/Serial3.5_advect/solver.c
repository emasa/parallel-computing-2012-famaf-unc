#include <stddef.h>
#include <assert.h>
#include <math.h>       // soporte operaciones matematicas

#include <x86intrin.h> 

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

    for (unsigned int k = 0; k < 20; k++) {
        for (unsigned int i = 1; i <= n; i += 4) {
            __m128 r0, r1, r2, res;
            float res_1 = x[IX(i-1, 1)];
            r0  = _mm_loadu_ps((float*) &x[IX(i, 0)]);

            for (unsigned int j = 1; j <= n; j++) {
   /* original: x[IX(i, j)] = (x0[IX(i, j)] + a * (x[IX(i - 1, j)] + x[IX(i + 1, j)] +
                                                     x[IX(i, j - 1)] + x[IX(i, j + 1)]) ) / c; */             

                __m128 _x0 = _mm_loadu_ps((float*) &x0[IX(i, j)]);
                _x0 = _mm_mul_ps(_x0, inv_c_s);

                r1  = _mm_loadu_ps((float*) &x[IX(i+1, j)]);
                r2  = _mm_loadu_ps((float*) &x[IX(i, j + 1)]);
               
                __m128 add = _mm_add_ps(_mm_add_ps(r0, r1), r2);
                __m128 add_mul_a_div_c = _mm_mul_ps(add, a_div_c_s);
                
                res = _mm_add_ps(add_mul_a_div_c, _x0);

                for(unsigned int l = 0; l < 4; l++){
                    res[l] = res_1 = (a / c) * res_1 + res[l];
                }
                _mm_storeu_ps((float*) &x0[IX(i, j)], res);
                r0 = res ; res_1 = x[IX(i-1, j+1)];
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

    // recorrido row major (inverti el orden del recorrido)
    for (unsigned int j = 1; j <= n; j++) {
        for (unsigned int i = 1; i <= n; i++) {
            x = i - dt0 * u[IX(i, j)];
            y = j - dt0 * v[IX(i, j)];

            // evito los branches            
            x = fminf(fmaxf(0.5f, x), n + 0.5f); 
            y = fminf(fmaxf(0.5f, y), n + 0.5f);
            
            i0 = (int) x;
            j0 = (int) y;

            s1 = x - i0;
            t1 = y - j0;

            i1 =  (i0 + 1);
            j1 =  (j0 + 1);

            s0 = 1 - s1;    
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
            v[IX(j, i)]   -= 0.5f * n * (p[IX(j, i + 1)] - p[IX(j, i - 1)]);
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