#include <stddef.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include <x86intrin.h>  // soporte para intrisics

#include "solver.h"

#define IX(i,j) ((i)+(n+2)*(j))
#define SWAP(x0,x) {float * tmp=x0;x0=x;x=tmp;}
#define N_MAX_DUMMY (2 << 30)

#define SHIFT_LEFT(_mm_value) _mm_shuffle_ps((_mm_value), (_mm_value), _MM_SHUFFLE(0, 3, 2, 1))  
#define SHIFT_RIGHT(_mm_value) _mm_shuffle_ps((_mm_value), (_mm_value), _MM_SHUFFLE(2, 1, 0, 3))  

typedef enum { NEITHER = 0, HORIZONTAL = 1, VERTICAL = 2 } boundary;

static void add_source(unsigned int n, float * __restrict__ x, const float * __restrict__ s, float dt)
{
    unsigned int size = (n + 2) * (n + 2);
    // vectorizado automatico con -ffast-math && __restrict__
    for (unsigned int i = 0; i < size; i++) {
        x[i] += dt * s[i];
    }
}

static void set_bnd(unsigned int n, boundary b, float * x)
{   
    // no influye
    for (unsigned int i = 1; i <= n; i++) {
        x[IX(0, i)]     = b == HORIZONTAL ? -x[IX(1, i)] : x[IX(1, i)];
        x[IX(n + 1, i)] = b == HORIZONTAL ? -x[IX(n, i)] : x[IX(n, i)];
        x[IX(i, 0)]     = b == VERTICAL ? -x[IX(i, 1)] : x[IX(i, 1)];
        x[IX(i, n + 1)] = b == VERTICAL ? -x[IX(i, n)] : x[IX(i, n)];
    }

    x[IX(0, 0)]         = 0.5f * (x[IX(1, 0)]     + x[IX(0, 1)]);
    x[IX(0, n + 1)]     = 0.5f * (x[IX(1, n + 1)] + x[IX(0, n)]);
    x[IX(n + 1, 0)]     = 0.5f * (x[IX(n, 0)]     + x[IX(n + 1, 1)]);
    x[IX(n + 1, n + 1)] = 0.5f * (x[IX(n, n + 1)] + x[IX(n + 1, n)]);
}

static void lin_solve(unsigned int n, boundary b, float * x, const float * x0, float a, float c)
{    
    __m128 inv_c_s   = _mm_set1_ps(1. / c);  // (1/c, 1/c, 1/c, 1/c)
    __m128 a_div_c_s = _mm_set1_ps(a / c);   // (a/c, a/c, a/c, a/c)

    float res_tmp[4] __attribute__((aligned(16)));
    for (unsigned int k = 0; k < 20; k++) {
        for (unsigned int i = 1; i <= n; i++) {
            // extremo que no se puede vectorizar
            for (unsigned int j = 1; j <= 3; j++) {
                x[IX(j, i)] = (x0[IX(j, i)] + a * (x[IX(j - 1, i)] + x[IX(j + 1, i)] +
                                                   x[IX(j, i - 1)] + x[IX(j, i + 1)]) ) / c;
            }
                
            float left1 = x[IX(3, i)]; // para opcion 1            
            //__m128 left = _mm_loadu_ps((float*) &x[IX(0, i)]); // para opcion 2
            for (unsigned int j = 4; j <= n-3; j += 4) {
                __m128 x0_vec = _mm_loadu_ps((float*) &x0[IX(j, i)]);
                __m128 x0_div_c = _mm_mul_ps(x0_vec, inv_c_s);
                
                __m128 up  = _mm_loadu_ps((float*) &x[IX(j, i-1)]);
                __m128 down = _mm_loadu_ps((float*) &x[IX(j, i+1)]);                
                __m128 right = _mm_loadu_ps((float*) &x[IX(j+1, i)]); // desalineado
                
                __m128 sum = _mm_add_ps(_mm_add_ps(up, down), right);
                __m128 res = _mm_add_ps(_mm_mul_ps(sum, a_div_c_s), x0_div_c);
                
                /********************* opcion 1 ******************/
                _mm_storeu_ps((float*) &res_tmp, res);
                for(unsigned int l = 0; l < 4; l++){
                    res_tmp[l] = left1 = left1*a/c + res_tmp[l];
                }
                res = _mm_loadu_ps((float*) &res_tmp);
                /*************************************************/                                
                 /**************** opcion 2 (mas lenta) **********/
                /*
                left = SHIFT_RIGHT(left);                
                for(unsigned int l = 0; l < 4; l++){
                    left  = _mm_mul_ss(left, a_div_c_s);
                    left  = _mm_add_ss(res, left);
                    res = SHIFT_LEFT(left);
                }
                left = res;                
                */
                /*************************************************/                
                _mm_storeu_ps((float*) &x[IX(j, i)], res);
            }
            // extremo que no se puede vectorizar
            for (unsigned int j = n-2; j <= n; j++) {
                x[IX(j, i)] = (x0[IX(j, i)] + a * (x[IX(j - 1, i)] + x[IX(j + 1, i)] +
                                                   x[IX(j, i - 1)] + x[IX(j, i + 1)]) ) / c;
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

static void advect(unsigned int n, boundary b, float * d, const float * d0, 
                   const float * u, const float * v, float dt)
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
            
            // el acceso a d0 no es vectorizable            
            d[IX(i, j)] = s0 * (t0 * d0[IX(i0, j0)] + t1 * d0[IX(i0, j1)]) +
                          s1 * (t0 * d0[IX(i1, j0)] + t1 * d0[IX(i1, j1)]);
        }
    }
    set_bnd(n, b, d);
}

static void project(unsigned int n, float * __restrict__ u, float * __restrict__ v, 
                    float * __restrict__ p, float *__restrict__ div)
{   
    __m128 inv_neg_2n = _mm_set1_ps(-1. / (2 * n));
    // recorrido row major (inverti el orden del recorrido)
    for (unsigned int i = 1; i <= n; i++) {
        // extremo que no se puede vectorizar
        for (unsigned int j = 1; j <= 3; j++) {
            div[IX(j, i)] = -0.5f * (u[IX(j + 1, i)] - u[IX(j - 1, i)] +
                                     v[IX(j, i + 1)] - v[IX(j, i - 1)]) / n;
        }
        for (unsigned int j = 4; j <= n-3; j += 4) {
            // lecturas desalineadas
            __m128 u_left  = _mm_loadu_ps((float*) &u[IX(j-1, i)]);
            __m128 u_right = _mm_loadu_ps((float*) &u[IX(j+1, i)]);
            __m128 u_sub = _mm_sub_ps(u_right, u_left);            

            __m128 v_up = _mm_loadu_ps((float*) &v[IX(j, i-1)]);            
            __m128 v_down = _mm_loadu_ps((float*) &v[IX(j, i+1)]);
            __m128 v_sub = _mm_sub_ps(v_down, v_up);
            
            __m128 add = _mm_add_ps(u_sub, v_sub);            
            __m128 res = _mm_mul_ps(add, inv_neg_2n);
            
            _mm_storeu_ps((float*) &div[IX(j, i)], res);            
        }
        // extremo que no se puede vectorizar
        for (unsigned int j = n-2; j <= n; j++) {
            div[IX(j, i)] = -0.5f * (u[IX(j + 1, i)] - u[IX(j - 1, i)] +
                                     v[IX(j, i + 1)] - v[IX(j, i - 1)]) / n;
        }

    }
    // equivalente a setear el centro de p en 0 y despues correr set_bnd
    memset(p, 0, (n+2)*(n+2)*sizeof(float));
    
    set_bnd(n, 0, div);
    lin_solve(n, 0, p, div, 1, 4);

    __m128 n_div_2 = _mm_set1_ps(n / 2.);
    // recorrido row major (inverti el orden del recorrido)
    for (unsigned int i = 1; i <= n; i++) {
        // extremo que no se puede vectorizar
        for (unsigned int j = 1; j <= 3; j++) {
            u[IX(j, i)] -= 0.5f * n * (p[IX(j + 1, i)] - p[IX(j - 1, i)]);
            v[IX(j, i)] -= 0.5f * n * (p[IX(j, i + 1)] - p[IX(j, i - 1)]);
        }
        for (unsigned int j = 4; j <= n-3; j+=4) {
            // lecturas desalineadas
            __m128 p_left  = _mm_loadu_ps((float*) &p[IX(j-1, i)]);
            __m128 p_right = _mm_loadu_ps((float*) &p[IX(j+1, i)]);            
            __m128 p_subh = _mm_sub_ps(p_right, p_left);
            __m128 p_subh_mul_n_div_2 = _mm_mul_ps(p_subh, n_div_2); 
            __m128 u_center = _mm_loadu_ps((float*) &u[IX(j, i)]);
            __m128 u_sub_p  = _mm_sub_ps(u_center, p_subh_mul_n_div_2);
            _mm_storeu_ps((float*) &u[IX(j, i)], u_sub_p);    

            __m128 p_up = _mm_loadu_ps((float*) &p[IX(j, i-1)]);            
            __m128 p_down = _mm_loadu_ps((float*) &p[IX(j, i+1)]);
            __m128 p_subv = _mm_sub_ps(p_down, p_up);                
            __m128 p_subv_mul_n_div_2 = _mm_mul_ps(p_subv, n_div_2); 
            __m128 v_center = _mm_loadu_ps((float*) &v[IX(j, i)]);
            __m128 v_sub_p  = _mm_sub_ps(v_center, p_subv_mul_n_div_2);
            _mm_storeu_ps((float*) &v[IX(j, i)], v_sub_p);             
        }
        // extremo que no se puede vectorizar        
        for (unsigned int j=n-2; j <= n; j++) {
            u[IX(j, i)] -= 0.5f * n * (p[IX(j + 1, i)] - p[IX(j - 1, i)]);
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
