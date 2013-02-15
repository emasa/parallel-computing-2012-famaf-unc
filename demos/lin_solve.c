#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <nmmintrin.h> // soporte para sse4.2

#define AT(m, i, j, n) (m)[(i) * (n) + (j)]
#define IX(i,j) ((i)+(n+2)*(j))

typedef enum { NEITHER = 0, HORIZONTAL = 1, VERTICAL = 2 } boundary;

static void lin_solve(unsigned int n, boundary b, float * __restrict__ x, const float * __restrict__ x0, float a, float c)
{    
    __m128 inv_c_s   = _mm_set1_ps(1. / c);  // (1/c, 1/c, 1/c, 1/c)
    __m128 a_div_c_s = _mm_set1_ps(a / c);   // (a/c, a/c, a/c, a/c)
    __m128 zeros     = _mm_setzero_ps();     // (0, 0, 0, 0)
    
    int dummy = 0;
    //for (unsigned int k = 0; k < 20; k++) {
        for (unsigned int i = 1; i <= n; i += 2) {
            for (unsigned int j = 1; j <= n; j++) {
   /* original: x[IX(i, j)] = (x0[IX(i, j)] + a * (x[IX(i - 1, j)] + x[IX(i + 1, j)] +
                                                     x[IX(i, j - 1)] + x[IX(i, j + 1)]) ) / c; */             
                // leo de la memoria (desalineada)
                __m128 _x0 = _mm_loadu_ps((float*) &x0[IX(i - 1, j + 0)]);
                __m128 r0  = _mm_loadu_ps((float*) &x[IX(i - 1, j - 1)]);
                __m128 r1  = _mm_loadu_ps((float*) &x[IX(i - 1, j + 0)]);
                __m128 r2  = _mm_loadu_ps((float*) &x[IX(i - 1, j + 1)]);
                
                dummy++;
                                
                // add1 = ( _ , x[i-1][j] + x[i+1][j] , x[i-1][j+1] + x[i+1][j+1] , _ )
                __m128 add1 = _mm_add_ps(r0, r2);
                // ( x[i][j+1] , x[i][j+1] , x[i][j+2] , x[i][j+2] )
                __m128 right_x2   = _mm_shuffle_ps(r1, r1, _MM_SHUFFLE(3, 3, 2, 2));
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
                __m128 res = _mm_blend_ps(r1, add5, 6); //0110
                // escribo en la memoria                
                _mm_storeu_ps((float*) &x[IX(i - 1, j + 0)], res);
            }            
        }
    //    set_bnd(n, b, x);
    //}
}

int main(){
    
    int size;
    scanf("%i\n", &size);
    assert (4 <= size && size <= 1024);
    
    float * x = (float *) malloc (size * size * sizeof(float));
    assert(x != NULL);
    float * x0 = (float *) malloc (size * size * sizeof(float));
    assert(x0 != NULL);
    
    float a = 2., c = 1.;
    
    for (int i = 0 ; i < size; i++){
        for (int j = 0 ; j < size; j++){
            scanf("%f ", &AT(x, i, j, size));
            AT(x0, i, j, size-1) = 0.0;
        }
    }
    
    lin_solve(size - 2, 0, x, x0, a, c);
    
    for (int i = 0 ; i < size; i++){
        for (int j = 0 ; j < size; j++){
            printf("%.1f\t", AT(x, i, j, size));
        }
        printf("\n");
    }
    
    free(x);
    free(x0);
        
    return 0;
}

