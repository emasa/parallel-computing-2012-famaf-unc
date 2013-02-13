#define _BSD_SOURCE

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include <sys/time.h>
#include <time.h>

#include "transpose_intrinsic.h"

#include <ammintrin.h>

#define REP(i, n) for(size_t i = 0; i < (n) ; i++)
#define AT(a, i, j, n) (a)[(i) * (n) + (j)]

#define SIZE (1 << 10)

float _random() {
    float sign = 1. * rand() / RAND_MAX;
    float value = 1. * rand() / RAND_MAX;
    return sign < 0.5 ? value : -value;
}

void init(size_t n, float * a, float * b, float * c)
{
    srand(time(NULL));
    REP(i, n)
    {
        REP(j, n)
        {
            AT(a, i, j, n) = _random();
            AT(c, i, j, n) = _random();
        }
    }
    transpose_2d_blocking_intrinsic_aligned(SIZE, SIZE, c, b);
    REP(i, n)
    {
        REP(j, n)
        {
            AT(c, i, j, n) = 0.0f;
        }
    }
}
void mmulf_lineal(size_t n, const float * a, const float * b, float * c)
{
    REP(i, n)
    {
        REP(j, n)
        {
            REP(k, n)
            {
                AT(c, i, j, n) += AT(a, i, k, n) * AT(b, j, k, n);
            }
        }
    }
}
								       
int main()
{
	float * a = (float *) _mm_malloc(sizeof(float) * SIZE * SIZE, 16);
	assert(a != NULL);
	float * b = (float *) _mm_malloc(sizeof(float) * SIZE * SIZE, 16);
    assert(b != NULL);
	float * c = (float *) _mm_malloc(sizeof(float) * SIZE * SIZE, 16);
    assert(c != NULL);
                            
	init(SIZE, a, b, c);        
    mmulf_lineal(SIZE, a, b, c);                
    printf("c[0][0] = %f\n", AT(c, 0, 0, SIZE));       
    
    _mm_free(a);
    _mm_free(b);
    _mm_free(c);
    
	return 0;
}
