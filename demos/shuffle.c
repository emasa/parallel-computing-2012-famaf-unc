
#include <stdio.h>
#include <nmmintrin.h>

int main()
{   
    float x[] = {-1., 3., 1., 2.};
    float res[4];
    
    __m128 zeros = _mm_setzero_ps();  //(0, 0, 0, 0)
    __m128 _mm_x = _mm_loadu_ps(&x[0]);
    
    __m128 _mm_res;
    _mm_res = _mm_shuffle_ps(_mm_x, zeros, _MM_SHUFFLE(0, 0, 0, 0));
    _mm_storeu_ps((float*) &res[0], _mm_res);
    printf("%f %f %f %f\n", res[0], res[1], res[2], res[3]);

    _mm_res = _mm_shuffle_ps(_mm_x, _mm_x, _MM_SHUFFLE(3, 3, 2, 2));
    _mm_storeu_ps((float*) &res[0], _mm_res);
    printf("%f %f %f %f\n", res[0], res[1], res[2], res[3]);

    _mm_res = _mm_shuffle_ps(zeros, _mm_x, _MM_SHUFFLE(1, 1, 0, 0));
    _mm_storeu_ps((float*) &res[0], _mm_res);
    printf("%f %f %f %f\n", res[0], res[1], res[2], res[3]);
    
    _mm_res = _mm_blend_ps(zeros, _mm_x, 6);    
    _mm_storeu_ps((float*) &res[0], _mm_res);
    printf("%f %f %f %f\n", res[0], res[1], res[2], res[3]);

    return 0;
}
