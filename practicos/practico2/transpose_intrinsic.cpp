
#include <stddef.h>
#include <assert.h>

#include <ammintrin.h>

#include "transpose_intrinsic.h"

#define BX 4
#define BY 16

void transpose_4x4_aligned(float * in, 
                   float * out, 
                   size_t height, 
                   size_t width, 
                   size_t y0, 
                   size_t x0)
{
    __m128 row0 = _mm_load_ps((float *) &in[(y0 + 0) * height + x0]);
    __m128 row1 = _mm_load_ps((float *) &in[(y0 + 1) * height + x0]);
    __m128 row2 = _mm_load_ps((float *) &in[(y0 + 2) * height + x0]);
    __m128 row3 = _mm_load_ps((float *) &in[(y0 + 3) * height + x0]);
    
    _MM_TRANSPOSE4_PS(row0, row1, row2, row3);

    _mm_store_ps((float *) &out[(x0 + 0) * width + y0], row0);
    _mm_store_ps((float *) &out[(x0 + 1) * width + y0], row1);
    _mm_store_ps((float *) &out[(x0 + 2) * width + y0], row2);
    _mm_store_ps((float *) &out[(x0 + 3) * width + y0], row3);
}

void transpose_2d_blocking_intrinsic_aligned(  size_t width, 
								       size_t height,
								       float * in, 
								       float * out)
{   
    assert(0 < width && 0 < height);
    
    for (size_t i = 0 ; i < height; i += BY)
    {
        for (size_t j = 0 ; j < width ; j += BX)
        {
/*			transpose_4x4(in, out, height, width, i + 0, j + 0);
			transpose_4x4(in, out, height, width, i + 4, j + 0);
			transpose_4x4(in, out, height, width, i + 0, j + 4);
			transpose_4x4(in, out, height, width, i + 4, j + 4);            
*/
			transpose_4x4_aligned(in, out, height, width, i + 0, j + 0);
			transpose_4x4_aligned(in, out, height, width, i + 4, j + 0);
			transpose_4x4_aligned(in, out, height, width, i + 8, j + 0);
			transpose_4x4_aligned(in, out, height, width, i + 12, j + 0);

        }
    }
}

void transpose_4x4_unaligned(float * in, 
                   float * out, 
                   size_t height, 
                   size_t width, 
                   size_t y0, 
                   size_t x0)
{
    __m128 row0 = _mm_loadu_ps((float *) &in[(y0 + 0) * height + x0]);
    __m128 row1 = _mm_loadu_ps((float *) &in[(y0 + 1) * height + x0]);
    __m128 row2 = _mm_loadu_ps((float *) &in[(y0 + 2) * height + x0]);
    __m128 row3 = _mm_loadu_ps((float *) &in[(y0 + 3) * height + x0]);
    
    _MM_TRANSPOSE4_PS(row0, row1, row2, row3);

    _mm_storeu_ps((float *) &out[(x0 + 0) * width + y0], row0);
    _mm_storeu_ps((float *) &out[(x0 + 1) * width + y0], row1);
    _mm_storeu_ps((float *) &out[(x0 + 2) * width + y0], row2);
    _mm_storeu_ps((float *) &out[(x0 + 3) * width + y0], row3);
}

void transpose_2d_blocking_intrinsic_unaligned(  size_t width, 
								       size_t height,
								       float * in, 
								       float * out)
{   
    assert(0 < width && 0 < height);
    
    for (size_t i = 0 ; i < height; i += BY)
    {
        for (size_t j = 0 ; j < width ; j += BX)
        {
/*			transpose_4x4(in, out, height, width, i + 0, j + 0);
			transpose_4x4(in, out, height, width, i + 4, j + 0);
			transpose_4x4(in, out, height, width, i + 0, j + 4);
			transpose_4x4(in, out, height, width, i + 4, j + 4);            
*/
			transpose_4x4_unaligned(in, out, height, width, i + 0, j + 0);
			transpose_4x4_unaligned(in, out, height, width, i + 4, j + 0);
			transpose_4x4_unaligned(in, out, height, width, i + 8, j + 0);
			transpose_4x4_unaligned(in, out, height, width, i + 12, j + 0);

        }
    }
}
