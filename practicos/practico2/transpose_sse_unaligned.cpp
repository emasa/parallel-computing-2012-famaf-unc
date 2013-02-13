#define _BSD_SOURCE

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include <sys/time.h>

#include <ammintrin.h>

#include "transpose_utils.h"
#include "transpose_intrinsic.h"

#define BX 4
#define BY 16

#define HEIGHT (1 << 10)
#define WIDTH HEIGHT

#define TIMES 100
								       
int main()
{
	float * in = (float *) _mm_malloc(sizeof(float) * HEIGHT * WIDTH, 16);
	assert(in != NULL);
	float * out = (float *) _mm_malloc(sizeof(float) * HEIGHT * WIDTH, 16);
    assert(out != NULL);
                        
	init_matrix(WIDTH, HEIGHT, in);
    
    double avg_time = 0.0, avg_gfps = 0.0, avg_gbps=0.0;
    struct timeval start, end, elapsed;
    for (int i = 0; i < TIMES; i++)
    {
        gettimeofday(&start, 0);
        
        transpose_2d_blocking_intrinsic_aligned(WIDTH, HEIGHT, in, out);
        
        gettimeofday(&end, 0);
        timersub(&end, &start, &elapsed);
        
        double time = (double) (elapsed.tv_sec + 1e-6 * elapsed.tv_usec);
        
        avg_time += time;
        avg_gfps += WIDTH * HEIGHT  * 10e-9 / time;
        avg_gbps += sizeof(float) * WIDTH * HEIGHT * 10e-9 / time;
    }

    check_traspose(HEIGHT, WIDTH, out);
    
    avg_time /= 1. * TIMES;
    avg_gfps /= 1. * TIMES;
    avg_gbps /= 1. * TIMES;
                
    printf("BY=%i\tBX=%i\tBY*BX=%i\ttime=%f\tGfps=%.3f\tGBps=%.3f\n", 
            BY, BX, BY * BX, avg_time, avg_gfps, avg_gbps);
    
    _mm_free(in);
    _mm_free(out);
    
	return 0;
}
