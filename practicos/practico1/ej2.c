
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include "timing.h"

#define N_MAX (1 << 29)

const float avg(const float * a, size_t n)
{
    float res = 0.0;
    for (size_t i = 0; i < n; i++) {
        res += a[i];
    }
    return res / n;
}

int main(int argc, char* argv[])
{
    float* a = (float *) malloc(N_MAX * sizeof (float));
    assert (a != NULL);
    
    for (int i = 0; i <= 9; i++)
    {
        long int flops = ( 1 << (20 + i) );

        double t_start = wtime();
        float result = avg(a, flops);
        double t_total = wtime() - t_start;
                
        printf("N=[2^%i], segs=%f, gflops/seg=%f, MB/segs=%f, result=%f\n", 
                20+i, t_total, flops * 10e-9 / t_total, flops * sizeof(float) * 10e-6 / t_total, result);
    }
    free(a);
    
    return 0;
}
