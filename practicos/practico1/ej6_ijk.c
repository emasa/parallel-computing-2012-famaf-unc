
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <time.h>

#include "timing.h"

#define REP(i, n) for(size_t i = 0; i < (n) ; i++)
#define AT(a, i, j, n) (a)[(i) * (n) + (j)]

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
            AT(b, i, j, n) = _random();
            AT(c, i, j, n) = 0.0f;
        }
    }
}
void mmulf(const float * a, const float * b, size_t n, float * c)
{
    REP(i, n)
    {
        REP(j, n)
        {
            REP(k, n)
            {
                AT(c, i, j, n) += AT(a, i, k, n) * AT(b, k, j, n);
            }
        }
    }
}

int main(int argc, char* argv[])
{
    assert(argc == 2);
    int size = atoi(argv[1]);
    assert(0 < size && size <= 1024);
 
    float* a = (float *) malloc (sizeof(float) * size * size);
    assert(a != NULL);

    float* b = (float *) malloc (sizeof(float) * size * size);
    assert(b != NULL);

    float* c = (float *) malloc (sizeof(float) * size * size);
    assert(c != NULL);

    //init(size, a, b, c);
    
    double t_start = wtime();
    mmulf(a, b, size, c);
    double t_total = wtime() - t_start;

    double gflops_per_segs = size * size * size * 10e-9 / t_total;
    double gbs_per_segs = 1. * sizeof(float) * size * size * size * 10e-9 / t_total;
    printf("size=%i, tiempo=%f segs, gflops/seg=%f, MB/segs=%f, c[0][0]=%f\n", 
            size, t_total, gflops_per_segs, gbs_per_segs, AT(c, 0, 0, size));
   
    return 0;    
}
