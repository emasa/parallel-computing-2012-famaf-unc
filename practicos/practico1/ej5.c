
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#define REP(i, n) for(int i=0; i < (n); i++)
#define N_MAX (1 << 26)

const float avg(const float * a, size_t n, int step)
{
    float res = 0.0;
    REP(i, step)
    {
        for (size_t index=i; index < N_MAX ; index += step)
        {
            res += a[index]; 
        }
    }
    return res / n;
}

int main(int argc, char* argv[])
{
    assert(argc == 2);
    int pot = atoi(argv[1]);
    assert(0 <= pot && pot <= 25);
    int step = 1 << pot;     
    
    float* a = (float *) malloc (N_MAX * sizeof(float));
    assert(a != NULL);    
    
    printf("resultado = %f\n", avg(a, N_MAX, step));
    
    return 0;
}
