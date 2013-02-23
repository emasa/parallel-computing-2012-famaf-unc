	/*******************************************************************************
 Computación Paralela 2012 - FaMAF
 Universidad Nacional de Córdoba

Ancho de banda de memoria en paralelo
=====================================

Ejecutar el programa para distinta cantidad de hilos en diversas máquinas y
tabular los resulados.

OMP_NUM_THREADS=2 ./parallel_membw 1 2 3

¿Qué sucede si quitamos la paralelización por completo, compilamos y corremos?
Explique mirando el assembler porque la versión secuencial no es equivalente a
la versión paralela con OMP_NUM_THREADS=1.

********************************************************************************/

#include <stdio.h>
#include <omp.h>

#define N (1<<26)

float a[N], b[N];

int main(int argc, char **argv)
{
    unsigned int i = 0;
    double start = 0.0;

    start = omp_get_wtime();
    #pragma omp parallel for
    for (i=0; i<N; ++i) {
        b[i] = (float)argc * a[i];
    }
    printf("BW: %f GBps\n", (2*N*sizeof(float))/((1<<30)*(omp_get_wtime()-start)));
    return 0;
}