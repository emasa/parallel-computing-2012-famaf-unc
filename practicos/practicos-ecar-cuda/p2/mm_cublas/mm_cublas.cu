#define __BSD_SOURCE
#include <math.h>           // fabsf
#include <stdlib.h>         // malloc/free
#include <stdio.h>          // printf
#include <time.h>           // time
#include <sys/time.h>       // gettimeofday, timersub

#include <cuda.h>           // API de cuda
#include <cutil_inline.h>   // Funciones para chequeo de errores

#include <cublas_v2.h>      // cublasSgemm

#define N 1024


// índice de una coordenada bidimensional de una
// matriz NxN en el arreglo que la almacena
// (en formato column-major)
__host__ __device__ __inline__ uint index_cm(uint y, uint x) {
    return x * N + y;
}


// implementación trivial ikj en CPU de referencia
// con algo de suerte el compilador vectoriza
static void mm_cpu_columnmajor(const float * a, const float * b, float * c) {
    for (uint x = 0; x < N; ++x) {
        for (uint y = 0; y < N; ++y) {
            c[index_cm(y,x)] = 0.0f;
        }
        for (uint k = 0; k < N; ++k) {
            for (uint y = 0; y < N; ++y) {
                c[index_cm(y, x)] += a[index_cm(y, k)] * b[index_cm(k, x)];
            }
        }
    }
}


// comprobar dos resultados y listar diferencias significativas
static void check_result_cm(const float * reference, const float * other) {
    for (uint x = 0; x < N; ++x) {
        for (uint y = 0; y < N; ++y) {
            if (fabsf(reference[index_cm(y, x)] - other[index_cm(y, x)]) > 0.001f) {
                printf("y:%u x:%u reference:%f result:%f\n", y, x, reference[index_cm(y, x)], other[index_cm(y, x)]);
            }
        }
    }
}


int main(int argc, char *argv[]) {
    // inicializar cublas
    cublasHandle_t handle;
    // FALTA: cublasCreate

    // pedir memoria en el host
    size_t matrix_size = N * N * sizeof(float);
    float * host_a = (float *) malloc(matrix_size);
    float * host_b = (float *) malloc(matrix_size);
    float * host_c = (float *) malloc(matrix_size);
    float * host_c_reference = (float *) malloc(matrix_size);

    // llenar A y B con numeros aleatorios
    srand(time(0));
    for (uint x = 0; x < N; ++x) {
        for (uint y = 0; y < N; ++y) {
            host_a[index_cm(y, x)] = (float) rand() / RAND_MAX;
            host_b[index_cm(y, x)] = (float) rand() / RAND_MAX;
        }
    }

    // correr en CPU y tomar el tiempo
    struct timeval start, finish, elapsed;
    double cpusecs;
    gettimeofday(&start, NULL);
    mm_cpu_columnmajor(host_a, host_b, host_c_reference);
    gettimeofday(&finish, NULL);
    timersub(&finish, &start, &elapsed);
    cpusecs = elapsed.tv_sec + elapsed.tv_usec / 1000000.0;
    printf("CPU time: %f\n", cpusecs);

    // pedir memoria en la GPU para A, B y C
    float * dev_a;
    float * dev_b;
    float * dev_c;
    cutilSafeCall(cudaMalloc((void **) &dev_a, matrix_size));
    cutilSafeCall(cudaMalloc((void **) &dev_b, matrix_size));
    cutilSafeCall(cudaMalloc((void **) &dev_c, matrix_size));

    // copiar A y B al device
    cutilSafeCall(cudaMemcpy(dev_a, host_a, matrix_size, cudaMemcpyDefault));
    cutilSafeCall(cudaMemcpy(dev_b, host_b, matrix_size, cudaMemcpyDefault));

    // FALTA: llamar a cublasSgemm

    // Copiar datos al host y verificar la validez del resultado
    cutilSafeCall(cudaMemcpy(host_c, dev_c, matrix_size, cudaMemcpyDefault));
    check_result_cm(host_c_reference, host_c);

    // liberar memoria
    free(host_a);
    free(host_b);
    free(host_c);
    free(host_c_reference);
    cutilSafeCall(cudaFree(dev_a));
    cutilSafeCall(cudaFree(dev_b));
    cutilSafeCall(cudaFree(dev_c));

    // FALTA: llamar a cublasDestroy
    return 0;
}
