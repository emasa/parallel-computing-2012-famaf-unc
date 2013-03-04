#define __BSD_SOURCE
#include <math.h>           // fabsf
#include <stdlib.h>         // malloc/free
#include <stdio.h>          // printf
#include <time.h>           // time
#include <sys/time.h>       // gettimeofday, timersub

#include <cuda.h>           // API de cuda
#include <cutil_inline.h>   // Funciones para chequeo de errores


#define N 1024

// dimensiones del bloque
#define BLOCK_WIDTH 32
#define BLOCK_HEIGHT 32


// índice de una coordenada bidimensional de una
// matriz NxN en el arreglo que la almacena
__host__ __device__ __inline__ uint index(uint y, uint x) {
    return x + y * N;
}


// multiplicación trivial de dos matrices NxN
// asume que N es divisible por las dimensiones
// del bloque para simplificar el código
__global__ void mm_slow(const float * a, const float * b, float * c) {
    
    __shared__ float shared_a[BLOCK_HEIGHT][BLOCK_WIDTH];
        
    uint x = threadIdx.x + blockIdx.x * blockDim.x;
    uint y = threadIdx.y + blockIdx.y * blockDim.y;
    
    shared_a[threadIdx.y][threadIdx.x] = a[index(y, x)];
        
    __syncthreads();

    float sum = 0.0f;
    for (uint i = 0; i < N; ++i) {
        sum += ((float*)shared_a)[i] * b[index(i, x)];
    }
    
    c[index(y, x)] = sum;

    __syncthreads();
}


// implementación trivial ikj en CPU de referencia
// con algo de suerte el compilador vectoriza
static void mm_cpu(const float * a, const float * b, float * c) {
    for (uint y = 0; y < N; ++y) {
        for (uint x = 0; x < N; ++x) {
            c[index(y,x)] = 0.0f;
        }
        for (uint k = 0; k < N; ++k) {
            for (uint x = 0; x < N; ++x) {
                c[index(y, x)] += a[index(y, k)] * b[index(k, x)];
            }
        }
    }
}


// comprobar dos resultados y listar diferencias significativas
static void check_result(const float * reference, const float * other) {
    for (uint y = 0; y < N; ++y) {
        for (uint x = 0; x < N; ++x) {
            if (fabsf(reference[index(y, x)] - other[index(y, x)]) > 0.001f) {
                printf("y:%u x:%u reference:%f result:%f\n", y, x, reference[index(y, x)], other[index(y, x)]);
            }
        }
    }
}


int main(int argc, char *argv[]) {
    // pedir memoria en el host
    size_t matrix_size = N * N * sizeof(float);
    float * host_a = (float *) malloc(matrix_size);
    float * host_b = (float *) malloc(matrix_size);
    float * host_c = (float *) malloc(matrix_size);
    float * host_c_reference = (float *) malloc(matrix_size);

    // llenar A y B con numeros aleatorios
    srand(time(0));
    for (uint y = 0; y < N; ++y) {
        for (uint x = 0; x < N; ++x) {
            host_a[index(y, x)] = (float) rand() / RAND_MAX;
            host_b[index(y, x)] = (float) rand() / RAND_MAX;
        }
    }

    // correr en CPU y tomar el tiempo
    struct timeval start, finish, elapsed;
    double cpusecs;
    gettimeofday(&start, NULL);
    mm_cpu(host_a, host_b, host_c_reference);
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

    // configurar la grilla y lanzar el kernel
    dim3 block(BLOCK_WIDTH, BLOCK_HEIGHT);
    dim3 grid(N/block.x, N/block.y);
    mm_slow<<<grid, block>>>(dev_a, dev_b, dev_c);

    // esperar que termine
    cutilSafeCall(cudaDeviceSynchronize());

    // Copiar datos al host y verificar la validez del resultado
    cutilSafeCall(cudaMemcpy(host_c, dev_c, matrix_size, cudaMemcpyDefault));
    check_result(host_c_reference, host_c);

    // liberar memoria
    free(host_a);
    free(host_b);
    free(host_c);
    free(host_c_reference);
    cutilSafeCall(cudaFree(dev_a));
    cutilSafeCall(cudaFree(dev_b));
    cutilSafeCall(cudaFree(dev_c));
    return 0;
}
