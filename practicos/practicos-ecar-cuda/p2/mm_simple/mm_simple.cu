#define __BSD_SOURCE
#include <math.h>           // fabsf
#include <stdlib.h>         // malloc/free
#include <stdio.h>          // printf
#include <time.h>           // time
#include <sys/time.h>       // gettimeofday, timersub

#include <cuda.h>           // API de cuda
#include <cutil_inline.h>   // Funciones para chequeo de errores


// matrices NxN
#define N 1024

// dimensiones del bloque de threads
#define BLOCK_WIDTH 64
#define BLOCK_HEIGHT 8
#define MATRIX_BYTES N * N * sizeof(float)

// índice de una coordenada bidimensional de una
// matriz NxN en el arreglo que la almacena
__host__ __device__ __inline__ uint index(uint y, uint x) {
    return x + y * N;
}


// multiplicación trivial de dos matrices NxN
// asume que N es divisible por las dimensiones
// del bloque para simplificar el código
__global__ void mm_simple(const float * a, const float * b, float * c) {

    size_t x = blockIdx.x * blockDim.x + threadIdx.x; 
    size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < N && y < N){
	c[index(y, x)] = 0.0f;
	for (size_t k = 0; k < N; k++){
	    c[index(y, x)] += a[index(y, k)] * b[index(k, x)];
	}
    }
}


// multiplicacion de referencia en cpu
void mm_cpu(const float * a, const float * b, float * c) {
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


// compara resultados
void check_result(const float * reference, const float * other) {
    for (uint y = 0; y < N; ++y) {
        for (uint x = 0; x < N; ++x) {
            if (fabsf(reference[index(y, x)] - other[index(y, x)]) > 0.001f) {
                printf("y:%u x:%u reference:%f result:%f\n", y, x, reference[index(y, x)], other[index(y, x)]);
            }
        }
    }
}


int main(int argc, char *argv[]) {

    // pedir memoria
    size_t matrix_size = N * N * sizeof(float);
    float * host_a = (float *) malloc(matrix_size);
    float * host_b = (float *) malloc(matrix_size);
    float * host_c = (float *) malloc(matrix_size);
    float * host_c_reference = (float *) malloc(matrix_size);

    // inicializar con valores aleatorios
    srand(time(0));
    for (uint y = 0; y < N; ++y) {
        for (uint x = 0; x < N; ++x) {
            host_a[index(y, x)] = (float) rand() / RAND_MAX;
            host_b[index(y, x)] = (float) rand() / RAND_MAX;
        }
    }

    // correr en cpu y tomar el tiempo
    struct timeval start, finish, elapsed;
    gettimeofday(&start, NULL);
    mm_cpu(host_a, host_b, host_c_reference);
    gettimeofday(&finish, NULL);
    timersub(&finish, &start, &elapsed);
    double cpusecs = elapsed.tv_sec + elapsed.tv_usec / 1000000.0;
    printf("CPU time: %f\n", cpusecs);


    // pedir memoria en la gpu
    float * dev_a;
    float * dev_b;
    float * dev_c;
    // FALTA: pedir las 3 en la placa!
    cutilSafeCall(cudaMalloc(&dev_a, matrix_size));
    cutilSafeCall(cudaMalloc(&dev_b, matrix_size));
    cutilSafeCall(cudaMalloc(&dev_c, matrix_size));
    // copiar las matrices a gpu
    // FALTA: copiar A y B!
    cutilSafeCall(cudaMemcpy(dev_a, host_a, matrix_size, cudaMemcpyDefault)); 
    cutilSafeCall(cudaMemcpy(dev_b, host_b, matrix_size, cudaMemcpyDefault));

    // llamar al kernel
    dim3 block(BLOCK_WIDTH, BLOCK_HEIGHT);
    dim3 grid(N/block.x, N/block.y);    // asumiendo que N % block.{x,y} == 0
    // FALTA: llamada al kernel!
    mm_simple<<<grid, block>>> (dev_a, dev_b, dev_c);
    // FALTA: sincronizar!
    cudaDeviceSynchronize();
    // FALTA: copiar C al host! 
    cutilSafeCall(cudaMemcpy(host_c, dev_c, matrix_size, cudaMemcpyDefault));
    
    // ver que no nos mandemos ninguna macana
    check_result(host_c_reference, host_c);

    // liberar memoria
    free(host_a);
    free(host_b);
    free(host_c);
    free(host_c_reference);
    // FALTA: liberar memoria de la placa
    cutilSafeCall(cudaFree(dev_a));
    cutilSafeCall(cudaFree(dev_b));
    cutilSafeCall(cudaFree(dev_c));

    return 0;

}
