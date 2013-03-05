#define __BSD_SOURCE
#include <math.h>           // fabsf
#include <stdlib.h>         // malloc/free
#include <stdio.h>          // printf
#include <time.h>           // time
#include <sys/time.h>       // gettimeofday, timersub

#include <cuda.h>           // API de cuda
#include <cutil.h>

#include <cutil_inline.h>   // Funciones para chequeo de errores

#define N 1024

// índice de una coordenada bidimensional de una
// matriz NxN en el arreglo que la almacena
__host__ __device__ __inline__ uint index(uint y, uint x) {
    return x + y * N;
}


// multiplicación trivial de dos matrices NxN
// asume que N es divisible por las dimensiones
// del bloque para simplificar el código

__global__ void mm_dynamic_block(const float * a, const float * b, float * c) { 
    
    uint x = blockIdx.x * blockDim.x + threadIdx.x;
    uint y = blockIdx.y * blockDim.y + threadIdx.y;

    uint block_width  = blockDim.x;

    if (x < N && y < N){
        // matriz compartida donde guardamos parte de A temporalmente
        extern __shared__ float tmp_a[];

        // acumulador temporal del resultado
        float result = 0.0f;

        // avanzamos de a bloques
        for (uint i = 0; i < N; i += block_width) {
            // copiar todo el bloque de A a memoria compartida
            tmp_a[threadIdx.y * block_width + threadIdx.x] = a[index(y, i + threadIdx.x)];
            // esperar que todos los threads hayan copiado su valor
            __syncthreads();
            // actualizar result para los valores de A que tenemos en shared
            for (uint j = 0; j < block_width; ++j) {
                result += tmp_a[threadIdx.y * block_width + j] * b[index(i + j, x)];
            }
            // esperar que todos los threads terminen antes de sobreescribir tmp_a
            __syncthreads();
        }
        // guardar el resultado final
        c[index(y,x)] = result;
    }
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
    
    float kernelTime;
    for (uint block_width = 1; block_width <= N; block_width *= 2) {
        for (uint block_height = 1; block_height <= N; block_height *= 2) {
            if (block_width * block_height <= N){
                
                // configurar la grilla y lanzar el kernel
                dim3 block(block_width, block_height);
                dim3 grid(N/block.x, N/block.y);
                
                // creo eventos
                cudaEvent_t start, end;
                cutilSafeCall(cudaEventCreate(&start));
                cutilSafeCall(cudaEventCreate(&end));
                
                // disparo eventos (se ejecutan en el sm 0) y kernel 
                cutilSafeCall(cudaEventRecord(start, 0));
                mm_dynamic_block<<<grid, block, block_width*block_height>>>(dev_a, dev_b, dev_c);
                cutilSafeCall(cudaEventRecord(end, 0));
                
                // esperar a que termine el kernel
                cutilSafeCall(cudaDeviceSynchronize());
                cutilCheckMsg("shared_a");
                
                // espero a que termine el evento end 
                cutilSafeCall(cudaEventSynchronize(end));
                
                // Copiar datos al host y verificar la validez del resultado
                cutilSafeCall(cudaMemcpy(host_c, dev_c, matrix_size, cudaMemcpyDefault));
                check_result(host_c_reference, host_c);
                
                // calculo tiempo transcurrido
                cutilSafeCall(cudaEventElapsedTime(&kernelTime, start, end));
                printf ("%i %i %f\n", block_height, block_width, kernelTime);
                
                // destruyo eventos
                cutilSafeCall(cudaEventDestroy(start));
                cutilSafeCall(cudaEventDestroy(end));                
            }
        }
    }
 
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
