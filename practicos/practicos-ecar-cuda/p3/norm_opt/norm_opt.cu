#include <cuda.h>
#include <cutil_inline.h>
#include <stdio.h>
#include <stdlib.h>

#include "../../common/colorstuff.h"
#include "../../common/sdlstuff.h"

#define BLOCK_X 16
#define BLOCK_Y 16
#define BLOCK_SIZE ((BLOCK_X) * (BLOCK_Y))


// helper
uint divceil(uint a, uint b) {
    return (a + b - 1) / b;
}


// normaliza un punto
__device__ grayscale normalize(grayscale value, grayscale old_min, grayscale old_max) {
    grayscale new_min = 0;
    grayscale new_max = 255;
    return (grayscale) ((float) (value - old_min) * (new_max - new_min) / (old_max - old_min)) + new_min;
}


// kernel que normaliza una imagen de acuerdo a los valores
// maximos y minimos apuntados por min y max
__global__ void norm(const grayscale * in, const uint * max, const uint * min, size_t width, size_t height, grayscale * out) {
    uint x = blockIdx.x * blockDim.x + threadIdx.x;
    uint y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height) {
        size_t idx = x + y * width;
        out[idx] = normalize(in[idx], (grayscale) *min, (grayscale) *max);
    }
}


// reducir el minimo en arbol usando memoria compartida
__global__ void max_tree(const grayscale * image, size_t width, size_t height, uint * result) {
    uint x = blockIdx.x * blockDim.x + threadIdx.x;
    uint y = blockIdx.y * blockDim.y + threadIdx.y;
    uint idx = x + y * width;

    __shared__ uint tmp[BLOCK_SIZE];
    uint tid = threadIdx.y * blockDim.x + threadIdx.x;
    if (x < width && y < height) {
        tmp[tid] = image[idx];
    } else {
        tmp[tid] = 0;
    }

    __syncthreads();

    for (uint distance = 1; distance <= BLOCK_SIZE / 2; distance = distance * 2) {
        if ( tid % distance * 2 == 0) {
            tmp[tid] = max(tmp[tid], tmp[tid+distance]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicMax(result, tmp[0]);
    }
}


// reducir el minimo de la forma más sencilla
__global__ void min_atomic(const grayscale * image, size_t width, size_t height, uint * result) {
    uint x = blockIdx.x * blockDim.x + threadIdx.x;
    uint y = blockIdx.y * blockDim.y + threadIdx.y;
    size_t idx = x + y * width;

    if (x < width && y < height) {
        atomicMin(result, image[idx]);
    }
}


int main(int argc, char ** argv) {

    if (argc != 2) {
        fprintf(stderr, "Error: normalize [nombre de imagen]\n");
        exit(1);
    }

    // cargar imagen
    grayscale * host_orig;
    size_t width;
    size_t height;
    host_orig = sdls_loadimage_grayscale(argv[1], &width, &height);
    if (host_orig == 0) {
        fprintf(stderr, "Error: Problemas abriendo la imagen\n");
        exit(1);
    }

    // copiar al device
    grayscale * dev_orig;
    cutilSafeCall(cudaMalloc(&dev_orig, width * height * sizeof(grayscale)));
    cutilSafeCall(cudaMemcpy(dev_orig, host_orig, width * height * sizeof(grayscale), cudaMemcpyDefault));

    // lugar para la imagen normalizada
    grayscale * dev_norm;
    cutilSafeCall(cudaMalloc(&dev_norm, width * height * sizeof(grayscale)));

    // pedir lugar para mínimo e inicializar al máximo valor
    uint * dev_min;
    cutilSafeCall(cudaMalloc(&dev_min, sizeof(uint)));
    cutilSafeCall(cudaMemset(dev_min, 0xff, sizeof(uint)));

    // pedir lugar para máximo e inicializar al mínimo valor
    uint * dev_max;
    cutilSafeCall(cudaMalloc(&dev_max, sizeof(uint)));
    cutilSafeCall(cudaMemset(dev_max, 0, sizeof(uint)));

    // buscar mínimo y máximo para poder normalizar
    dim3 block(BLOCK_X, BLOCK_Y);
    dim3 grid(divceil(width, block.x), divceil(height, block.y));
    min_atomic<<<grid, block>>>(dev_orig, width, height, dev_min);
    max_tree<<<grid, block>>>(dev_orig, width, height, dev_max);
    cutilCheckMsg("Error en reducción: ");

    // normalizar la imagen
    norm<<<grid, block>>>(dev_orig, dev_max, dev_min, width, height, dev_norm);
    cutilCheckMsg("Error en norm: ");
    cutilSafeCall(cudaDeviceSynchronize());


    // imprimir resultados para verificacion
    uint host_min, host_max;
    cutilSafeCall(cudaMemcpy(&host_min, dev_min, sizeof(uint), cudaMemcpyDefault));
    cutilSafeCall(cudaMemcpy(&host_max, dev_max, sizeof(uint), cudaMemcpyDefault));
    printf("Min: %u\nMax: %u\n", host_min, host_max);

    // copiar imagen normalizada al host
    grayscale * host_norm = (grayscale *) malloc(width * height * sizeof(grayscale));
    cutilSafeCall(cudaMemcpy(host_norm, dev_norm, width * height * sizeof(grayscale), cudaMemcpyDefault));

    // inicializar sdl
    sdls_init(width, height);

    // dibujar en pantalla
    sdls_blitrectangle_grayscale(0, 0, width, height, host_norm);
    sdls_draw();

    printf("<ENTER> para salir\n");
    getchar();

    // limpieza
    free(host_orig);
    free(host_norm);
    cutilSafeCall(cudaFree(dev_orig));
    cutilSafeCall(cudaFree(dev_norm));
    cutilSafeCall(cudaFree(dev_min));
    cutilSafeCall(cudaFree(dev_max));

    return 0;
}
