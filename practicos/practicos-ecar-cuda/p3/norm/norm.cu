#include <cuda.h>
#include <cutil_inline.h>
#include <stdio.h>
#include <stdlib.h>

#include "../../common/colorstuff.h"
#include "../../common/sdlstuff.h"

#define BLOCK_X 16
#define BLOCK_Y 16
#define INF (1 << 30);

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


// Calcular el maximo o minimo valor de la imagen
typedef enum { REDUCE_MIN, REDUCE_MAX } reduce_op;

// CUDA es C++, podemos usar templates en kernels
// para pasar parametros en tiempo de compilacion
template<reduce_op op>
__global__ void reduce(const grayscale * image, size_t width, size_t height, uint * result) {
    uint x = blockIdx.x * blockDim.x + threadIdx.x;
    uint y = blockIdx.y * blockDim.y + threadIdx.y;
    size_t idx = x + y * width;

    __shared__ uint tmp;
    //solo un hilo por bloque inicializa
    if (threadIdx.x == 0 && threadIdx.y == 0){
	if (op == REDUCE_MIN) {
           tmp = INF; 
        } else {
           tmp = 0;
        }
    }     
    __syncthreads();
    
    if (x < width && y < height) {
        if (op == REDUCE_MIN) {
            atomicMin(&tmp, image[idx]);
        } else {
            atomicMax(&tmp, image[idx]);
        }
    }
    __syncthreads();
    
    if (threadIdx.x == 0 && threadIdx.y == 0){
	if (op == REDUCE_MIN) {
            atomicMin(result, tmp);
        } else {
            atomicMax(result, tmp);
        }
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
    reduce<REDUCE_MIN><<<grid, block>>>(dev_orig, width, height, dev_min);
    reduce<REDUCE_MAX><<<grid, block>>>(dev_orig, width, height, dev_max);
    CUT_CHECK_ERROR("Error en reducción: ");

    // normalizar la imagen
    norm<<<grid, block>>>(dev_orig, dev_max, dev_min, width, height, dev_norm);
    CUT_CHECK_ERROR("Error en norm: ");
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
    sdls_init(2 * width, height);

    // dibujar en pantalla
    sdls_blitrectangle_grayscale(0, 0, width, height, host_orig);
    sdls_blitrectangle_grayscale(width, 0, width, height, host_norm);
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
