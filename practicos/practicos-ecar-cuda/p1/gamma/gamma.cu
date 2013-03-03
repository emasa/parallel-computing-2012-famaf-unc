#include <cuda.h>           // API de CUDA
#include <cutil_inline.h>   // macros para verificar llamadas
#include <stdio.h>
#include <stdlib.h>

#include "../../common/colorstuff.h"      // conversión de [0,1] a color
#include "../../common/sdlstuff.h"       // gráficos


// dimensiones de los bloques de threads
#define BLOCK_WIDTH 32
#define BLOCK_HEIGHT 32


// aplica corrección de gamma a un canal
__device__ unsigned char channelgamma(unsigned char orig, float gamma) {
    return (unsigned char) min(255u, (uint) rintf(powf(orig, 1.0f / gamma)));
}


// aplica corrección de gamma a un pixel
__device__ rgba applygamma(rgba orig, float gamma) {
    rgba result;
    result.r = channelgamma(orig.r, gamma);
    result.g = channelgamma(orig.g, gamma);
    result.b = channelgamma(orig.b, gamma);
    result.a = orig.a;
    return result;
}


// kernel que aplica la corrección de gamma
__global__ void gamma_kernel(rgba * image, size_t width, size_t height, float gamma) {

    // FALTA: calcular la columna del thread
    size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    // FALTA: calcular la fila del thread
    size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    // FALTA: pasar (x,y) a un índice en el array de memoria
    size_t idx = y * width + x;

    // FALTA: verificar que el thread opere sobre una columna y fila válida
    if ( x < width && y < height){
        // FALTA: leer el pixel que le corresponde al thread
        rgba orig = image[idx];
        // FALTA: aplicarle la corrección de gamma especificada
        rgba result = applygamma(orig, gamma);
        // FALTA: escribir el pixel corregido
        image[idx] = result;
    }
}


int main(int argc, char ** argv) {

    if (argc != 3) {
        fprintf(stderr, "Uso: gamma IMAGEN GAMMA\n");
        exit(1);
    }
    float gamma = atof(argv[2]);
    if (gamma == 0.0) {
        fprintf(stderr, "Gamma inválido\n");
        exit(1);
    }

    // leer imagen de disco
    size_t width, height;
    rgba * image_before = sdls_loadimage_rgba(argv[1], &width, &height);
    if (!image_before) {
        fprintf(stderr, "No se pudo leer la imagen\n");
        exit(1);
    }

    // pedir memoria para la imagen destino
    size_t image_bytes = width * height * sizeof(rgba);
    rgba * image_after = (rgba *) malloc(image_bytes);

    // pedir memoria en la placa para la imagen y copiar el original
    rgba * device_image;
    cutilSafeCall(cudaMalloc(&device_image, image_bytes));
    cutilSafeCall(cudaMemcpy(device_image, image_before, image_bytes, cudaMemcpyDefault));

    // correr kernel
    dim3 block(BLOCK_WIDTH, BLOCK_HEIGHT);                   // bloque
    dim3 grid((width + BLOCK_WIDTH - 1) / BLOCK_WIDTH,
              (height + BLOCK_HEIGHT - 1) / BLOCK_HEIGHT);  // grilla
    gamma_kernel<<<grid, block>>>(device_image, width, height, gamma);

    // verificar errores
    cutilCheckMsg("Fallo al lanzar el kernel:");

    // esperar a que el kernel termine
    cutilSafeCall(cudaDeviceSynchronize());

    // copiar la imagen procesada al host
    cutilSafeCall(cudaMemcpy(image_after, device_image, image_bytes, cudaMemcpyDefault));


    // inicializar gráficos, dibujar en pantalla
    sdls_init(width * 2, height);
    sdls_blitrectangle_rgba(0, 0, width, height, image_before);
    sdls_blitrectangle_rgba(width, 0, width, height, image_after);
    sdls_draw();

    // esperar input para salir
    printf("<ENTER> para salir\n");
    getchar();

    // limpieza de memoria
    free(image_before);
    free(image_after);
    cutilSafeCall(cudaFree(device_image));

    return 0;
}
