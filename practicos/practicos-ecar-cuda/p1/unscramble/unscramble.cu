#include <cuda.h>           // API de CUDA
#include <cutil_inline.h>   // macros para verificar llamadas
#include <stdio.h>
#include <stdlib.h>

#include "../../common/colorstuff.h"      // conversión de [0,1] a color
#include "../../common/sdlstuff.h"       // gráficos


// dimensiones de los bloques de threads
#define BLOCK_WIDTH 16
#define BLOCK_HEIGHT 16


// deofuscar un pixel
__device__ rgba unscramble(rgba scrambled, uint x, uint y) {
    // flip some channels
    if (y % 8 == 0) {
        // nada
    } else if (y % 8 == 1) {
        scrambled.r = 255u - scrambled.r;
    } else if (y % 8 == 2) {
        scrambled.r = 255u - scrambled.r;
        scrambled.g = 255u - scrambled.g;
    } else if (y % 8 == 3) {
        scrambled.r = 255u - scrambled.r;
        scrambled.g = 255u - scrambled.g;
        scrambled.b = 255u - scrambled.b;
    } else if (y % 8 == 4) {
        scrambled.g = 255u - scrambled.g;
        scrambled.b = 255u - scrambled.b;
    } else if (y % 8 == 5) {
        scrambled.g = 255u - scrambled.g;
    } else if (y % 8 == 6) {
        scrambled.b = 255u - scrambled.b;
    } else if (y % 8 == 7) {
        scrambled.r = 255u - scrambled.r;
        scrambled.b = 255u - scrambled.b;
    }

    rgba result;
    // shuffle color channels
    unsigned char * scrambled_as_vector = (unsigned char *) &scrambled;
    unsigned char * result_as_vector = (unsigned char *) &result;
    for (uint i = 0; i < 3; ++i) {
        result_as_vector[i] = scrambled_as_vector[(x + i) % 3];
    }

    result.a = scrambled.a;
    return result;
}


// kernel que aplica unscramble
__global__ void unscramble_kernel(rgba * image, size_t width, size_t height) {

    // columna del thread
    size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    // fila del thread
    size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    // indice en el array de memoria
    size_t idx = y * width + x;

    if (x < width && y < height) {
        // escribir color a memoria
        rgba unscrambledpixel = unscramble(image[idx], x, y);
        image[idx] = unscrambledpixel;
    }
}


int main(int argc, char ** argv) {

    if (argc != 2) {
        fprintf(stderr, "Uso: unscramble IMAGEN\n");
        exit(1);
    }

    // leer imagen de disco
    size_t width, height;
    rgba * image_scrambled = sdls_loadimage_rgba(argv[1], &width, &height);
    if (!image_scrambled) {
        fprintf(stderr, "No se pudo leer la imagen\n");
        exit(1);
    }

    // pedir memoria para la imagen destino
    size_t image_bytes = width * height * sizeof(rgba);
    rgba * image = (rgba *) malloc(image_bytes);

    // pedir memoria en la placa para la imagen y copiar el original
    rgba * device_image;
    // FALTA: pedir memoria para la imagen en la placa
    cutilSafeCall(cudaMalloc(&device_image, image_bytes));
    // FALTA: copiar la imagen desde el host a la placa
    cutilSafeCall(cudaMemcpy(device_image, image_scrambled, image_bytes, cudaMemcpyDefault));
    // correr kernel
    dim3 block(BLOCK_WIDTH, BLOCK_HEIGHT);                  // bloque
    dim3 grid((width + BLOCK_WIDTH - 1) / BLOCK_WIDTH,
              (height + BLOCK_HEIGHT - 1) / BLOCK_HEIGHT);  // grilla
    unscramble_kernel<<<grid, block>>>(device_image, width, height);

    // verificar errores
    cutilCheckMsg("Fallo al lanzar el kernel:");

    // FALTA: esperar a que el kernel termine
    cutilSafeCall(cudaDeviceSynchronize());
    // FALTA: copiar la imagen procesada al host
    cutilSafeCall(cudaMemcpy(image, device_image, image_bytes, cudaMemcpyDefault));

    // inicializar gráficos, dibujar en pantalla
    sdls_init(width, height);
    sdls_blitrectangle_rgba(0, 0, width, height, image);
    sdls_draw();

    // esperar input para salir
    printf("<ENTER> para salir\n");
    getchar();

    // limpieza de memoria
    free(image_scrambled);
    free(image);

    // FALTA: liberar la memoria de la placa
    cudaFree(device_image);
    return 0;
}
