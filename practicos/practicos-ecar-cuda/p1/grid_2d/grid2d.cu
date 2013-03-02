#include <cuda.h>           // API de CUDA
#include <cutil_inline.h>   // macros para verificar llamadas
#include <stdio.h>
#include <stdlib.h>

#include "../../common/colorstuff.h"      // conversión de [0,1] a color
#include "../../common/sdlstuff.h"       // gráficos

// Dimensiones de la imagen
// Ojo! No es divisible por el tamaño de bloque
#define IMAGE_WIDTH 563
#define IMAGE_HEIGHT 563

// tamaño de la imagen en bytes
#define IMAGE_BYTES (IMAGE_WIDTH * IMAGE_HEIGHT * sizeof(rgba))

// dimensiones de los bloques de threads
#define BLOCK_WIDTH 32
#define BLOCK_HEIGHT 32

#define DIV_CEIL(n, m) ((n) + (m) -1) / (m)

//dimensiones de la grilla
#define GRID_WIDTH DIV_CEIL(IMAGE_WIDTH, BLOCK_WIDTH)
#define GRID_HEIGHT DIV_CEIL(IMAGE_HEIGHT, BLOCK_HEIGHT)

// kernel que pinta los bloques de un grid 2D de distintos colores
__global__ void block_gradient_2d(rgba * image, size_t width, size_t height) {

    // columna del thread
    size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    // fila del thread
    size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    // indice en el array de memoria
    size_t idx = y * width + x;

    if (x < width && y < height) {
        // posición normalizada del bloque en cada coordenada [0..1]
        float norm_x = blockIdx.x / (gridDim.x - 1.0f);
        float norm_y = blockIdx.y / (gridDim.y - 1.0f);

        // convertir a color
        rgba color = color2(norm_x, norm_y);

        // escribir color a memoria
        image[idx] = color;
    }
}


int main(int argc, char ** argv) {

    // pedir memoria en el host para la imagen
    rgba * host_image = (rgba *) malloc(IMAGE_BYTES);

    // pedir memoria en la placa para la imagen e inicializarla con ceros
    rgba * device_image;
    // FALTA: pedir memoria en la placa
    cutilSafeCall(cudaMalloc(&device_image, IMAGE_BYTES));
    // FALTA: inicializar a 0 la memoria de la placa
    cutilSafeCall(cudaMemset(device_image, 0, IMAGE_BYTES));
    // correr kernel
    dim3 block(BLOCK_WIDTH, BLOCK_HEIGHT);                      // bloque
    // FALTA: definir el tamaño del grid
    dim3 grid(GRID_WIDTH, GRID_HEIGHT); 
    // FALTA: llamar al kernel
    block_gradient_2d<<<grid, block>>>(device_image, IMAGE_WIDTH, IMAGE_HEIGHT);

    // verificar errores
    cutilCheckMsg("Fallo al lanzar el kernel:");

    // FALTA: esperar a que el kernel termine
    cutilSafeCall(cudaDeviceSynchronize());

    // FALTA: copiar la imagen al host
    cutilSafeCall(cudaMemcpy(host_image, device_image, IMAGE_BYTES, cudaMemcpyDefault));

    // inicializar gráficos, dibujar en pantalla
    sdls_init(IMAGE_WIDTH, IMAGE_HEIGHT);
    sdls_blitrectangle_rgba(0, 0, IMAGE_WIDTH, IMAGE_HEIGHT, host_image);
    sdls_draw();

    // esperar input para salir
    printf("<ENTER> para salir\n");
    getchar();

    // limpieza de memoria
    free(host_image);
    cutilSafeCall(cudaFree(device_image));

    return 0;
}
