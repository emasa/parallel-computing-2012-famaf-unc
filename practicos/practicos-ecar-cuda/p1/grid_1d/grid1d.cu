#include <cuda.h>           // API de CUDA
#include <cutil_inline.h>   // macros para verificar llamadas
#include <stdio.h>
#include <stdlib.h>

#include "../../common/colorstuff.h"      // conversión de [0,1] a color
#include "../../common/sdlstuff.h"       // gráficos

// Longitud de la escala de colores en pixels
// Ojo! No es divisible por el tamaño de bloque
#define GRADIENT_SIZE 719

// tamaño de la escala en bytes
#define GRADIENT_BYTES (GRADIENT_SIZE * sizeof(rgba))

// longitud del bloque de threads
#define BLOCK_SIZE 32

// altura de la imagen (para que sea fácil de ver)
// la escala generada se dibuja completa una vez por fila
#define IMAGE_HEIGHT 32

// tamaño de la imagen
#define IMAGE_SIZE (GRADIENT_SIZE * IMAGE_HEIGHT)

#define DIV_CEIL(n, m) ((n) + (m) - 1) / (m)

// kernel que pinta bloques de distintos colores
__global__ void gradient_1d(rgba * image, size_t n) {

    // indice del thread en el vector
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        // posicion normalizada del bloque en la grilla [0..1]
        float norm_pos = blockIdx.x / (gridDim.x - 1.0f);
        // convertir a una escala de colores
        rgba color = color1(norm_pos);
        // escribir a memoria
        image[idx] = color;
    }
}


int main(int argc, char ** argv) {

    // pedir memoria en el host para la imagen
    rgba * host_gradient = (rgba *) malloc(GRADIENT_BYTES);

    // pedir memoria en la placa para la imagen e inicializarla con ceros
    rgba * device_gradient;
    // FALTA: pedir memoria en la placa
    cutilSafeCall(cudaMalloc(&device_gradient, GRADIENT_BYTES));

    // FALTA: inicializar a 0 la memoria en la placa
    cutilSafeCall(cudaMemset(device_gradient, 0, GRADIENT_BYTES));
    
    // correr kernel
    dim3 block(BLOCK_SIZE);         // bloque
    // FALTA: definir el tamaño del grid de acuerdo al tamaño del bloque
    dim3 grid(DIV_CEIL(GRADIENT_SIZE, BLOCK_SIZE));
    // FALTA: llamar al kernel
    gradient_1d<<<grid, block>>>(device_gradient, GRADIENT_SIZE);

    // verificar errores
    cutilCheckMsg("Fallo al lanzar el kernel:");

    // FALTA: esperar a que el kernel termine
    cutilSafeCall(cudaDeviceSynchronize());

    // FALTA: copiar la imagen al host
    cutilSafeCall(cudaMemcpy(host_gradient, device_gradient, GRADIENT_BYTES, cudaMemcpyDefault));    
    
    // inicializar gráficos, dibujar en pantalla
    sdls_init(GRADIENT_SIZE, IMAGE_HEIGHT);
    for (uint i = 0; i < IMAGE_HEIGHT; ++i) {
        sdls_blitrectangle_rgba(0, i, GRADIENT_SIZE, 1, host_gradient);
    }
    sdls_draw();

    // esperar input para salir
    printf("<ENTER> para salir\n");
    getchar();

    // limpieza de memoria
    free(host_gradient);

    // FALTA: liberar memoria de la placa
    cutilSafeCall(cudaFree(device_gradient));

    return 0;
}
