#include <stddef.h>

#include <cuda.h>           // API de cuda
#include <cutil_inline.h>   // Funciones para chequeo de errores

#define IX(i,j) ((i)+(n+2)*(j))
#define SWAP(x0,x) {float * tmp=x0;x0=x;x=tmp;}
#define DIV_CEIL(n, m) ((n) + (m) -1) / (m)

#define BLOCK_WIDTH 32
#define BLOCK_HEIGHT 16

typedef enum { NONE = 0, VERTICAL = 1, HORIZONTAL = 2 } boundary;
typedef enum { RED = 0, BLACK = 1} Color;

__global__ static void add_source_kernel(unsigned int n_total, float * x, const float * s, float dt)
{
    uint i = blockIdx.y * blockDim.y + threadIdx.y;
    uint j = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < n_total && j < n_total){
        uint idx = i * n_total + j;
        x[idx] += dt * s[idx];
    }
}

static void add_source_parallel(uint n, float * x, const float * s, float dt)
{    
    uint n_total = n + 2;
    // dimensiones para add_source_kernel
    dim3 block(BLOCK_WIDTH, BLOCK_HEIGHT);
    dim3 grid(DIV_CEIL(n_total, block.x), DIV_CEIL(n_total, block.y));
       
    add_source_kernel<<<grid, block>>>(n_total, x, s, dt);
    
    CUT_CHECK_ERROR("Error al incrementar source :");
    cutilSafeCall(cudaDeviceSynchronize()); // espero a que termine el kernel
}

static void add_source(uint n, float * x, const float * s, float dt)
{    
    uint total_bytes = (n+2) * (n+2) * sizeof(float);
    float *dev_x = NULL, *dev_s = NULL;

    // reservo memoria en la placa
    cutilSafeCall(cudaMalloc(&dev_x, total_bytes));
    cutilSafeCall(cudaMalloc(&dev_s, total_bytes));
    // copio x y s a la placa
    cutilSafeCall(cudaMemcpy(dev_x, x, total_bytes, cudaMemcpyDefault));
    cutilSafeCall(cudaMemcpy(dev_s, s, total_bytes, cudaMemcpyDefault));
    
    add_source_parallel(n, dev_x, dev_s, dt);

    // traigo x devuelta al host 
    cutilSafeCall(cudaMemcpy(x, dev_x, total_bytes, cudaMemcpyDefault));
    // libero la memoria de la placa
    cutilSafeCall(cudaFree(dev_x));
    cutilSafeCall(cudaFree(dev_s));
}

__global__ static void set_bnd_kernel(uint n, boundary b, float * x)
{
    uint i = blockIdx.x * blockDim.x + threadIdx.x + 1;

    if (i <= n) {
        x[IX(0, i)]     = b == VERTICAL ? -x[IX(1, i)] : x[IX(1, i)];
        x[IX(n + 1, i)] = b == VERTICAL ? -x[IX(n, i)] : x[IX(n, i)];
        x[IX(i, 0)]     = b == HORIZONTAL ? -x[IX(i, 1)] : x[IX(i, 1)];
        x[IX(i, n + 1)] = b == HORIZONTAL ? -x[IX(i, n)] : x[IX(i, n)];
    }

    if (i == 1) { 
        x[IX(0, 0)] = 0.5f * (x[IX(1, 0)]     + x[IX(0, 1)]);
    }
    
    if (i == n) {
        x[IX(0, n + 1)]     = 0.5f * (x[IX(1, n + 1)] + x[IX(0, n)]);
        x[IX(n + 1, 0)]     = 0.5f * (x[IX(n, 0)]     + x[IX(n + 1, 1)]);
        x[IX(n + 1, n + 1)] = 0.5f * (x[IX(n, n + 1)] + x[IX(n + 1, n)]);
    }
}

static void set_bnd_parallel(unsigned int n, boundary b, float * x)
{
    // dimensiones para set_bnd_kernel
    dim3 block(BLOCK_WIDTH);
    dim3 grid(DIV_CEIL(n, block.x));

    set_bnd_kernel<<<grid, block>>>(n, b, x);
    CUT_CHECK_ERROR("Error en la actualizacion de los bordes: ");
    cutilSafeCall(cudaDeviceSynchronize()); // espero a que los kernels terminen
}

static void set_bnd(unsigned int n, boundary b, float * x)
{
    uint total_bytes = (n+2) * (n+2) * sizeof(float);
    float *dev_x = NULL;

    // reservo memoria en la placa
    cutilSafeCall(cudaMalloc(&dev_x, total_bytes));
    // copio x a la placa
    cutilSafeCall(cudaMemcpy(dev_x, x, total_bytes, cudaMemcpyDefault));
    
    set_bnd_parallel(n, b, dev_x);

    // traigo x devuelta al host 
    cutilSafeCall(cudaMemcpy(x, dev_x, total_bytes, cudaMemcpyDefault));
    
    // libero la memoria de la placa
    cutilSafeCall(cudaFree(dev_x));    
}

template<Color color>
__global__  static void lin_solve_update_cell(unsigned int n, float * x, const float * x0, float a, float c) {
    
    // centro sumando 1 a cada coordenada
    uint i = threadIdx.y + blockIdx.y * blockDim.y + 1;     
    uint j = threadIdx.x + blockIdx.x * blockDim.x + 1;
    
    if (i <= n && j <= n && (i+j) % 2 == color){
        x[IX(j, i)] = (x0[IX(j, i)] + a * (x[IX(j - 1, i)] + x[IX(j + 1, i)] + x[IX(j, i - 1)] + x[IX(j, i + 1)])) / c;
    }
}

static void lin_solve_parallel(unsigned int n, boundary b, float * x, const float * x0, float a, float c)
{    
     // dimensiones para el kernel lin_solve_update_cell
    dim3 block(BLOCK_WIDTH, BLOCK_HEIGHT);
    dim3 grid(DIV_CEIL(n, block.x), DIV_CEIL(n, block.y));
    
    for (unsigned int k = 0; k < 20; k++) 
    {
        // red
        lin_solve_update_cell<RED> <<<grid, block>>>(n, x, x0, a, c);
        // black
        lin_solve_update_cell<BLACK> <<<grid, block>>>(n, x, x0, a, c);

        CUT_CHECK_ERROR("Error en la actualizacion de la celdas: ");
        cutilSafeCall(cudaDeviceSynchronize()); // espero a que los kernels terminen

        // bordes
        set_bnd_parallel(n, b, x);
    }
}

static void lin_solve(unsigned int n, boundary b, float * x, const float * x0, float a, float c)
{    
    float *dev_x = NULL, *dev_x0 = NULL;
    unsigned int total_bytes = (n+2) * (n+2) * sizeof(float);
    
    // reservo memoria en la placa
    cutilSafeCall(cudaMalloc(&dev_x, total_bytes));
    cutilSafeCall(cudaMalloc(&dev_x0, total_bytes));
    
    // copio x y x0 a la placa
    cutilSafeCall(cudaMemcpy(dev_x, x, total_bytes, cudaMemcpyDefault));
    cutilSafeCall(cudaMemcpy(dev_x0, x0, total_bytes, cudaMemcpyDefault));

    lin_solve_parallel(n, b, dev_x, dev_x0, a, c);        
    
    // traigo x devuelta al host 
    cutilSafeCall(cudaMemcpy(x, dev_x, total_bytes, cudaMemcpyDefault));
    
    // libero la memoria de la placa
    cutilSafeCall(cudaFree(dev_x));
    cutilSafeCall(cudaFree(dev_x0));
}

static void diffuse(unsigned int n, boundary b, float * x, const float * x0, float diff, float dt)
{
    float a = dt * diff * n * n;
    lin_solve(n, b, x, x0, a, 1 + 4 * a);
}

__global__ static void advect_kernel(unsigned int n, boundary b, float * d, const float * d0, const float * u, const float * v, float dt)
{
    // centro sumando 1 a cada coordenada
    uint j = threadIdx.y + blockIdx.y * blockDim.y + 1;     
    uint i = threadIdx.x + blockIdx.x * blockDim.x + 1;
    
    if (i <= n && j <= n) {
        int i0, i1, j0, j1;
        float x, y, s0, t0, s1, t1;
        float dt0 = dt * n;

        x = i - dt0 * u[IX(i, j)];
        y = j - dt0 * v[IX(i, j)];
        if (x < 0.5f) {
            x = 0.5f;
        } else if (x > n + 0.5f) {
            x = n + 0.5f;
        }
        i0 = (int) x;
        i1 = i0 + 1;
        if (y < 0.5f) {
            y = 0.5f;
        } else if (y > n + 0.5f) {
            y = n + 0.5f;
        }
        j0 = (int) y;
        j1 = j0 + 1;
        s1 = x - i0;
        s0 = 1 - s1;
        t1 = y - j0;
        t0 = 1 - t1;
        d[IX(i, j)] = s0 * (t0 * d0[IX(i0, j0)] + t1 * d0[IX(i0, j1)]) +
                      s1 * (t0 * d0[IX(i1, j0)] + t1 * d0[IX(i1, j1)]);
    }
}

static void advect_parallel(unsigned int n, boundary b, float * d, const float * d0, const float * u, const float * v, float dt)
{    
    // dimensiones para advect_kernel
    dim3 block(BLOCK_WIDTH, BLOCK_HEIGHT);
    dim3 grid(DIV_CEIL(n, block.x), DIV_CEIL(n, block.y));
       
    advect_kernel<<<grid, block>>>(n, b, d, d0, u, v, dt);
    
    CUT_CHECK_ERROR("Error en advect :");
    cutilSafeCall(cudaDeviceSynchronize()); // espero a que termine el kernel

    set_bnd_parallel(n, b, d);
}


static void advect(unsigned int n, boundary b, float * d, const float * d0, const float * u, const float * v, float dt)
{
    float *dev_u = NULL, *dev_v = NULL, *dev_d = NULL, *dev_d0 = NULL;
    unsigned int total_bytes = (n+2) * (n+2) * sizeof(float);
    
    // reservo memoria en la placa
    cutilSafeCall(cudaMalloc(&dev_u, total_bytes));
    cutilSafeCall(cudaMalloc(&dev_v, total_bytes));
    cutilSafeCall(cudaMalloc(&dev_d, total_bytes));
    cutilSafeCall(cudaMalloc(&dev_d0, total_bytes));
            
    // copio u, v, d, d0 a la placa
    cutilSafeCall(cudaMemcpy(dev_u, u, total_bytes, cudaMemcpyDefault));
    cutilSafeCall(cudaMemcpy(dev_v, v, total_bytes, cudaMemcpyDefault));
    cutilSafeCall(cudaMemcpy(dev_d, d, total_bytes, cudaMemcpyDefault));
    cutilSafeCall(cudaMemcpy(dev_d0, d0, total_bytes, cudaMemcpyDefault));
    
    advect_parallel(n, b, dev_d, dev_d0, dev_u, dev_v, dt);
    
    // traigo d devuelta al host 
    cutilSafeCall(cudaMemcpy(d, dev_d, total_bytes, cudaMemcpyDefault));
                
    // libero la memoria de la placa
    cutilSafeCall(cudaFree(dev_u));
    cutilSafeCall(cudaFree(dev_v));
    cutilSafeCall(cudaFree(dev_d));
    cutilSafeCall(cudaFree(dev_d0));
}

__global__  static void project_update_div_and_p_cell(unsigned int n, float *u, float *v, float *p, float *div) {
    
    // centro sumando 1 a cada coordenada
    uint i = threadIdx.y + blockIdx.y * blockDim.y + 1;     
    uint j = threadIdx.x + blockIdx.x * blockDim.x + 1;
    
    if (i <= n && j <= n){
        div[IX(j, i)] = -0.5f * (u[IX(j + 1, i)] - u[IX(j - 1, i)] +
                                 v[IX(j, i + 1)] - v[IX(j, i - 1)]) / n;
        p[IX(j, i)] = 0;
    }
}

__global__  static void project_update_u_and_v_cell(unsigned int n, float *u, float *v, float *p, float *div) {
    
    // centro sumando 1 a cada coordenada
    uint i = threadIdx.y + blockIdx.y * blockDim.y + 1;     
    uint j = threadIdx.x + blockIdx.x * blockDim.x + 1;
    
    if (i <= n && j <= n){
        u[IX(j, i)] -= 0.5f * n * (p[IX(j + 1, i)] - p[IX(j - 1, i)]);
        v[IX(j, i)] -= 0.5f * n * (p[IX(j, i + 1)] - p[IX(j, i - 1)]);
    }
}


static void project_parallel(unsigned int n, float *u, float *v, float *p, float *div)
{
    // dimensiones para los kernels project_update_*_cells
    dim3 block(BLOCK_WIDTH, BLOCK_HEIGHT);
    dim3 grid(DIV_CEIL(n, block.x), DIV_CEIL(n, block.y));
    
    project_update_div_and_p_cell<<<grid, block>>>(n, u, v, p, div);
    
    CUT_CHECK_ERROR("Error en la actualizacion de la celdas de div y p : ");
    cutilSafeCall(cudaDeviceSynchronize()); // espero a que el kernel termine
    
    set_bnd_parallel(n, NONE, div);
    set_bnd_parallel(n, NONE, p);

    lin_solve_parallel(n, NONE, p, div, 1, 4);

    project_update_u_and_v_cell<<<grid, block>>>(n, u, v, p, div);
    CUT_CHECK_ERROR("Error en la actualizacion de la celdas de u y v : ");
    cutilSafeCall(cudaDeviceSynchronize()); // espero a que el kernel termine

    set_bnd_parallel(n, VERTICAL, u);
    set_bnd_parallel(n, HORIZONTAL, v);
}

static void project(unsigned int n, float *u, float *v, float *p, float *div)
{

    float *dev_u = NULL, *dev_v = NULL, *dev_p = NULL, *dev_div = NULL;
    unsigned int total_bytes = (n+2) * (n+2) * sizeof(float);
    
    // reservo memoria en la placa
    cutilSafeCall(cudaMalloc(&dev_u, total_bytes));
    cutilSafeCall(cudaMalloc(&dev_v, total_bytes));
    cutilSafeCall(cudaMalloc(&dev_p, total_bytes));
    cutilSafeCall(cudaMalloc(&dev_div, total_bytes));
            
    // copio u, v, p, div a la placa
    cutilSafeCall(cudaMemcpy(dev_u, u, total_bytes, cudaMemcpyDefault));
    cutilSafeCall(cudaMemcpy(dev_v, v, total_bytes, cudaMemcpyDefault));
    cutilSafeCall(cudaMemcpy(dev_p, p, total_bytes, cudaMemcpyDefault));
    cutilSafeCall(cudaMemcpy(dev_div, div, total_bytes, cudaMemcpyDefault));
    
    project_parallel(n, dev_u, dev_v, dev_p, dev_div);
    
    // traigo u, v, p, div devuelta al host 
    cutilSafeCall(cudaMemcpy(u, dev_u, total_bytes, cudaMemcpyDefault));
    cutilSafeCall(cudaMemcpy(v, dev_v, total_bytes, cudaMemcpyDefault));
    cutilSafeCall(cudaMemcpy(p, dev_p, total_bytes, cudaMemcpyDefault));
    cutilSafeCall(cudaMemcpy(div, dev_div, total_bytes, cudaMemcpyDefault));
                
    // libero la memoria de la placa
    cutilSafeCall(cudaFree(dev_u));
    cutilSafeCall(cudaFree(dev_v));
    cutilSafeCall(cudaFree(dev_p));
    cutilSafeCall(cudaFree(dev_div));
}

extern "C" void dens_step(unsigned int n, float *x, float *x0, float *u, float *v, float diff, float dt)
{
    add_source(n, x, x0, dt);
    SWAP(x0, x);
    diffuse(n, NONE, x, x0, diff, dt);
    SWAP(x0, x);
    advect(n, NONE, x, x0, u, v, dt);
}

extern "C" void vel_step(unsigned int n, float *u, float *v, float *u0, float *v0, float visc, float dt)
{
    add_source(n, u, u0, dt);
    add_source(n, v, v0, dt);
    SWAP(u0, u);
    diffuse(n, VERTICAL, u, u0, visc, dt);
    SWAP(v0, v);
    diffuse(n, HORIZONTAL, v, v0, visc, dt);
    project(n, u, v, u0, v0);
    SWAP(u0, u);
    SWAP(v0, v);
    advect(n, VERTICAL, u, u0, u0, v0, dt);
    advect(n, HORIZONTAL, v, v0, u0, v0, dt);
    project(n, u, v, u0, v0);
}
