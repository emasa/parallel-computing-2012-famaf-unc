#include <stddef.h>

#include <cuda.h>           // API de cuda
#include <cutil_inline.h>   // Funciones para chequeo de errores

#define IX(i,j) ((i)+(n+2)*(j))
#define SWAP(x0,x) {float * tmp=x0;x0=x;x=tmp;}
#define DIV_CEIL(n, m) ((n) + (m) -1) / (m)

#define BLOCK_WIDTH 32
#define BLOCK_HEIGHT 5

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

static void add_source(uint n, float * x, const float * s, float dt)
{    
    uint n_total = n + 2;
    // dimensiones para add_source_kernel
    dim3 block(BLOCK_WIDTH, BLOCK_HEIGHT);
    dim3 grid(DIV_CEIL(n_total, block.x), DIV_CEIL(n_total, block.y));
       
    add_source_kernel<<<grid, block>>>(n_total, x, s, dt);
    
    CUT_CHECK_ERROR("Error al incrementar source :");
    cutilSafeCall(cudaDeviceSynchronize()); // espero a que termine el kernel
}

__global__ static void set_bnd_kernel(uint n, boundary b, float * x)
{
    // lanzo 4 n hilos 
    uint idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint i = (idx % n) + 1, sel = idx / n;
    
    if (sel == 0) x[IX(0, i)]          = b == VERTICAL   ? -x[IX(1, i)] : x[IX(1, i)];
    else if (sel == 3) x[IX(i, n + 1)] = b == HORIZONTAL ? -x[IX(i, n)] : x[IX(i, n)];
    else if (sel == 1) x[IX(n + 1, i)] = b == VERTICAL   ? -x[IX(n, i)] : x[IX(n, i)];
    else if (sel == 2) x[IX(i, 0)]     = b == HORIZONTAL ? -x[IX(i, 1)] : x[IX(i, 1)];

    if (idx == 1)      x[IX(0, 0)]      = 0.5f * (x[IX(1, 0)]     + x[IX(0, 1)]);
    else if (idx == n) x[IX(0, n + 1)]  = 0.5f * (x[IX(1, n + 1)] + x[IX(0, n)]);
    else if (idx == 3 * n + 1) x[IX(n + 1, 0)]     = 0.5f * (x[IX(n, 0)]     + x[IX(n + 1, 1)]);
    else if (idx == 4 * n)     x[IX(n + 1, n + 1)] = 0.5f * (x[IX(n, n + 1)] + x[IX(n + 1, n)]);
}

static void set_bnd(unsigned int n, boundary b, float * x)
{
    // dimensiones para set_bnd_kernel
    dim3 block(BLOCK_WIDTH);
    dim3 grid(DIV_CEIL(n*4, block.x));

    set_bnd_kernel<<<grid, block>>>(n, b, x);
    CUT_CHECK_ERROR("Error en la actualizacion de los bordes: ");
    cutilSafeCall(cudaDeviceSynchronize()); // espero a que los kernels terminen
}

template<Color color>
__global__  static void lin_solve_update_cell(unsigned int n, float * x, const float * x0, float a, float c) {

    // centro sumando 1 a cada coordenada
    uint i = threadIdx.y + blockIdx.y * blockDim.y + 1;     
    uint j = threadIdx.x + blockIdx.x * blockDim.x + 1;    
    
    if (i <= n && j <= n) {        
        const uint tmp_block_width  = BLOCK_WIDTH  + 2;
        const uint tmp_block_height = BLOCK_HEIGHT + 2;
        
        __shared__ float tmp_x[tmp_block_height][tmp_block_width];

        uint offset_x = blockIdx.x * blockDim.x;
        uint offset_y = blockIdx.y * blockDim.y;

        uint idx = threadIdx.y * blockDim.x + threadIdx.x; //indice lineal dentro del bloque                
        if (idx < tmp_block_height) {
            tmp_x[idx][0] = x[IX(offset_x, offset_y + idx)];
            tmp_x[idx][tmp_block_width - 1] = x[IX(offset_x + tmp_block_width - 1, offset_y + idx)];
        } else if (tmp_block_height <= idx && idx < tmp_block_height + tmp_block_width) { 
            idx -= tmp_block_height;
            tmp_x[0][idx] = x[IX(offset_x + idx, offset_y)];
            tmp_x[tmp_block_height-1][idx] = x[IX(offset_x + idx, offset_y + tmp_block_height - 1)];
        } else if (idx == tmp_block_height + tmp_block_width) { 
            tmp_x[0][0] = x[IX(offset_x, offset_y)];
            tmp_x[0][tmp_block_width-1]  = x[IX(offset_x + tmp_block_width - 1, offset_y)];
            tmp_x[tmp_block_height-1][0] = x[IX(offset_x, offset_y + tmp_block_height - 1)];
            tmp_x[tmp_block_height-1][tmp_block_width-1] = x[IX(offset_x + tmp_block_width - 1, offset_y + tmp_block_height - 1)];
        }

        uint tmp_i = threadIdx.y + 1;
        uint tmp_j = threadIdx.x + 1;        
        
        tmp_x[tmp_i][tmp_j] = x[IX(j, i)];
        
        __syncthreads();

        if ((i + j) % 2 == color) {
            x[IX(j, i)] = (x0[IX(j, i)] + a * (tmp_x[tmp_i][tmp_j - 1] + tmp_x[tmp_i][tmp_j + 1] + tmp_x[tmp_i - 1][tmp_j] + tmp_x[tmp_i + 1][tmp_j])) / c;
        }
    }
}

static void lin_solve(unsigned int n, boundary b, float * x, const float * x0, float a, float c)
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

        CUT_CHECK_ERROR("Error en la actualizacion de la celdas en lin_solve: ");
        cutilSafeCall(cudaDeviceSynchronize()); // espero a que los kernels terminen

        // bordes
        set_bnd(n, b, x);
    }
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

static void advect(unsigned int n, boundary b, float * d, const float * d0, const float * u, const float * v, float dt)
{    
    // dimensiones para advect_kernel
    dim3 block(BLOCK_WIDTH, BLOCK_HEIGHT);
    dim3 grid(DIV_CEIL(n, block.x), DIV_CEIL(n, block.y));
       
    advect_kernel<<<grid, block>>>(n, b, d, d0, u, v, dt);
    
    CUT_CHECK_ERROR("Error en advect :");
    cutilSafeCall(cudaDeviceSynchronize()); // espero a que termine el kernel

    set_bnd(n, b, d);
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

static void project(unsigned int n, float *u, float *v, float *p, float *div)
{
    // dimensiones para los kernels project_update_*_cells
    dim3 block(BLOCK_WIDTH, BLOCK_HEIGHT);
    dim3 grid(DIV_CEIL(n, block.x), DIV_CEIL(n, block.y));
    
    project_update_div_and_p_cell<<<grid, block>>>(n, u, v, p, div);
    
    CUT_CHECK_ERROR("Error en la actualizacion de la celdas de div y p : ");
    cutilSafeCall(cudaDeviceSynchronize()); // espero a que el kernel termine
    
    set_bnd(n, NONE, div);
    set_bnd(n, NONE, p);

    lin_solve(n, NONE, p, div, 1, 4);

    project_update_u_and_v_cell<<<grid, block>>>(n, u, v, p, div);
    CUT_CHECK_ERROR("Error en la actualizacion de la celdas de u y v : ");
    cutilSafeCall(cudaDeviceSynchronize()); // espero a que el kernel termine

    set_bnd(n, VERTICAL, u);
    set_bnd(n, HORIZONTAL, v);
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
