/*
  ======================================================================
   demo.c --- protoype to show off the simple solver
  ----------------------------------------------------------------------
   Author : Jos Stam (jstam@aw.sgi.com)
   Creation Date : Jan 9 2003

   Description:

	This code is a simple prototype that demonstrates how to use the
	code provided in my GDC2003 paper entitles "Real-Time Fluid Dynamics
	for Games". This code uses OpenGL and GLUT for graphics and interface

  =======================================================================
*/

#include <stdlib.h>
#include <stdio.h>

#include <cuda.h>           // API de cuda
#include <cutil_inline.h>   // Funciones para chequeo de errores

#include "timing.h"

/* macros */

#define IX(i,j) ((i)+(N+2)*(j))

/*
#define BLOCK_WIDTH 32
#define BLOCK_HEIGHT 16
*/

/* external definitions (from solver.c) */

extern "C" void dens_step ( int N, float * x, float * x0, float * u, float * v, float diff, float dt );
extern "C" void vel_step ( int N, float * u, float * v, float * u0, float * v0, float visc, float dt );

/* global variables */

static int N;
static float dt, diff, visc;
static float force, source;

static float * u, * v, * u_prev, * v_prev;
static float * dens, * dens_prev;

// arreglos auxiliares para almacenar memoria en host para utilizar en react
static float * host_u_prev, * host_v_prev, * host_dens_prev;

/*
  ----------------------------------------------------------------------
   free/clear/allocate simulation data
  ----------------------------------------------------------------------
*/


static void free_data ( void )
{
    // libero memoria en el device
	if ( u ) cutilSafeCall(cudaFree( u ));
	if ( v ) cutilSafeCall(cudaFree( v ));
	if ( u_prev ) cutilSafeCall(cudaFree( u_prev ));
	if ( v_prev ) cutilSafeCall(cudaFree( v_prev ));
	if ( dens ) cutilSafeCall(cudaFree( dens ));
	if ( dens_prev ) cutilSafeCall(cudaFree( dens_prev ));
    
    // libero memoria en el host
	if ( host_u_prev ) free( host_u_prev );
	if ( host_v_prev ) free( host_v_prev );
	if ( host_dens_prev ) free( host_dens_prev );
}

static void clear_data ( void )
{
	int size_bytes = (N+2) * (N+2) * sizeof(float);

    // seteo la memoria en el device a 0
    cutilSafeCall(cudaMemset(u, 0, size_bytes));
    cutilSafeCall(cudaMemset(u_prev, 0, size_bytes));
    cutilSafeCall(cudaMemset(v, 0, size_bytes));
    cutilSafeCall(cudaMemset(v_prev, 0, size_bytes));
    cutilSafeCall(cudaMemset(dens, 0, size_bytes));
    cutilSafeCall(cudaMemset(dens_prev, 0, size_bytes));
}

static int allocate_data ( void )
{
	int size_bytes = (N+2) * (N+2) * sizeof(float);
	
    // reservo memoria en el device
    cutilSafeCall(cudaMalloc(&u, size_bytes));
    cutilSafeCall(cudaMalloc(&v, size_bytes));
    cutilSafeCall(cudaMalloc(&u_prev, size_bytes));
    cutilSafeCall(cudaMalloc(&v_prev, size_bytes));
    cutilSafeCall(cudaMalloc(&dens, size_bytes));
    cutilSafeCall(cudaMalloc(&dens_prev, size_bytes));
	
	//reservo memoria en el host
	host_u_prev		= (float *) malloc ( size_bytes );
	host_v_prev		= (float *) malloc ( size_bytes );
    host_dens_prev  = (float *) malloc ( size_bytes );
    
    if ( !host_u_prev || !host_v_prev || !host_dens_prev ) {
		fprintf ( stderr, "cannot allocate data\n" );
		return ( 0 );
	}

	return ( 1 );
}

// reduccion del maximo de un array utilizando un arbol y memoria compartida

/*
__global__ void max_tree(float * array, float * result) {

    __shared__ float tmp[BLOCK_WIDTH * BLOCK_HEIGHT];

    uint x = blockIdx.x * blockDim.x + threadIdx.x;
    uint y = blockIdx.y * blockDim.y + threadIdx.y;
    uint idx = x + y * N;
    uint tid = threadIdx.y * blockDim.x + threadIdx.x;

    if (x <= N && y <= N) {
        tmp[tid] = array[idx];
    } else {
        tmp[tid] = 0;
    }

    __syncthreads();
    
    for (uint distance = BLOCK_WIDTH * BLOCK_HEIGHT / 2; distance >= 1; distance = distance / 2) {
        if ( tid < distance ) {
            tmp[tid] = max(tmp[tid], tmp[tid + distance]);
        }
        if (distance > 16) { //se evita sinchronizacion en un warp (<= 32 elementos)
            __syncthreads();
        }
    }

    if (tid == 0) {
        atomicMax(result, tmp[0]);
    }
}

static void react ( float * d, float * u, float * v )
{
	int size_bytes = (N+2) * (N+2) * sizeof(float);
	
	float max_velocity2;
	float max_density;

    cutilSafeCall(cudaMalloc(&max_velocity2, sizeof(float)));
    cutilSafeCall(cudaMemset(&max_velocity2, 0, sizeof(float)));
    max_tree(u_pow2_plus_v_pow2, &max_velocity2);
    
    CUT_CHECK_ERROR("Error al reducir maxima velocidad :");
    cutilSafeCall(cudaDeviceSynchronize()); // espero a que termine el kernel    
    
    cutilSafeCall(cudaMalloc(&max_density, sizeof(float)));
    cutilSafeCall(cudaMemset(&max_density, 0, sizeof(float)));
    max_tree(d, &max_density);

    CUT_CHECK_ERROR("Error al reducir maxima densidad :");
    cutilSafeCall(cudaDeviceSynchronize()); // espero a que termine el kernel    

    cutilSafeCall(cudaMemset(u, 0, size_bytes));
    cutilSafeCall(cudaMemset(v, 0, size_bytes));
    cutilSafeCall(cudaMemset(d, 0, size_bytes));
        
	for ( i=0 ; i<size ; i++ ) {
		if (max_velocity2 < u[i]*u[i] + v[i]*v[i]) {
			max_velocity2 = u[i]*u[i] + v[i]*v[i];
		}
		if (max_density < d[i]) {
			max_density = d[i];
		}
	}

	for ( i=0 ; i<size ; i++ ) {
		u[i] = v[i] = d[i] = 0.0f;
	}

	if (max_velocity2<0.0000005f) {
		u[IX(N/2,N/2)] = force * 10.0f;
		v[IX(N/2,N/2)] = force * 10.0f;
	}
	if (max_density<1.0f) {
		d[IX(N/2,N/2)] = source * 10.0f;
	}

	return;
}
*/
static void react ( float * d, float * u, float * v )
{
	int size = (N+2) * (N+2);
	
	float max_velocity2 = 0.0f;
	float max_density   = 0.0f;

	for ( int i=0 ; i<size ; i++ ) {
		if (max_velocity2 < u[i]*u[i] + v[i]*v[i]) {
			max_velocity2 = u[i]*u[i] + v[i]*v[i];
		}

		if (max_density < d[i]) {
			max_density = d[i];
		}
	}

	for ( int i=0 ; i<size ; i++ ) {
		u[i] = v[i] = d[i] = 0.0f;
	}

	if (max_velocity2<0.0000005f) {
		u[IX(N/2,N/2)] = force * 10.0f;
		v[IX(N/2,N/2)] = force * 10.0f;
	}
	if (max_density<1.0f) {
		d[IX(N/2,N/2)] = source * 10.0f;
	}

	return;
}


static void one_step ( void )
{
	static int times = 1;
	static double start_t = 0.0;
	static double one_second = 0.0;
	static double react_ns_p_cell = 0.0;
	static double vel_ns_p_cell = 0.0;
	static double dens_ns_p_cell = 0.0;

	int size_bytes = (N+2) * (N+2) * sizeof(float);
	
	// copio memoria auxiliar del device al host
	cutilSafeCall(cudaMemcpy(host_u_prev, u_prev, size_bytes, cudaMemcpyDefault));
	cutilSafeCall(cudaMemcpy(host_v_prev, v_prev, size_bytes, cudaMemcpyDefault));
	cutilSafeCall(cudaMemcpy(host_dens_prev, dens_prev, size_bytes, cudaMemcpyDefault));
    
    // no cuento lo que tarda mover la memoria por que voy a implementar el react en gpu 
	start_t = wtime();
	react ( host_dens_prev, host_u_prev, host_v_prev );
	react_ns_p_cell += 1.0e9 * (wtime()-start_t)/(N*N);
	
	// copio memoria auxiliar del host al device
	cutilSafeCall(cudaMemcpy(u_prev, host_u_prev, size_bytes, cudaMemcpyDefault));
	cutilSafeCall(cudaMemcpy(v_prev, host_v_prev, size_bytes, cudaMemcpyDefault));
	cutilSafeCall(cudaMemcpy(dens_prev, host_dens_prev, size_bytes, cudaMemcpyDefault));

	start_t = wtime();
	vel_step ( N, u, v, u_prev, v_prev, visc, dt );
	vel_ns_p_cell += 1.0e9 * (wtime()-start_t)/(N*N);

	start_t = wtime();
	dens_step ( N, dens, dens_prev, u, v, diff, dt );
	dens_ns_p_cell += 1.0e9 * (wtime()-start_t)/(N*N);

	if (1.0<wtime()-one_second) { /* at least 1s between stats */
		printf("%lf, %lf, %lf, %lf: ns per cell total, react, vel_step, dens_step\n",
			(react_ns_p_cell+vel_ns_p_cell+dens_ns_p_cell)/times,
			react_ns_p_cell/times, vel_ns_p_cell/times, dens_ns_p_cell/times);
		one_second = wtime();
		react_ns_p_cell = 0.0;
		vel_ns_p_cell = 0.0;
		dens_ns_p_cell = 0.0;
		times = 1;
	} else {
		times++;
	}
}


/*
  ----------------------------------------------------------------------
   main --- main routine
  ----------------------------------------------------------------------
*/

int main ( int argc, char ** argv )
{
	int i = 0;

	if ( argc != 1 && argc != 2 && argc != 6 ) {
		fprintf ( stderr, "usage : %s N dt diff visc force source\n", argv[0] );
		fprintf ( stderr, "where:\n" );\
		fprintf ( stderr, "\t N      : grid resolution\n" );
		fprintf ( stderr, "\t dt     : time step\n" );
		fprintf ( stderr, "\t diff   : diffusion rate of the density\n" );
		fprintf ( stderr, "\t visc   : viscosity of the fluid\n" );
		fprintf ( stderr, "\t force  : scales the mouse movement that generate a force\n" );
		fprintf ( stderr, "\t source : amount of density that will be deposited\n" );
		exit ( 1 );
	}

	if ( argc == 1 ) {
		N = 128;
		dt = 0.1f;
		diff = 0.0f;
		visc = 0.0f;
		force = 5.0f;
		source = 100.0f;
		fprintf ( stderr, "Using defaults : N=%d dt=%g diff=%g visc=%g force = %g source=%g\n",
			N, dt, diff, visc, force, source );
	} else if (argc == 2) {
	    N = atoi( argv[1] );
	    dt = 0.1f;
		diff = 0.0f;
		visc = 0.0f;
		force = 5.0f;
		source = 100.0f;
		fprintf ( stderr, "Using defaults : N=%d dt=%g diff=%g visc=%g force = %g source=%g\n",
			N, dt, diff, visc, force, source );		
	} else {
		N = atoi(argv[1]);
		dt = atof(argv[2]);
		diff = atof(argv[3]);
		visc = atof(argv[4]);
		force = atof(argv[5]);
		source = atof(argv[6]);
	}

	if ( !allocate_data () ) exit ( 1 );
	clear_data ();
	for (i=0; i<2048; i++)
		one_step ();
	free_data ();

	exit ( 0 );
}
