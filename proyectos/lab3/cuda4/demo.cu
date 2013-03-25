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
#include <assert.h>
#include <GL/glut.h>

#include <thrust/device_vector.h> // API para la reduccion del maximo
#include <cuda.h>           // API de cuda
#include <cutil_inline.h>   // Funciones para chequeo de errores

#include "timing.h"

/* macros */

#define IX(i,j) ((i)+(N+2)*(j))
#define DIV_CEIL(n, m) ((n) + (m) -1) / (m)

#define OPTIMAL_BLOCK_WIDTH 32
#define OPTIMAL_BLOCK_HEIGHT 5

/* external definitions (from solver.c) */

extern "C" void dens_step ( int N, float * x, float * x0, float * u, float * v, float diff, float dt );
extern "C" void vel_step ( int N, float * u, float * v, float * u0, float * v0, float visc, float dt );

/* global variables */

static int N;
static float dt, diff, visc;
static float force, source;
static int dvel;

static float * u, * v, * u_prev, * v_prev;
static float * dens, * dens_prev;

static int win_id;
static int win_x, win_y;
static int mouse_down[3];
static int omx, omy, mx, my;

/* global helper variables */

static float * vel2; // calculo de u*u + v*v
static float * host_u, * host_v, * host_dens; // renderizado

int BLOCK_WIDTH, BLOCK_HEIGHT; // deben ser importado desde el solver

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
    	
	if ( vel2 ) cutilSafeCall(cudaFree( vel2 ));    
    
    // libero memoria en el host
	if ( host_u ) free( host_u );
	if ( host_v ) free( host_v );
	if ( host_dens ) free( host_dens );
}

static void clear_data ( void )
{
	int size_bytes = (N+2) * (N+2) * sizeof(float);

    // seteo la memoria en el device a 0
    cutilSafeCall(cudaMemset(u, 0, size_bytes));
    cutilSafeCall(cudaMemset(dens, 0, size_bytes));
    cutilSafeCall(cudaMemset(v, 0, size_bytes));
    cutilSafeCall(cudaMemset(u_prev, 0, size_bytes));
    cutilSafeCall(cudaMemset(v_prev, 0, size_bytes));
    cutilSafeCall(cudaMemset(dens_prev, 0, size_bytes));

    cutilSafeCall(cudaMemset(vel2, 0, size_bytes));
}

static int allocate_data ( void )
{
	int size_bytes = (N+2) * (N+2) * sizeof(float);
	
    // reservo memoria en el device
    cutilSafeCall(cudaMalloc(&u, size_bytes));
    cutilSafeCall(cudaMalloc(&v, size_bytes));
    cutilSafeCall(cudaMalloc(&dens, size_bytes));
    cutilSafeCall(cudaMalloc(&u_prev, size_bytes));
    cutilSafeCall(cudaMalloc(&v_prev, size_bytes));
    cutilSafeCall(cudaMalloc(&dens_prev, size_bytes));

    cutilSafeCall(cudaMalloc(&vel2, size_bytes));
	
	//reservo memoria en el host
	host_u		= (float *) malloc ( size_bytes );
	host_v		= (float *) malloc ( size_bytes );
    host_dens   = (float *) malloc ( size_bytes );
    
    if (!host_u || !host_v || !host_dens) {
		fprintf ( stderr, "cannot allocate data\n" );
		return ( 0 );
	}

	return ( 1 );
}


/*
  ----------------------------------------------------------------------
   OpenGL specific drawing routines
  ----------------------------------------------------------------------
*/

static void pre_display ( void )
{
	glViewport ( 0, 0, win_x, win_y );
	glMatrixMode ( GL_PROJECTION );
	glLoadIdentity ();
	gluOrtho2D ( 0.0, 1.0, 0.0, 1.0 );
	glClearColor ( 0.0f, 0.0f, 0.0f, 1.0f );
	glClear ( GL_COLOR_BUFFER_BIT );
}

static void post_display ( void )
{
	glutSwapBuffers ();
}

static void draw_velocity ( void )
{
	// copio memoria auxiliar del device al host para el renderizado
    int size_bytes = (N+2) * (N+2) * sizeof(float);
	cutilSafeCall(cudaMemcpy(host_u, u, size_bytes, cudaMemcpyDefault));
	cutilSafeCall(cudaMemcpy(host_v, v, size_bytes, cudaMemcpyDefault));

	int i, j;
	float x, y, h;

	h = 1.0f/N;

	glColor3f ( 1.0f, 1.0f, 1.0f );
	glLineWidth ( 1.0f );

	glBegin ( GL_LINES );

		for ( i=1 ; i<=N ; i++ ) {
			x = (i-0.5f)*h;
			for ( j=1 ; j<=N ; j++ ) {
				y = (j-0.5f)*h;

				glVertex2f ( x, y );
				glVertex2f ( x + host_u[IX(i,j)], y + host_v[IX(i,j)] );
			}
		}

	glEnd ();
}

static void draw_density ( void )
{
	// copio memoria auxiliar del device al host para el renderizado
    int size_bytes = (N+2) * (N+2) * sizeof(float);
	cutilSafeCall(cudaMemcpy(host_dens, dens, size_bytes, cudaMemcpyDefault));

	int i, j;
	float x, y, h, d00, d01, d10, d11;

	h = 1.0f/N;

	glBegin ( GL_QUADS );

		for ( i=0 ; i<=N ; i++ ) {
			x = (i-0.5f)*h;
			for ( j=0 ; j<=N ; j++ ) {
				y = (j-0.5f)*h;

				d00 = host_dens[IX(i,j)];
				d01 = host_dens[IX(i,j+1)];
				d10 = host_dens[IX(i+1,j)];
				d11 = host_dens[IX(i+1,j+1)];

				glColor3f ( d00, d00, d00 ); glVertex2f ( x, y );
				glColor3f ( d10, d10, d10 ); glVertex2f ( x+h, y );
				glColor3f ( d11, d11, d11 ); glVertex2f ( x+h, y+h );
				glColor3f ( d01, d01, d01 ); glVertex2f ( x, y+h );
			}
		}

	glEnd ();
}

/*
  ----------------------------------------------------------------------
   relates mouse movements to forces sources
  ----------------------------------------------------------------------
*/

__global__ static void velocity2(unsigned int n, const float* u, const float* v, float* vel2){

    uint x = blockIdx.x * blockDim.x + threadIdx.x;
    uint y = blockIdx.y * blockDim.y + threadIdx.y;
    uint idx = y * n + x;
    if (x < n && y < n){
        vel2[idx] = u[idx] * u[idx] + v[idx] * v[idx];
    }
}

__global__ static void init_u_v_d(unsigned int n, float * d, float * u, float * v, 
                                  float max_velocity2, float max_density, 
                                  float force, float source, 
                                  int mouse_down0, int mouse_down2, 
                                  int mx, int my, int omx, int omy,
                                  int win_x, int win_y) {
    // un solo hilo inicializa u, v, d
    uint x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x == 0){
	    uint idx = (n / 2) * (n + 2) +  n / 2;
	    if (max_velocity2<0.0000005f) {
		    u[idx] = force * 10.0f;

		    v[idx] = force * 10.0f;
	    }
	    if (max_density<1.0f) {
		    d[idx] = source * 10.0f;
	    }
    
        if ( mouse_down0 || mouse_down2 ) return;

	    int i = (int)((       mx /(float)win_x)*n+1);
	    int j = (int)(((win_y-my)/(float)win_y)*n+1);

	    if ( i<1 || i>n || j<1 || j>n ) return;

	    if ( mouse_down0 ) {
		    u[j * (n+2) + i] = force * (mx-omx);
		    v[j * (n+2) + i] = force * (omy-my);
	    }

	    if ( mouse_down2 ) {
		    d[j * (n+2) + i] = source;
	    }
    }
}

void update_omx_omy() {
    // modifico omx, omy en el host
    if ( mouse_down[0] || mouse_down[2] ) return;
    int i = (int)((       mx /(float)win_x)*N+1);
    int j = (int)(((win_y-my)/(float)win_y)*N+1);
	if ( i<1 || i>N || j<1 || j>N ) return;
    omx = mx;
    omy = my;
}

static void react ( float * d, float * u, float * v )
{
	int size_bytes = (N+2) * (N+2) * sizeof(float);	    
    dim3 block(BLOCK_WIDTH, BLOCK_HEIGHT);
    dim3 grid(DIV_CEIL(N+2, block.x), DIV_CEIL(N+2, block.y));

    // calculo velocidad    
    velocity2<<<grid, block>>>(N+2, u, v, vel2);
    CUT_CHECK_ERROR("Error al calcular u*u + v*v :");
    cutilSafeCall(cudaDeviceSynchronize()); // espero a que termine el kernel   

    // calculo maxima velocidad
    thrust::device_ptr<float> thrust_vel2(vel2);
    float max_velocity2 = *thrust::max_element(thrust_vel2, thrust_vel2 + (N+2)*(N+2));
        
    // calculo maxima densidad
    thrust::device_ptr<float> thrust_d(d);
    float max_density = *thrust::max_element(thrust_d, thrust_d + (N+2)*(N+2));

    // seteo u, v, d a 0
    cutilSafeCall(cudaMemset(u, 0, size_bytes));
    cutilSafeCall(cudaMemset(v, 0, size_bytes));
    cutilSafeCall(cudaMemset(d, 0, size_bytes));
    
    // inicializo u, v, d
    init_u_v_d<<<dim3(1), dim3(BLOCK_WIDTH)>>>(N, d, u, v, max_velocity2, max_density, 
                                               force, source, mouse_down[0], mouse_down[2], 
                                               mx, my, omx, omy, win_x, win_y);
    CUT_CHECK_ERROR("Error al inicializar u, v, d :");
    cutilSafeCall(cudaDeviceSynchronize()); // espero a que termine el kernel
    update_omx_omy();
}

/*
  ----------------------------------------------------------------------
   GLUT callback routines
  ----------------------------------------------------------------------
*/

static void key_func ( unsigned char key, int x, int y )
{
	switch ( key )
	{
		case 'c':
		case 'C':
			clear_data ();
			break;

		case 'q':
		case 'Q':
			free_data ();
			exit ( 0 );
			break;

		case 'v':
		case 'V':
			dvel = !dvel;
			break;
	}
}

static void mouse_func ( int button, int state, int x, int y )
{
	omx = mx = x;
	omx = my = y;

	mouse_down[button] = state == GLUT_DOWN;
}

static void motion_func ( int x, int y )
{
	mx = x;
	my = y;
}

static void reshape_func ( int width, int height )
{
	glutSetWindow ( win_id );
	glutReshapeWindow ( width, height );

	win_x = width;
	win_y = height;
}

static void idle_func ( void )
{
	static int times = 1;
	static double start_t = 0.0;
	static double one_second = 0.0;
	static double react_ns_p_cell = 0.0;
	static double vel_ns_p_cell = 0.0;
	static double dens_ns_p_cell = 0.0;

	start_t = wtime();
	react ( dens_prev, u_prev, v_prev );
	react_ns_p_cell += 1.0e9 * (wtime()-start_t)/(N*N);

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

	glutSetWindow ( win_id );
	glutPostRedisplay ();
}

static void display_func ( void )
{
	pre_display ();

		if ( dvel ) draw_velocity ();
		else		draw_density ();

	post_display ();
}


/*
  ----------------------------------------------------------------------
   open_glut_window --- open a glut compatible window and set callbacks
  ----------------------------------------------------------------------
*/

static void open_glut_window ( void )
{
	glutInitDisplayMode ( GLUT_RGBA | GLUT_DOUBLE );

	glutInitWindowPosition ( 0, 0 );
	glutInitWindowSize ( win_x, win_y );
	win_id = glutCreateWindow ( "Alias | wavefront" );

	glClearColor ( 0.0f, 0.0f, 0.0f, 1.0f );
	glClear ( GL_COLOR_BUFFER_BIT );
	glutSwapBuffers ();
	glClear ( GL_COLOR_BUFFER_BIT );
	glutSwapBuffers ();

	pre_display ();

	glutKeyboardFunc ( key_func );
	glutMouseFunc ( mouse_func );
	glutMotionFunc ( motion_func );
	glutReshapeFunc ( reshape_func );
	glutIdleFunc ( idle_func );
	glutDisplayFunc ( display_func );
}


/*
  ----------------------------------------------------------------------
   main --- main routine
  ----------------------------------------------------------------------
*/

int main ( int argc, char ** argv )
{
	glutInit ( &argc, argv );

	if ( argc != 1 && argc != 4 && argc != 6 ) {
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
	    BLOCK_WIDTH = OPTIMAL_BLOCK_WIDTH;
	    BLOCK_HEIGHT = OPTIMAL_BLOCK_HEIGHT;
	} else if (argc == 4) {
	    N = atoi( argv[1] );
	    dt = 0.1f;
		diff = 0.0f;
		visc = 0.0f;
		force = 5.0f;
		source = 100.0f;
		fprintf ( stderr, "Using defaults : N=%d dt=%g diff=%g visc=%g force = %g source=%g\n",
			N, dt, diff, visc, force, source );		
	    BLOCK_WIDTH = atoi( argv [2] );
	    BLOCK_HEIGHT = atoi( argv [3] );
	} else {
		N = atoi(argv[1]);
		dt = atof(argv[2]);
		diff = atof(argv[3]);
		visc = atof(argv[4]);
		force = atof(argv[5]);
		source = atof(argv[6]);
	    BLOCK_WIDTH = OPTIMAL_BLOCK_WIDTH;
	    BLOCK_HEIGHT = OPTIMAL_BLOCK_HEIGHT;
	}

	assert (N > 0 && BLOCK_WIDTH > 0 && BLOCK_HEIGHT > 0);

	dvel = 0;

	if ( !allocate_data () ) exit ( 1 );
	clear_data ();

	win_x = 512;
	win_y = 512;
	open_glut_window ();

	glutMainLoop ();

	exit ( 0 );
}
