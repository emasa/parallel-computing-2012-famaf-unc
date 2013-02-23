/*******************************************************************************
 Computación Paralela 2012 - FaMAF
 Universidad Nacional de Córdoba

OpenMP schedules
================
El siguiente código paraleliza una tarea que toma en promedio AVGTIME
microsegundos pero varía significativamente. Para simplificar el problema, cada
thread demora siempre el mismo tiempo.

a) Corra el código con distintas cantidades de threads (cambiando la variable de
entorno OMP_NUM_THREADS) y calcule eficiencia y speedup en cada caso.

b) Repita el experimento anterior cambiando el schedule con el que se paraleliza
el loop principal (revisar la documentación de OpenMP) y observe cómo cambia la
distribución de trabajo.

********************************************************************************/

#define _BSD_SOURCE

#include <omp.h>
#include <stdio.h>
#include <unistd.h>

#define WORK 200

// El trabajo toma cerca de 50ms
#define AVGTIME 50000

void work(int thread) {
	int threads = omp_get_num_threads();
	if (threads > 1) {
		// trabajo entre 0.5 * AVGTIME y 1.5 * AVGTIME
		// los threads altos son más lentos
		unsigned int work = (AVGTIME/2) + (AVGTIME / (omp_get_num_threads() - 1)) * thread;
		usleep(work);
	} else {
		usleep(AVGTIME);
	}
}


void print(int thread, int workitem) {	
	// cada thread imprime en una columna distinta
	for (int i = 0; i < thread; ++i) {
		printf("\t");
	}
	printf("%d\n", workitem);
}


int main(int argc, char **argv) {

	double t = omp_get_wtime();
	#pragma omp parallel
	{
		int tid = omp_get_thread_num();

		// jugar con schedules aquí
		#pragma omp for schedule(guided, 1)
		//#pragma omp for schedule(dynamic, 1)
		//#pragma omp for schedule(static, 1)
		for (int i = 0; i < WORK; ++i) {
			work(tid);

			// stdout es un recurso compartido
			#pragma omp critical
			{
				print(tid, i);
			}
		}
	}
	t = omp_get_wtime() - t;

	printf("%fs\n", t);
	return 0;
}
	