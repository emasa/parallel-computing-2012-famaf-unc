
#include <stdio.h>

#include "transpose_utils.h"

void init_matrix(int width, int height, float* in){	
	int c = 0;
	for (size_t i = 0; i < height; i++) {
		for (size_t j = 0; j < width; j++) {
			AT(in, i, j, width) = c++;
		}
	}
}

void check_traspose(int width, int height, float* out){
	int c = 0;
	for (size_t i = 0; i < height; i++) {
		for (size_t j = 0; j < width; j++) {
            if ((int) AT(out, i, j, width) != (i + j * height)) {
                printf ("(%i,%i) failed\n", i, j);
            }
		}
	}
}

void print_matrix(int width, int height, float* in){
	for (size_t i = 0; i < height; i++) {
		for (size_t j = 0; j < width; j++) {
			printf("%i ", (int) AT(in, i, j, width));
		}
		printf("\n");
	}
}
