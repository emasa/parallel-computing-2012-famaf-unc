
#ifndef TRANSPOSE_UTILS_H
#define TRANSPOSE_UTILS_H

#define AT(a, i, j, n) (a)[(i) * (n) + (j)]

void init_matrix(int width, int height, float* in);

void check_traspose(int width, int height, float* out);

void print_matrix(int width, int height, float* in);

#endif
