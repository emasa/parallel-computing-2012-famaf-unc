
#ifndef TRANSPOSE_INTRINSIC_H
#define TRANSPOSE_INTRINSIC_H

void transpose_4x4_aligned(float * in, 
                   float * out, 
                   size_t height, 
                   size_t width, 
                   size_t y0, 
                   size_t x0);
                   
void transpose_2d_blocking_intrinsic_aligned(size_t width, 
						             size_t height,
						             float * in, 
						             float * out);
						             
void transpose_4x4_unaligned(float * in, 
                   float * out, 
                   size_t height, 
                   size_t width, 
                   size_t y0, 
                   size_t x0);
                   
void transpose_2d_blocking_intrinsic_unaligned(size_t width, 
						             size_t height,
						             float * in, 
						             float * out);						             
								       
#endif
