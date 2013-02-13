
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include <vector>
#include <algorithm>

#include "transpose_utils.h"
#include "timing.h"

using namespace std; 

#define MAX_BLOCK (1 << 8)

#define HEIGHT (1 << 13)
#define WIDTH HEIGHT

void transpose_2d_rec(size_t height, 
					  size_t width, 
					  size_t total_height,
					  size_t total_width,
					  size_t init_x,
					  size_t init_y, 
					  float* in,
                      float* out,
                      size_t min_block) 
{
	if (height > min_block || width > min_block){
		if (height > width){
			transpose_2d_rec( 
							 height / 2, width,
							 total_height, total_width,
							 init_x, init_y, 
							 in, out, min_block);
			transpose_2d_rec( 
							 (height + 1)/2, width,
							 total_height, total_width,
							 init_x, init_y + (height / 2),
							 in, out, min_block);
 		}else{
			transpose_2d_rec( 
							 height, width / 2,
							 total_height, total_width,
							 init_x, init_y,
							 in, out, min_block);
			transpose_2d_rec( 
							 height, (width + 1) / 2,
							 total_height, total_width,
							 init_x + (width / 2), init_y,
							 in, out, min_block);
 		}
	}else{
		for (size_t i = init_y; i < init_y + height && i < total_height; i++) {
			for (size_t j = init_x; j < init_x + width && j < total_width; j++) {
				AT(out, j, i, total_height) = AT(in, i, j, total_width);
			}
		}
	}
}

int main(){

	float * in = (float *) malloc(sizeof(float) * HEIGHT * WIDTH);
	assert(in != NULL);
	float * out = (float *) malloc(sizeof(float) * HEIGHT * WIDTH);
    assert(out != NULL);

	init_matrix(WIDTH, HEIGHT, in);
    
    vector< pair<double, size_t> > blocks;
    
    for(size_t block_size = 1; block_size <= MAX_BLOCK; block_size *= 2)
    {
        double t_start = wtime();
	    transpose_2d_rec(HEIGHT, WIDTH, HEIGHT, WIDTH, 0, 0, in, out, block_size);
        double time = wtime() - t_start;
        check_traspose(HEIGHT, WIDTH, out);
        
        blocks.push_back(make_pair(time, block_size));    
    }
    
    sort(blocks.begin(), blocks.end());
    
    for(size_t i=0; i < blocks.size() && i < 20; i++)
    {
        double time = blocks[i].first;
        size_t block = blocks[i].second;
        
        double gfps = WIDTH * HEIGHT  * 10e-9 / time;
        double gbps = sizeof(float) * WIDTH * HEIGHT * 10e-9 / time;
        printf("block=%i time=%.3f\tGfps=%.3f\tGBps=%.3f\n", block, time, gfps, gbps);
    }
	return 0;
}
