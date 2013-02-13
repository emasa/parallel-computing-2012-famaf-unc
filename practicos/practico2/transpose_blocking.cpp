
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include <vector>
#include <algorithm>

#include "transpose_utils.h"
#include "timing.h"

#define HEIGHT (1 << 10)
#define WIDTH HEIGHT

#define MAX_BLOCK (1 << 8)

using namespace std;

void transpose_2d_blocking(size_t width, 
				           size_t height,
				           float * in, 
				           float * out,
			               size_t bx, 
			               size_t by)
{   
    assert(0 < width && 0 < height);
    assert(0 < bx && bx <= width);
    assert(0 < by && by <= height);
    
    for (size_t i = 0 ; i < height; i += by)
    {
        for (size_t j = 0 ; j < width ; j += bx)
        {
            for (size_t k = i; k < i + by && k < height; k++)
            {
                for (size_t l = j; l < j + bx && l < width; l++)
                {
                    AT(out, l, k, height) = AT(in, k, l, width);
                }
            }
        }
    }
}

int main(){

	float * in = (float *) malloc(sizeof(float) * HEIGHT * WIDTH);
	assert(in != NULL);
	float * out = (float *) malloc(sizeof(float) * HEIGHT * WIDTH);
    assert(out != NULL);
    
    vector< pair<double, pair<size_t, size_t> > > blocks;
                    
	init_matrix(WIDTH, HEIGHT, in);
	for(size_t bx=1; bx <= MAX_BLOCK; bx *= 2)
	{
	    for(size_t by=1; by <= MAX_BLOCK; by *= 2)
	    {	
            double t_start = wtime();
            transpose_2d_blocking(WIDTH, HEIGHT, in, out, bx, by);
            double t_total = wtime() - t_start;	        
            
            check_traspose(HEIGHT, WIDTH, out);    
                        
            blocks.push_back(make_pair(t_total, make_pair(by, bx)));
        }
	}
	
	sort(blocks.begin(), blocks.end());
	
	for (size_t i = 0; i < blocks.size() && i < 20; i++)
	{   
	    double time =  blocks[i].first;
	    size_t by =  blocks[i].second.first, bx =  blocks[i].second.second;
	    
        double gfps = WIDTH * HEIGHT  * 10e-9 / time;
        double gbps = sizeof(float) * WIDTH * HEIGHT * 10e-9 / time;
        
        printf("BY=%i\tBX=%i\tBY*BX=%i\ttime=%f\tGfps=%.3f\tGBps=%.3f\n", 
                by, bx, by * bx, time, gfps, gbps);
	}
    
    free(in);
    free(out);
    
	return 0;
}
