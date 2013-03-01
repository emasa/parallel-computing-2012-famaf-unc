#include <cuda.h>
#include <cutil_inline.h>
#include <stdio.h>

//volatile dice que 
__global__ void add_one( uint * sum) {
     atomicAdd(sum, 1);
}
int main() {
    uint host_sum, *dev_sum;
  
    cutilSafeCall(cudaMalloc((void **) &dev_sum, sizeof(uint)));
    cutilSafeCall(cudaMemset(dev_sum, 0, sizeof(uint)));
 
    dim3 block(1024), grid(1024); // 1024x1024 sumas = 1048576
    add_one<<<grid, block>>>(dev_sum);

    cutilSafeCall(cudaMemcpy(&host_sum, dev_sum, sizeof(uint), cudaMemcpyDefault));
    printf("%u\n", host_sum);

    return 0;
}










