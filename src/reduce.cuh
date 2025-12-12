
#ifndef REDUCE_H
#define REDUCE_H

#include <cstdint>

#define REDUCTION_BLOCK_SIZE 1024

__global__ void reduce(uint32_t *out, uint32_t *in, uint32_t n)
{
    __shared__ int shared[2 * REDUCTION_BLOCK_SIZE];
    const int tx = threadIdx.x, bx = blockIdx.x;
    const int tid = 2 * blockDim.x * bx + tx;

    /* Copy input to shared memory buffer. */
    shared[tx] = tid < n ? in[tid] : 0;
    shared[tx+REDUCTION_BLOCK_SIZE] = tid + REDUCTION_BLOCK_SIZE < n
        ? in[tid+REDUCTION_BLOCK_SIZE] : 0;

    /* Reduction with early return to give warps a chance to exit. */
    for (uint32_t stride = REDUCTION_BLOCK_SIZE; stride > 0; stride >>= 1) {
        __syncthreads();
        if (tx >= stride)
            return;
        shared[tx] = shared[tx] + shared[tx + stride];
    }

    /* Only the first thread writes the reduced sum. */
    if (tx == 0)
        out[bx] = shared[0];
}

__host__ __device__ void launch_reduction(uint32_t *result, uint32_t *data,
        uint32_t n, cudaStream_t s)
{
    /* Allocate swap buffer for h_data on the GPU. */
    unsigned int *d_data[2];
    cudaMalloc((void **)&(d_data[0]), n * sizeof(uint32_t));
    cudaMalloc((void **)&(d_data[1]), ceil((float)n / REDUCTION_BLOCK_SIZE) * sizeof(int));
    cudaMemcpy(d_data[0], data, n * sizeof(int), cudaMemcpyHostToDevice);

    /* Launch the kernel until we only have one element remaining. */
    int i = 0;
    while (num_elements > 1) {
        /* Launch the kernel to perform the reduction for the current size. */
        dim3 blockDim(BLOCK_SIZE, 1, 1);
        dim3 gridDim(ceil((float)num_elements / BLOCK_SIZE), 1, 1);
        reduction<<<gridDim, blockDim>>>(d_data[i], d_data[i ^ 1], num_elements);

        /* One reduction cuts size of input by twice BLOCK_SIZE. */
        num_elements = ceil((float)num_elements / (2 * BLOCK_SIZE));

        /* Swap the output and input buffers. */
        i ^= 1;
    }

    /* Copy the reduced sum back to the CPU. */
    cudaMemcpy(result, d_data[i], 1 * sizeof(int), cudaMemcpyDeviceToHost);
}

#endif /* REDUCE_H */
