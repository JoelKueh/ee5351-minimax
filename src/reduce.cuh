
#ifndef REDUCE_H
#define REDUCE_H

/* TODO: Remove me. */
#include <iostream>
#include <cstdint>

#define REDUCTION_BLOCK_SIZE 1024

__global__ void reduce(uint64_t *out, uint64_t *in, uint32_t n)
{
    __shared__ uint64_t shared[2 * REDUCTION_BLOCK_SIZE];
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

__global__ void reduce_to_u64(uint64_t *out, uint8_t *in, uint32_t n)
{
    __shared__ uint64_t shared[2 * REDUCTION_BLOCK_SIZE];
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

__host__ void launch_reduction(uint64_t *result, uint8_t *data,
        uint32_t n, cudaStream_t s)
{
    /* TODO: Remove me. */
    std::cout << "n: " << n << "\n";
    
    /* Allocate swap buffer for h_data on the GPU. */
    uint64_t *in_data;
    uint64_t *out_data;
    size_t bufsize = ceil((float)n / (2 * REDUCTION_BLOCK_SIZE)) * sizeof(uint64_t);
    cudaMalloc((void **)&in_data, bufsize);
    cudaMalloc((void **)&out_data, bufsize);

    /* First reduction is from uint8_t to uint64_t. */
    dim3 blockDim(REDUCTION_BLOCK_SIZE, 1, 1);
    dim3 gridDim(ceil((float)n / (2 * REDUCTION_BLOCK_SIZE)), 1, 1);
    reduce_to_u64<<<gridDim, blockDim, 0, s>>>(in_data, data, n);
    n = ceil((float)n / (2 * REDUCTION_BLOCK_SIZE));

    /* All remaining reductions are on the provided buffers. */
    int i = 0;
    while (n > 1) {
        /* Launch the kernel to perform the reduction for the current size. */
        dim3 blockDim(REDUCTION_BLOCK_SIZE, 1, 1);
        dim3 gridDim(ceil((float)n / (2 * REDUCTION_BLOCK_SIZE)), 1, 1);
        reduce<<<gridDim, blockDim, 0, s>>>(out_data, in_data, n);

        /* One reduction cuts size of input by twice BLOCK_SIZE. */
        n = ceil((float)n / (2 * REDUCTION_BLOCK_SIZE));

        /* Swap the output and input buffers. */
        uint64_t *tmp = out_data;
        out_data = in_data;
        in_data = tmp;
    }

    /* Copy the reduced sum back to the CPU and free gpu memory. */
    cudaMemcpy(result, in_data, 1 * sizeof(int), cudaMemcpyDeviceToDevice);
    cudaFree(in_data);
    cudaFree(out_data);
}

#endif /* REDUCE_H */
