
#ifndef REDUCE_H
#define REDUCE_H

#include <cstdint>

#define REDUCTION_BLOCK_SIZE 1024

__global__ void reduce(uint64_t *out, uint64_t *in, uint32_t n)
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

__global__ void u64_sum_bytes(uint64_t *data, uint32_t n)
{
    const int tid = blockDim.x * blockIdx.x + threadIdx.x;
    uint64_t original;
    uint64_t sum = 0;

    /* Guard for out of bounds access. */
    if (tid > n)
        return;

    /* Read the value from data. */
    original = data[tid];

    /* Sum over all bytes in the u64. */
    for (int i = 0; i < 8; i++) {
        sum += original & 0xFF;
        original >>= 8;
    }
    
    /* Write the result. */
    data[tid] = sum;
}

__host__ void launch_reduction(uint64_t *result, uint8_t *data,
        uint32_t n, cudaStream_t s)
{
    /* Cast the input data to a uint64_t*. */
    uint64_t *u64_data = (uint64_t*)data;

    /* Compute the sum of all bytes in the u64. */
    n >>= 3;
    dim3 blockDim(REDUCTION_BLOCK_SIZE, 1, 1);
    dim3 gridDim(ceil((float)n / REDUCTION_BLOCK_SIZE), 1, 1);
    u64_sum_bytes<<<gridDim, blockDim, 0, s>>>(u64_data, n);

    /* Allocate swap buffer for h_data on the GPU. */
    uint64_t *d_data[2];
    cudaMalloc((void **)&(d_data[0]), n * sizeof(uint64_t));
    cudaMalloc((void **)&(d_data[1]), ceil((float)n / REDUCTION_BLOCK_SIZE) * sizeof(uint64_t));
    d_data[0] = u64_data;

    /* Launch the kernel until we only have one element remaining. */
    int i = 0;
    while (n > 1) {
        /* Launch the kernel to perform the reduction for the current size. */
        dim3 blockDim(REDUCTION_BLOCK_SIZE, 1, 1);
        dim3 gridDim(ceil((float)n / REDUCTION_BLOCK_SIZE), 1, 1);
        reduce<<<gridDim, blockDim>>>(d_data[i], d_data[i ^ 1], n);

        /* One reduction cuts size of input by twice BLOCK_SIZE. */
        n = ceil((float)n / (2 * REDUCTION_BLOCK_SIZE));

        /* Swap the output and input buffers. */
        i ^= 1;
    }

    /* Copy the reduced sum back to the CPU. */
    cudaMemcpy(result, d_data[i], 1 * sizeof(int), cudaMemcpyDeviceToHost);
}

#endif /* REDUCE_H */
