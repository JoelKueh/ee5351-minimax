
#ifndef SCAN_H
#define SCAN_H

#include <stdint.h>

#define SCAN_BLOCK_SIZE 512
#define SCAN_TILE_SIZE (SCAN_BLOCK_SIZE << 1)

/* Struct to hold the intermediate results of a scan. */
typedef struct {
    int num_buffers;            /* Number of buffers in the structure. */
    int counts[7];              /* Counts of non-padded elements in buffer. */
    uint32_t *buffers[7];   /* Array of buffer pointers. */
} scan_buffer_t;

/**
 * @breif Performs an in-place block-wise scan on all elements of data.
 * @param out The block-wise scanned data.
 * @param in The data to scan.
 * @param block_sums array of block sums computed during the scan.
 */
__global__ void scan(uint32_t *out, uint32_t *in, uint32_t *block_sums, uint32_t n)
{
    __shared__ uint32_t s_data[2 * SCAN_BLOCK_SIZE];
    int tx = threadIdx.x, bx = blockIdx.x;

    /* Collaborative shared memory load from input. */
    s_data[tx] = bx * SCAN_TILE_SIZE + tx < n ?
        in[bx * SCAN_TILE_SIZE + tx] : 0;
    s_data[SCAN_BLOCK_SIZE + tx] = bx * SCAN_TILE_SIZE + SCAN_BLOCK_SIZE + tx < n ?
        in[bx * SCAN_TILE_SIZE + SCAN_BLOCK_SIZE + tx] : 0;
    __syncthreads();

    /* Summation part of the scan. */
    for (int stride = 1; stride <= SCAN_BLOCK_SIZE; stride <<= 1) {
        int idx = (tx + 1) * 2 * stride - 1;
        if (idx < 2 * SCAN_BLOCK_SIZE)
            s_data[idx] += s_data[idx - stride];
        __syncthreads();
    }

    /* Distribution part of the scan. */
    for (int stride = (SCAN_BLOCK_SIZE >> 1); stride > 0; stride >>= 1) {
        int idx = (tx + 1) * 2 * stride - 1;
        if (idx + stride < 2 * SCAN_BLOCK_SIZE)
            s_data[idx + stride] += s_data[idx];
        __syncthreads();
    }

    /* Copy back the result. */
    if (bx * SCAN_TILE_SIZE + tx < n)
        out[bx * SCAN_TILE_SIZE + tx] = s_data[tx];
    if (bx * SCAN_TILE_SIZE + SCAN_BLOCK_SIZE + tx < n)
        out[bx * SCAN_TILE_SIZE + SCAN_BLOCK_SIZE + tx] = s_data[SCAN_BLOCK_SIZE + tx];

    /* First thread copys back the block sum. */
    if (tx == 0) {
        block_sums[bx] = s_data[2 * SCAN_BLOCK_SIZE - 1];
    }
}

/**
 * @breif Distribute block sums to all elements in data.
 * @param data The block-wise scanned matrix.
 * @param block_sums The sums of each block in data.
 */
__global__ void distribute(uint32_t *data, uint32_t *block_sums, uint32_t n)
{
    /* Shared memory for the single integer that we will distribute. */
    __shared__ uint32_t s_block_sum;

    /* Load the block sum into shared memory. */
    int sum_idx = blockIdx.x - 1;
    if (threadIdx.x == 0)
        s_block_sum = sum_idx >= 0 ? block_sums[sum_idx] : 0;
    __syncthreads();

    /* Distribute the block sum back to every element in the block. */
    int idx = blockIdx.x * SCAN_TILE_SIZE + threadIdx.x;
    if (idx < n)
        data[idx] += s_block_sum;
}

__host__ void alloc_scan_buf(
        scan_buffer_t *__restrict__ buf, uint32_t n)
{
    int i = 0;

    /* block_sums is padded to even multiples of SCAN_TILE_SIZE at each level of iteration. */
    n = ceil(n / (float)SCAN_TILE_SIZE);
    while (n > 1) {
        /* Extend with zeros to a multiple of tile size. */
        int npadded = ceil(n / (float)SCAN_TILE_SIZE) * SCAN_TILE_SIZE;
        cudaMalloc((void **)&(buf->buffers[i]), npadded * sizeof(uint32_t));
        cudaMemset(buf->buffers[i], 0, npadded * sizeof(uint32_t));

        /* Update arrays and n. */
        buf->counts[i] = n;
        n = ceil(n / (float)SCAN_TILE_SIZE);
        i++;
    }

    /* Set the counts at the last level. */
    buf->counts[i] = 1;
    cudaMalloc((void **)&(buf->buffers[i]), sizeof(uint32_t));
    buf->num_buffers = i + 1;
}

__host__ void free_scan_buf(scan_buffer_t *__restrict__ buf)
{
    /* Free all of the buffers in the scan_buffer_t. */
    for (int i = 0; i < buf->num_buffers; i++) {
        cudaFree(buf->buffers[i]);
    }
}

__host__ void launch_scan(uint32_t *out, uint32_t *in, uint32_t n)
{
    scan_buffer_t buf;
    dim3 blockDim;
    dim3 gridDim;

    /* Preallocate intermediate result buffers. */
    alloc_scan_buf(&buf, n);

    /* Perform the first scan based on the input and output arrays. */
    blockDim = dim3(SCAN_BLOCK_SIZE, 1, 1);
    gridDim = dim3(ceil(n / (float)SCAN_TILE_SIZE), 1, 1);
    scan<<<gridDim, blockDim>>>(out, in, buf.buffers[0], n);

    /* If the computation fits on one block, compute in one shot. */
    if (n <= SCAN_TILE_SIZE) {
        /* TODO: Do I need this synchronization. */
        cudaDeviceSynchronize();
        free_scan_buf(&buf);
        return;
    }

    /* Perform subsequent scans on the block_sums_buffer. */
    for (int i = 1; i < buf.num_buffers; i++) {
        blockDim = dim3(SCAN_BLOCK_SIZE, 1, 1);
        gridDim = dim3(buf.counts[i], 1, 1);
        scan<<<gridDim, blockDim>>>(buf.buffers[i-1],
                buf.buffers[i-1], buf.buffers[i], buf.counts[i-1]);
    }

    /* Preform propagations on the block_sums_buffer. */
    for (int i = buf.num_buffers - 1; i >= 1; i--) {
        blockDim = dim3(SCAN_TILE_SIZE, 1, 1);
        gridDim = dim3(buf.counts[i], 1, 1);
        distribute<<<gridDim, blockDim>>>(buf.buffers[i-1],
                buf.buffers[i], buf.counts[i-1]);
    }

    /* Perform the last distribution into the output array. */
    blockDim = dim3(SCAN_TILE_SIZE, 1, 1);
    gridDim = dim3(buf.counts[0], 1, 1);
    distribute<<<gridDim, blockDim>>>(out, buf.buffers[0], n);

    /* Synchronize and free memory. */
    free_scan_buf(&buf);
}

#endif /* SCAN_H */

