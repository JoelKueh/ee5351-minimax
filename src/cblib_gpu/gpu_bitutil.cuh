
#ifndef GPU_BITUTIL_H
#define GPU_BITUTIL_H

#include <stdint.h>

__device__ __forceinline__ uint8_t gpu_peek_rbit(uint64_t bb)
{
    /* Cuda intrinsic for find first set bit. */
    return __ffsll(bb) - 1;
}

__device__ __forceinline__ uint8_t gpu_pop_rbit(uint64_t *__restrict__ bb)
{
    uint8_t idx = gpu_peek_rbit(*bb);
    *bb ^= UINT64_C(1) << idx;
    return idx;
}

__device__ __forceinline__ uint8_t gpu_popcnt(uint64_t bb)
{
    /* Cuda intrinsic for popcnt (return number of set bits in mask). */
    return __popcll(bb);
}

#endif /* GPU_BITUTIL_H */
