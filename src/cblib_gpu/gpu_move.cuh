
#ifndef GPU_MOVE_H
#define GPU_MOVE_H

#include <stdint.h>
#include <stdbool.h>

#include "gpu_types.cuh"
#include "gpu_const.cuh"

/**
 * Returns the "to" square for the move as a 6-bit integer.
 */
__device__ static inline uint8_t gpu_mv_get_to(gpu_move_t mv)
{
    return mv & GPU_MV_TO_MASK;
}

/**
 * Returns the from square for the move as a 6-bit integer.
 */
__device__ static inline uint8_t gpu_mv_get_from(gpu_move_t mv)
{
    return (mv & GPU_MV_FROM_MASK) >> 6;
}

/**
 * Returns the flags for the move as a cb_move_flags.
 */
__device__ static inline uint16_t gpu_mv_get_flags(gpu_move_t mv)
{
    return mv & GPU_MV_FLAG_MASK;
}

/**
 * Masks together a move from the raw data.
 */
__device__ static inline gpu_move_t gpu_mv_from_data(
        uint16_t from, uint16_t to, uint16_t flags)
{
    return flags | (from << 6) | to;
}

/**
 * Checks if a move was a capture.
 */
__device__ static inline bool gpu_mv_is_cap(gpu_move_t mv)
{
    uint16_t flag = mv & GPU_MV_FLAG_MASK;
    return flag == GPU_MV_CAPTURE || flag & (0b100 << 12);
}


#endif /* GPU_MOVE_H */

