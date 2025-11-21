
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
 * Returns the size of a move list.
 */
__device__ static inline uint8_t gpu_mvlst_size(
        gpu_mvlst_t *__restrict__ mvlst)
{
    return mvlst->head;
}

/**
 * Clears the move list.
 */
__device__ static inline void gpu_mvlst_clear(
        gpu_mvlst_t *__restrict__ mvlst)
{
    mvlst->head = 0;
}

/**
 * Pushes an element to the move list.
 */
__device__ static inline void gpu_mvlst_push(
        gpu_mvlst_t *__restrict__ mvlst, gpu_move_t move)
{
    mvlst->moves[mvlst->head++] = move;
}

/**
 * Returns the move at a specified index.
 */
__device__ static inline gpu_move_t gpu_mvlst_at(
        gpu_move_t *__restrict__ moves, uint32_t *__restrict__ offset, uint8_t idx)
{
    return mvlst->moves[idx];
}

#endif /* GPU_MOVE_H */

