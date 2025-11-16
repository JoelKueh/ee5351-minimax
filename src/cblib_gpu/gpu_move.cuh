
#ifndef GPU_MOVE_H
#define GPU_MOVE_H

#include <stdint.h>
#include <stdbool.h>

#include "gpu_types.cuh"
#include "gpu_const.cuh"

#define GPU_INVALID_MOVE 0b0110111111111111
#define GPU_MV_TO_MASK 0x3F
#define GPU_MV_FROM_MASK (0x3F << 6)
#define GPU_MV_FLAG_MASK (0xF << 12)

/**
 * Enum holding the different flags that a move can contain
 */
typedef enum {
    GPU_MV_QUIET                =  0 << 12,
    GPU_MV_DOUBLE_PAWN_PUSH     =  1 << 12,
    GPU_MV_KING_SIDE_CASTLE     =  2 << 12,
    GPU_MV_QUEEN_SIDE_CASTLE    =  3 << 12,
    GPU_MV_CAPTURE              =  4 << 12,
    GPU_MV_ENPASSANT            =  5 << 12,
    GPU_MV_KNIGHT_PROMO         =  8 << 12,
    GPU_MV_BISHOP_PROMO         =  9 << 12,
    GPU_MV_ROOK_PROMO           = 10 << 12,
    GPU_MV_QUEEN_PROMO          = 11 << 12,
    GPU_MV_KNIGHT_PROMO_CAPTURE = 12 << 12,
    GPU_MV_BISHOP_PROMO_CAPTURE = 13 << 12,
    GPU_MV_ROOK_PROMO_CAPTURE   = 14 << 12,
    GPU_MV_QUEEN_PROMO_CAPTURE  = 15 << 12
} cb_mv_flag_t;

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
        gpu_mvlst_t *restrict mvlst)
{
    return mvlst->head;
}

/**
 * Clears the move list.
 */
__device__ static inline void cb_mvlst_clear(
        gpu_mvlst_t *restrict mvlst)
{
    mvlst->head = 0;
}

/**
 * Pushes an element to the move list.
 */
__device__ static inline void cb_mvlst_push(
        gpu_mvlst_t *restrict mvlst, gpu_move_t move)
{
    mvlst->moves[mvlst->head++] = move;
}

/**
 * Pops one elemnt off the move list.
 * This function does not perform bounds checking.
 * User must guarantee that move list has elements in it.
 */
__device__ static inline gpu_move_t cb_mvlst_pop(
        gpu_mvlst_t *restrict mvlst)
{
    return mvlst->moves[--mvlst->head];
}

/**
 * Returns the move at a specified index.
 */
__device__ static inline gpu_move_t cb_mvlst_at(
        gpu_mvlst_t *restrict mvlst, uint8_t idx)
{
    return mvlst->moves[idx];
}

#endif /* GPU_MOVE_H */

