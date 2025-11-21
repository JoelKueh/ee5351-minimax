
#ifndef GPU_GEN_H
#define GPU_GEN_H

#include "gpu_types.cuh"

/**
 * @breif Counts the number of pawn moves avaliable at a given position.
 * @param pawns The list of pawns to moves.
 * @param color The color of the pawns to move (e.g., who's turn is it).
 */
__device__ int count_pawn_moves(uint64_t pawns, gpu_color_t color);

/**
 * @breif Generates pawn moves from a specific position.
 */
__device__ int gen_pawn_moves(gpu_move_t *out_moves, uint64_t pawns,
        gpu_color_t color);

#endif /* GPU_GEN_H. */
