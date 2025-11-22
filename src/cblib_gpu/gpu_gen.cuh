
#ifndef GPU_GEN_H
#define GPU_GEN_H

#include "gpu_types.cuh"

/**
 * @breif Counts the number of moves avaliable at a given position.
 * @param board The current board position.
 * @return The number of moves.
 */
__device__ int count_moves(gpu_board_t *__restrict__ board);

/**
 * @breif Generates all legal moves from a given position.
 * @param moves An output array of moves to write to.
 * @param offset The offset into the output array that to start writing moves.
 * @param board The current board position
 */
__device__ void gen_moves(
        gpu_move_t *__restrict__ moves, uint32_t *__restrict__ offset,
        gpu_board_t *__restrict__ board);

#endif /* GPU_GEN_H */
