
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
__device__ void gpu_gen_moves(
        gpu_search_struct_t *__restrict__ ss,
        gpu_board_t *__restrict__ board, gpu_state_tables_t *__restrict__ state);

/**
 * @breif Generates the state tables for a given position.
 * @param moves An output array of moves to write to.
 * @param offset The offset into the output array that to start writing moves.
 * @param board The current board position
 */
__device__ void gpu_gen_board_tables(
        gpu_board_t *__restrict__ board, gpu_state_tables_t *__restrict__ state);

#endif /* GPU_GEN_H */
