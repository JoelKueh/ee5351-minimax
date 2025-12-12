
#ifndef GPU_SEARCH_STRUCT_H
#define GPU_SEARCH_STRUCT_H

#include "gpu_types.cuh"
#include "gpu_gen.cuh"

/* TODO: Fix me. Ugly function prototypes to fix circular dependencies. */
__device__ __forceinline__ void gpu_make(
        gpu_search_struct_t *__restrict__ ss, gpu_board_t *__restrict__ board,
        const gpu_move_t mv);
__device__ __forceinline__ void gpu_unmake(gpu_search_struct_t *__restrict__ ss,
        gpu_board_t *__restrict__ board);

__device__ __forceinline__ void gpu_ss_push_move(
        gpu_search_struct_t *__restrict__ ss, gpu_move_t move)
{
    ss->moves[ss->count++] = move;
}

__device__ __forceinline__ gpu_move_t gpu_ss_get_move(
        gpu_search_struct_t *__restrict__ ss, uint8_t index)
{
    return ss->moves[index];
}

#endif /* GPU_SEARCH_STRUCT_H */

