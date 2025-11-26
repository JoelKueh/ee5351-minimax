
#ifndef GPU_SEARCH_STRUCT_H
#define GPU_SEARCH_STRUCT_H

#include "gpu_types.cuh"
#include "gpu_gen.cuh"

/* TODO: Fix me. Ugly function prototypes to fix circular dependencies. */
__device__ static inline void gpu_make(
        gpu_search_struct_t *__restrict__ ss, gpu_board_t *__restrict__ board,
        const gpu_move_t mv);
__device__ void gpu_unmake(gpu_search_struct_t *__restrict__ ss,
        gpu_board_t *__restrict__ board);

__device__ static inline void gpu_ss_push_move(
        gpu_search_struct_t *__restrict__ ss, gpu_move_t move)
{
    ss->positions[ss->depth].moves[ss->move_counts[ss->depth]++]
        [threadIdx.x & 0b11111] = move;
}

__device__ static inline gpu_move_t gpu_ss_get_move(
        gpu_search_struct_t *__restrict__ ss, uint8_t index)
{
    return ss->positions[ss->depth].moves[index][threadIdx.x & 0b11111];
}

__device__ static inline void gpu_ss_descend(
        gpu_search_struct_t *__restrict__ ss, gpu_hist_ele_t hist_ele)
{
    /* Update history in the global history structure. */
    ss->positions[ss->depth].hist_ele[threadIdx.x & 0b11111] = hist_ele;

    /* Update depth. */
    ss->depth++;
    ss->move_counts[ss->depth] = 0;
    ss->move_idx[ss->depth] = 0;
}

__device__ static inline gpu_hist_ele_t gpu_ss_ascend(
        gpu_search_struct_t *__restrict__ ss)
{
    ss->depth--;
    return ss->positions[ss->depth].hist_ele[threadIdx.x & 0b11111];
}

__device__ static inline bool gpu_all_nodes_traversed(
        gpu_search_struct_t *__restrict__ ss)
{
    return ss->move_idx[ss->depth] == ss->move_counts[ss->depth];
}

__device__ static inline void gpu_traverse_to_next_child(
        gpu_search_struct_t *__restrict__ ss,
        gpu_board_t *__restrict__ board)
{
    gpu_move_t mv = ss->positions[ss->depth].moves[ss->move_idx[ss->depth]++]
        [threadIdx.x & 0b11111];
    gpu_make(ss, board, mv);
}

__device__ static inline bool gpu_traversal_complete(
        gpu_search_struct_t *__restrict__ ss)
{
    return ss->move_idx[0] == ss->move_counts[0];
}

#endif /* GPU_SEARCH_STRUCT_H */

