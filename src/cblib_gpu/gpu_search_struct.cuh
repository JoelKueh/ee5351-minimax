
#ifndef GPU_SEARCH_STRUCT_H
#define GPU_SEARCH_STRUCT_H

#include "gpu_types.cuh"

__device__ static inline void gpu_ss_push_move(
        gpu_search_struct_t *__restrict__ ss, gpu_move_t move)
{
    ss->positions[ss->depth].moves[threadIdx.x & 0b11111][ss->offset++] = move;
}

__device__ static inline gpu_move_t gpu_ss_get_move(
        gpu_search_struct_t *__restrict__ ss, gpu_move_t move, uint8_t index)
{
    return ss->positions[ss->depth].moves[threadIdx.x & 0b11111][index];
}

__device__ static inline void gpu_ss_descend(
        gpu_search_struct_t *__restrict__ ss, gpu_hist_ele_t hist_ele)
{
    ss->positions[ss->depth].move_counts[threadIdx.x & 0b11111] = ss->offset;
    ss->positions[ss->depth].hist_ele[threadIdx.x & 0b11111] = hist_ele;
    ss->depth++;
    ss->offset = 0;
}

__device__ static inline gpu_hist_ele_t gpu_ss_ascend(
        gpu_search_struct_t *__restrict__ ss)
{
    ss->depth--;
    ss->offset = ss->positions[ss->depth].move_counts[threadIdx.x & 0b11111];
    return ss->positions[ss->depth].hist_ele[threadIdx.x & 0b11111];
}

#endif /* GPU_SEARCH_STRUCT_H */

