
#ifndef GPU_SEARCH_STRUCT_H
#define GPU_SEARCH_STRUCT_H

#include "gpu_types.cuh"

__device__ static inline void gpu_mv_write_buf_push(
        gpu_mv_write_buf_t *__restrict__ buf, gpu_move_t move)
{
    ss->global->moves[threadIdx.x][ss->depth][ss->offset++] = move;
}

__device__ static inline void gpu_(
        gpu_search_struct_t *__restrict__ ss,
        gpu_hist_ele_t history)
{
    ss->global->history[ss->depth][threadIdx.x] = history;
}

#endif /* GPU_SEARCH_STRUCT_H */

