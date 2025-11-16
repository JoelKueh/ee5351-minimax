
#ifndef GPU_LIB_H
#define GPU_LIB_H

#include "gpu_types.cuh"

void perft_gpu(uint32_t *out_nodes, gpu_board_t *in_boards, int depth);

#endif /* GPU_LIB_H */
