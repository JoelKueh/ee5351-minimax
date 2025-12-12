
#ifndef PERFT_GPU_H
#define PERFT_GPU_H


#ifdef __cplusplus
extern "C" {
#endif

#include <cb_types.h>

void cblib_gpu_init();
void cblib_gpu_free();
int perft_gpu_bfs(cb_board_t *board, int depth);

#ifdef __cplusplus
}
#endif

#endif /* PERFT_GPU_H */
