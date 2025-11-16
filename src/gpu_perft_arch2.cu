
#include <stdint.h>
#include "cblib_gpu/gpu_types.cuh"
#include "cblib_gpu/gpu_lib.h"

/* TODO: Need to implement the cuda based version of perft here. */

/**
 * @breif Takes a vector of input board positions and performs a perft
 * search on them in the GPU to the specified depth. Returning the leaf node
 * counts in out_nodes.
 * @param out_nodes Vector of output leaf node counts.
 * @param in_boards Vector of board positions to do the perft on.
 * @param depth The depth to search to.
 */
__global__ void perft_kernel(uint32_t *out_nodes, gpu_board_t *in_boards, int depth)
{

}

/**
 * @breif Host wrapper for perft_kernel above.
 */
void perft_gpu(uint32_t *out_nodes, gpu_board_t *in_boards, int depth)
{

}
