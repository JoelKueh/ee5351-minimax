
/* TODO: Need to implement the cuda based version of perft here. */

#include <stdint.h>
#include "cblib_gpu/gpu_types.h"
#include "cblib_gpu/gpu_lib.h"

/**
 * @breif Takes a vector of input board positions and performs a perft
 * search on them in the GPU to the specified depth. Returning the leaf node
 * counts in out_nodes.
 * @param out_nodes Vector of output leaf node counts.
 * @param in_boards Vector of board positions to do the perft on.
 * @param depth The depth to search to.
 */
__global__ void perft_gpu(uint32_t *out_nodes, gpu_board_t *in_boards, int depth)
{

}
