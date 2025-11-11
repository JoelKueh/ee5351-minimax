
#include "cblib_gpu/gpu_types.h"
#include "cblib_gpu/gpu_gen.h"

/**
 * @breif Counts the number of moves avaliable from a certain position.
 */
__global__ void count_moves()
{

}

/**
 * @breif Performs a scan on the movecount vector returned by count_moves.
 * @param in_counts Buffer of counts to perform the scan on.
 * @param out_scanned Fully scanned buffer.
 */
__global__ void scan_movecounts(
        uint32_t *in_counts,
        uint32_t *out_scanned)
{

}

/**
 * @breif Generates all moves from the specified position and writes them
 * into the specified buffer at the index in indices.
 * @param out_moves A pointer to a buffer to write the list of moves.
 * @param in_write_indices A scanned buffer for output indices in out_moves.
 * @param in_positions A buffer of positions to generate moves for.
 */
__global__ void gen_moves(
        gpu_move_t *out_moves,
        uint32_t *in_write_indices,
        gpu_bitboard_t *in_positions)
{

}

/**
 * @breif Counts the number of moves avaliable from a certain position.
 */
__global__ void make_moves(
        gpu_bitboard_t *out_positions,
        gpu_bitboard_t *in_positions,
        gpu_bitboard_t *in_indices,
        gpu_bitboard_t *in_moves)
{

}

