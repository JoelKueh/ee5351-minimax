
#include "cblib_gpu/gpu_types.cuh"
#include "cblib_gpu/gpu_gen.cuh"

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

/* Option 1: Launch 1 thread per board in the make_moves kernel call.
 *  - More divergence between threads on boards with different move counts.
 *  - Bad coallescense in writes.
 *
 * Option 2: Launch 1 thread per move in the make_moves kernel call.
 *  - Requires expanding scanned array with duplicate data so each thread
 *    knows which board to operate on.
 *  - Has larger memory requirements (4 more bytes per move).
 *  - See https://moderngpu.github.io/intervalmove.html for an example kernel
 *    that could accomplish this. We may steal their implementation.
 */

/** If we go with option 2 above:
 * @breif Expands the scanned array to be used in the make_moves kernel call.
 * @param out_expanded The expanded array.
 * @param in_scanned The scanned array.
 * @param times_duplicated Array of numbers specifying how many times elements
 * in in_scanned should be duplicated.
 */
__global__ void expand(
        uint32_t *out_expanded,
        uint32_t *in_scanned,
        uint8_t *times_duplicated)
{

}

/** Question: One thread per board or one thread per move? Worth trying both?
 * @breif Applies a vector of moves to a vector of positions.
 * @param out_positions A vector of generated positions.
 * @param in_positions A vector of generated 
 */
__global__ void make_moves(
        gpu_bitboard_t *out_positions,
        gpu_bitboard_t *in_positions,
        gpu_bitboard_t *in_indices,
        gpu_bitboard_t *in_moves)
{

}

/**
 * @breif Expands subtrees of a vector of positions on the GPU.
 * @param out_counts Vector of counts returned by the GPU.
 * @param in_positions Vector of positions to expand.
 * @param n Number of positions in the provided vector.
 */
uint64_t perft_subtrees_on_gpu(
        uint64_t *out_counts,
        gpu_bitboard_t *in_positions,
        uint32_t n)
{

}

