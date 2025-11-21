
#ifndef GPU_BOARD_H
#define GPU_BOARD_H

#include <stdint.h>
#include <stdbool.h>

#include "gpu_types.cuh"

__device__ static inline gpu_color_t gpu_color_at_sq(
        const gpu_board_t *__restrict__ board, uint8_t sq)
{
    return board->bb.color & (UINT64_C(1) << sq) ? GPU_WHITE : GPU_BLACK;
}

__device__ static inline gpu_color_t gpu_color_at(
        const gpu_board_t *__restrict__ board, uint8_t row, uint8_t col)
{
    return gpu_color_at_sq(board, row * 8 + col);
}

/* Functions for manipulating the board representation. */
__device__ static inline void gpu_replace_piece(
        gpu_board_t *__restrict__ board, uint8_t sq, uint8_t ptype,
        uint8_t pcolor, uint8_t old_ptype, uint8_t old_pcolor)
{
    /* Unset the bits for the piece type. */
    board->bb.piece[ptype] |= UINT64_C(1) << sq;
    board->bb.piece[old_ptype] &= ~(UINT64_C(1) << sq);

    /* Update the color bitboard based on the new type and color. */
    if (pcolor == GPU_WHITE)
        board->bb.color |= UINT64_C(1) << sq;
    if (old_pcolor == GPU_WHITE)
        board->bb.color &= ~(UINT64_C(1) << sq);
}

__device__ static inline void gpu_write_piece(
        gpu_board_t *__restrict__ board, uint8_t sq,
        uint8_t ptype, uint8_t pcolor)
{
    board->bb.piece[ptype] |= UINT64_C(1) << sq;
    if (pcolor == GPU_WHITE)
        board->bb.color |= UINT64_C(1) << sq;
}

__device__ static inline void gpu_delete_piece(
        gpu_board_t *__restrict__ board, uint8_t sq,
        uint8_t ptype, uint8_t pcolor)
{
    board->bb.piece[ptype] &= ~(UINT64_C(1) << sq);
    if (pcolor == GPU_WHITE)
        board->bb.color &= ~(UINT64_C(1) << sq);
}

__device__ static inline void gpu_wipe_board(gpu_board_t *__restrict__ board)
{
    board->bb.color = 0;
    board->bb.piece[0] = 0;
    board->bb.piece[1] = 0;
    board->bb.piece[2] = 0;
    board->bb.piece[3] = 0;
    board->bb.piece[4] = 0;
}

#endif /* GPU_BOARD_H */

