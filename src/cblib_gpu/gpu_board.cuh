
#ifndef GPU_BOARD_H
#define GPU_BOARD_H

#include <stdint.h>
#include <stdbool.h>

#include "cb_types.h"
#include "gpu_types.cuh"

__device__ static inline cb_color_t cb_color_at_sq(
        const gpu_board_t *restrict board, uint8_t sq)
{
    return board->bb.color[CB_WHITE] & (UINT64_C(1) << sq) ? CB_WHITE : CB_BLACK;
}

__device__ static inline cb_color_t cb_color_at(
        const gpu_board_t *restrict board, uint8_t row, uint8_t col)
{
    return cb_color_at_sq(board, row * 8 + col);
}

/* Functions for manipulating the board representation. */
__device__ static inline void cb_replace_piece(
        gpu_board_t *restrict board, uint8_t sq, uint8_t ptype, uint8_t pcolor,
        uint8_t old_ptype, uint8_t old_pcolor)
{
    /* Unset the bits for the piece type. */
    board->bb.piece[ptype] |= UINT64_C(1) << sq;
    board->bb.piece[old_ptype] &= ~(UINT64_C(1) << sq);

    /* Update the color bitboard based on the new type and color. */
    if (pcolor == CB_WHITE)
        board->bb.color |= UINT64_C(1) << sq;
    if (old_pcolor == CB_WHITE)
        board->bb.color &= ~(UINT64_C(1) << sq);
}

__device__ static inline void cb_write_piece(
        gpu_board_t *restrict board, uint8_t sq, uint8_t ptype, uint8_t pcolor)
{
    board->bb.piece[ptype] |= UINT64_C(1) << sq;
    if (pcolor == CB_WHITE)
        board->bb.color |= UINT64_C(1) << sq;
}

__device__ static inline void cb_delete_piece(
        gpu_board_t *restrict board, uint8_t sq, uint8_t ptype, uint8_t pcolor)
{
    board->bb.piece[ptype] &= ~(UINT64_C(1) << sq);
    if (pcolor == CB_WHITE)
        board->bb.color &= ~(UINT64_C(1) << sq);
}

__device__ static inline void cb_wipe_board(gpu_board_t *restrict board)
{
    board->bb.color = 0;
    board->bb.piece[0] = 0;
    board->bb.piece[1] = 0;
    board->bb.piece[2] = 0;
    board->bb.piece[3] = 0;
    board->bb.piece[4] = 0;
}

#endif /* GPU_BOARD_H */
