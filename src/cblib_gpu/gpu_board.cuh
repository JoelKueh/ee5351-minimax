
#ifndef GPU_BOARD_H
#define GPU_BOARD_H

#include <stdint.h>
#include <stdbool.h>

#include "gpu_types.cuh"

__device__ static inline gpu_ptype_t gpu_ptype_at_sq(
        const gpu_board_t *__restrict__ board, uint8_t sq)
{
    gpu_ptype_t ptype = GPU_PTYPE_EMPTY;

    ptype = GPU_BB_PAWNS(board->bb, board->turn) & (UINT64_C(1) << sq) ?
        GPU_PTYPE_PAWN : ptype;
    ptype = GPU_BB_KNIGHTS(board->bb, board->turn) & (UINT64_C(1) << sq) ?
        GPU_PTYPE_KNIGHT : ptype;
    ptype = GPU_BB_B_AND_Q(board->bb, board->turn) & (UINT64_C(1) << sq) ?
        GPU_PTYPE_BISHOP : ptype;
    ptype = GPU_BB_R_AND_Q(board->bb, board->turn) & (UINT64_C(1) << sq) ?
        (ptype == GPU_PTYPE_BISHOP ? GPU_PTYPE_QUEEN : GPU_PTYPE_ROOK) : ptype;
    ptype = GPU_BB_KINGS(board->bb, board->turn) & (UINT64_C(1) << sq) ?
        GPU_PTYPE_KING : ptype;

    return ptype;
}

__device__ static inline gpu_ptype_t gpu_ptype_at(
        const gpu_board_t *__restrict__ board, uint8_t row, uint8_t col)
{
    return gpu_ptype_at_sq(board, row * 8 + col);
}

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

__device__ static inline void gpu_write_piece(
        gpu_board_t *__restrict__ board, uint8_t sq,
        uint8_t ptype, uint8_t pcolor)
{
    /* Queens are split among bishop and rook bitboards. */
    if (ptype == GPU_PTYPE_QUEEN) {
        board->bb.piece[GPU_PTYPE_BISHOP] |= UINT64_C(1) << sq;
        board->bb.piece[GPU_PTYPE_ROOK] |= UINT64_C(1) << sq;
    } else if (ptype == GPU_PTYPE_KING) {
        board->bb.piece[4] |= UINT64_C(1) << sq;
    } else {
        board->bb.piece[ptype] |= UINT64_C(1) << sq;
    }
    /* TODO: Remove me. */
    printf("Sq: %d, Type: %d\n", sq, ptype);

    /* Set the color and occupancy normal bitboards. */
    if (pcolor == GPU_WHITE)
        board->bb.color |= UINT64_C(1) << sq;
    board->bb.occ |= UINT64_C(1) << sq;
}

__device__ static inline void gpu_delete_piece(
        gpu_board_t *__restrict__ board, uint8_t sq,
        uint8_t ptype, uint8_t pcolor)
{
    /* Queens are split among bishop and rook bitboards. */
    if (ptype == GPU_PTYPE_QUEEN) {
        board->bb.piece[GPU_PTYPE_BISHOP] &= ~(UINT64_C(1) << sq);
        board->bb.piece[GPU_PTYPE_ROOK] &= ~(UINT64_C(1) << sq);
    } else if (ptype == GPU_PTYPE_KING) {
        board->bb.piece[4] &= ~(UINT64_C(1) << sq);
    } else {
        board->bb.piece[ptype] &= ~(UINT64_C(1) << sq);
    }

    /* Set the color and occupancy normal bitboards. */
    if (pcolor == GPU_WHITE)
        board->bb.color &= ~(UINT64_C(1) << sq);
    board->bb.occ &= ~(UINT64_C(1) << sq);
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

