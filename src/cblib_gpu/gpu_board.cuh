
#ifndef GPU_BOARD_H
#define GPU_BOARD_H

#include <stdint.h>
#include <stdbool.h>
#include <inttypes.h>

#include "gpu_types.cuh"
#include "gpu_dbg.cuh"

__device__ __forceinline__ gpu_ptype_t gpu_ptype_at_sq(
        const gpu_board_t *__restrict__ board, uint8_t sq)
{
    gpu_ptype_t ptype = GPU_PTYPE_EMPTY;

    ptype = board->bb.pawns & (UINT64_C(1) << sq) ? GPU_PTYPE_PAWN : ptype;
    ptype = board->bb.knights & (UINT64_C(1) << sq) ? GPU_PTYPE_KNIGHT : ptype;
    ptype = board->bb.bishops & (UINT64_C(1) << sq) ? GPU_PTYPE_BISHOP : ptype;
    ptype = board->bb.rooks & (UINT64_C(1) << sq) ?
        (ptype == GPU_PTYPE_BISHOP ? GPU_PTYPE_QUEEN : GPU_PTYPE_ROOK) : ptype;
    ptype = board->bb.kings & (UINT64_C(1) << sq) ? GPU_PTYPE_KING : ptype;

    return ptype;
}

__device__ __forceinline__ gpu_ptype_t gpu_ptype_at(
        const gpu_board_t *__restrict__ board, uint8_t row, uint8_t col)
{
    return gpu_ptype_at_sq(board, row * 8 + col);
}

__device__ __forceinline__ gpu_color_t gpu_color_at_sq(
        const gpu_board_t *__restrict__ board, uint8_t sq)
{
    return board->bb.color & (UINT64_C(1) << sq) ? GPU_WHITE : GPU_BLACK;
}

__device__ __forceinline__ gpu_color_t gpu_color_at(
        const gpu_board_t *__restrict__ board, uint8_t row, uint8_t col)
{
    return gpu_color_at_sq(board, row * 8 + col);
}

__device__ __forceinline__ void gpu_write_piece(
        gpu_board_t *__restrict__ board, uint8_t sq,
        uint8_t ptype, uint8_t pcolor)
{
    /* Set the piece bitboards. */
    board->bb.pawns |= ptype == GPU_PTYPE_PAWN ? UINT64_C(1) << sq : 0;
    board->bb.knights |= ptype == GPU_PTYPE_KNIGHT ? UINT64_C(1) << sq : 0;
    board->bb.bishops |= ptype == GPU_PTYPE_BISHOP || ptype == GPU_PTYPE_QUEEN
        ? UINT64_C(1) << sq : 0;
    board->bb.rooks |= ptype == GPU_PTYPE_ROOK || ptype == GPU_PTYPE_QUEEN
        ? UINT64_C(1) << sq : 0;
    board->bb.kings |= ptype == GPU_PTYPE_KING ? UINT64_C(1) << sq : 0;

    /* Set the color and occupancy normal bitboards. */
    board->bb.color |= pcolor == GPU_WHITE ? UINT64_C(1) << sq : 0;
    board->bb.occ |= UINT64_C(1) << sq;
}

__device__ __forceinline__ void gpu_delete_piece(
        gpu_board_t *__restrict__ board, uint8_t sq,
        uint8_t ptype, uint8_t pcolor)
{
    /* Set the piece bitboards. */
    board->bb.pawns &= ~(ptype == GPU_PTYPE_PAWN ? UINT64_C(1) << sq : 0);
    board->bb.knights &= ~(ptype == GPU_PTYPE_KNIGHT ? UINT64_C(1) << sq : 0);
    board->bb.bishops &= ~(ptype == GPU_PTYPE_BISHOP || ptype == GPU_PTYPE_QUEEN
        ? UINT64_C(1) << sq : 0);
    board->bb.rooks &= ~(ptype == GPU_PTYPE_ROOK || ptype == GPU_PTYPE_QUEEN
        ? UINT64_C(1) << sq : 0);
    board->bb.kings &= ~(ptype == GPU_PTYPE_KING ? UINT64_C(1) << sq : 0);

    /* Set the color and occupancy normal bitboards. */
    board->bb.color &= ~(UINT64_C(1) << sq);
    board->bb.occ &= ~(UINT64_C(1) << sq);
}

__device__ __forceinline__ void gpu_wipe_board(gpu_board_t *__restrict__ board)
{
    board->bb.color = 0;
    board->bb.pawns = 0;
    board->bb.knights = 0;
    board->bb.bishops = 0;
    board->bb.rooks = 0;
    board->bb.kings = 0;
    board->bb.occ = 0;
}

#endif /* GPU_BOARD_H */

