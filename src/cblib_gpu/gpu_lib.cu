
/* TODO: This file is pretty much just make-unmake. */

#include <threads.h>

#include "gpu_lib.h"
#include "gpu_tables.cuh"
#include "gpu_const.cuh"
#include "gpu_move.cuh"
#include "gpu_board.cuh"
#include "gpu_history.cuh"

__device__ void gpu_make(gpu_board_t *board, const gpu_move_t mv)
{
    /* Fields of the move and variables for board state. */
    gpu_history_t old_state = board->state;
    gpu_hist_ele_t new_ele;
    gpu_mv_flag_t flag = (gpu_mv_flag_t)gpu_mv_get_flags(mv);
    uint8_t to = gpu_mv_get_to(mv);
    uint8_t from = gpu_mv_get_from(mv);

    /* Variables for all moves. */
    gpu_ptype_t ptype;
    gpu_ptype_t new_ptype;
    gpu_ptype_t cap_ptype;
    gpu_history_t new_state = old_state;

    /* Variables for castles. */
    uint8_t rook_from;
    uint8_t rook_to;

    /* Variables for enp. */
    int8_t direction;

    /* Handle enpassant separately (it's rare so divergence is fine). */
    if (flag == GPU_MV_ENPASSANT) {
        direction = board->turn == GPU_WHITE ? 8 : -8;
        gpu_state_set_captured_piece(&new_state, GPU_PTYPE_PAWN);
        gpu_write_piece(board, to, GPU_PTYPE_PAWN, board->turn);
        gpu_delete_piece(board, from, GPU_PTYPE_PAWN, board->turn);
        gpu_delete_piece(board, to + direction, GPU_PTYPE_PAWN, !board->turn);
        return;
    }

    /* Read the piece type from the board. */
    ptype = gpu_ptype_at_sq(board, from);
    cap_ptype = gpu_ptype_at_sq(board, to);

    /* If a piece was captured, set it in the board state. */
    gpu_state_set_captured_piece(&new_state, cap_ptype);
    gpu_state_decay_castle_rights(&new_state, board->turn, to, from);

    /* Piece type changes if this is a promotion. Remember that
     *  - Flag types are sequential in lowest 3 bits.
     *  - Only promos have the 4th bit of the flag set. */
    new_ptype = ptype + 1 + ((flag & 0b111 << 12) >> 12);
    new_ptype = flag & (0b1000 << 12) ? ptype : new_ptype;

    /* Move the piece from its previous position to its new position. */
    if (cap_ptype != GPU_PTYPE_EMPTY)
        gpu_delete_piece(board, to, ptype, board->turn);
    gpu_write_piece(board, to, new_ptype, board->turn);
    gpu_delete_piece(board, from, ptype, board->turn);

    /* Set the enpassant field if this was a double pawn push. */
    if (flag == GPU_MV_DOUBLE_PAWN_PUSH)
        gpu_state_set_enp(&new_state, to & 0b111);

    /* Extra work for king side castling. */
    if (flag == GPU_MV_KING_SIDE_CASTLE) {
        rook_from = board->turn ? M_WHITE_KING_SIDE_ROOK_START :
            M_BLACK_KING_SIDE_ROOK_START;
        rook_to = board->turn ? M_WHITE_KING_SIDE_ROOK_TARGET :
            M_BLACK_KING_SIDE_ROOK_TARGET;
        gpu_delete_piece(board, rook_from, GPU_PTYPE_ROOK, board->turn);
        gpu_write_piece(board, rook_to, GPU_PTYPE_ROOK, board->turn);
    }

    /* Extra work for queen side castling. */
    if (flag == GPU_MV_QUEEN_SIDE_CASTLE) {
        rook_from = board->turn ? M_WHITE_QUEEN_SIDE_ROOK_START :
            M_BLACK_QUEEN_SIDE_ROOK_START;
        rook_to = board->turn ? M_WHITE_QUEEN_SIDE_ROOK_TARGET :
            M_BLACK_QUEEN_SIDE_ROOK_TARGET;
        gpu_delete_piece(board, rook_from, GPU_PTYPE_ROOK, board->turn);
        gpu_write_piece(board, rook_to, GPU_PTYPE_ROOK, board->turn);
    }

    /* TODO: Figure out implementation for this. */

    /* Save the new state to the stack. */
    board->turn = !board->turn;
    new_ele.hist = new_state;
    new_ele.move = mv;
    cb_hist_stack_push(&board->hist, new_ele);
}

__device__ void gpu_unmake(gpu_board_t *board)
{
    gpu_hist_ele_t old_ele = cb_hist_stack_pop(&board->hist);
    gpu_hist_ele_t new_ele;
    gpu_mv_flag_t flag = gpu_mv_get_flags(old_ele.move);
    uint8_t to = gpu_mv_get_to(old_ele.move);
    uint8_t from = gpu_mv_get_from(old_ele.move);

    gpu_ptype_t ptype;
    gpu_ptype_t cap_ptype;

    /* Variables for castles. */
    uint8_t rook_from;
    uint8_t rook_to;

    /* Variables for enp. */
    int8_t direction;

    /* TODO: This code has massive divergence. Refactor. */
#if 0
    /* Unmake the move. */
    board->turn = !board->turn;
    switch (flag) {
        case GPU_MV_QUIET:
        case GPU_MV_DOUBLE_PAWN_PUSH:
            ptype = cb_ptype_at_sq(board, to);
            cb_write_piece(board, from, ptype, board->turn);
            cb_delete_piece(board, to, ptype, board->turn);
            break;
        case GPU_MV_CAPTURE:
            ptype = cb_ptype_at_sq(board, to);
            cap_ptype = cb_hist_get_captured_piece(&old_ele.hist);
            cb_write_piece(board, from, ptype, board->turn);
            cb_replace_piece(board, to, cap_ptype, !board->turn, ptype, board->turn);
            break;
        case GPU_MV_KING_SIDE_CASTLE:
            rook_from = board->turn ? M_WHITE_KING_SIDE_ROOK_START :
                M_BLACK_KING_SIDE_ROOK_START;
            rook_to = board->turn ? M_WHITE_KING_SIDE_ROOK_TARGET :
                M_BLACK_KING_SIDE_ROOK_TARGET;
            cb_write_piece(board, from, GPU_PTYPE_KING, board->turn);
            cb_delete_piece(board, to, GPU_PTYPE_KING, board->turn);
            cb_write_piece(board, rook_from, GPU_PTYPE_ROOK, board->turn);
            cb_delete_piece(board, rook_to, GPU_PTYPE_ROOK, board->turn);
            break;
        case GPU_MV_QUEEN_SIDE_CASTLE:
            rook_from = board->turn ? M_WHITE_QUEEN_SIDE_ROOK_START :
                M_BLACK_QUEEN_SIDE_ROOK_START;
            rook_to = board->turn ? M_WHITE_QUEEN_SIDE_ROOK_TARGET :
                M_BLACK_QUEEN_SIDE_ROOK_TARGET;
            cb_write_piece(board, from, GPU_PTYPE_KING, board->turn);
            cb_delete_piece(board, to, GPU_PTYPE_KING, board->turn);
            cb_write_piece(board, rook_from, GPU_PTYPE_ROOK, board->turn);
            cb_delete_piece(board, rook_to, GPU_PTYPE_ROOK, board->turn);
            break;
        case GPU_MV_ENPASSANT:
            direction = board->turn ? 8 : -8;
            cb_write_piece(board, from, GPU_PTYPE_PAWN, board->turn);
            cb_delete_piece(board, to, GPU_PTYPE_PAWN, board->turn);
            cb_write_piece(board, to + direction, GPU_PTYPE_PAWN, !board->turn);
            break;
        case GPU_MV_KNIGHT_PROMO:
            cb_write_piece(board, from, GPU_PTYPE_PAWN, board->turn);
            cb_delete_piece(board, to, GPU_PTYPE_KNIGHT, board->turn);
            break;
        case GPU_MV_BISHOP_PROMO:
            cb_write_piece(board, from, GPU_PTYPE_PAWN, board->turn);
            cb_delete_piece(board, to, GPU_PTYPE_BISHOP, board->turn);
            break;
        case GPU_MV_ROOK_PROMO:
            cb_write_piece(board, from, GPU_PTYPE_PAWN, board->turn);
            cb_delete_piece(board, to, GPU_PTYPE_ROOK, board->turn);
            break;
        case GPU_MV_QUEEN_PROMO:
            cb_write_piece(board, from, GPU_PTYPE_PAWN, board->turn);
            cb_delete_piece(board, to, GPU_PTYPE_QUEEN, board->turn);
            break;
        case GPU_MV_KNIGHT_PROMO_CAPTURE:
            cap_ptype = cb_hist_get_captured_piece(&old_ele.hist);
            cb_write_piece(board, from, GPU_PTYPE_PAWN, board->turn);
            cb_replace_piece(board, to, cap_ptype, !board->turn, GPU_PTYPE_KNIGHT, board->turn);
            break;
        case GPU_MV_BISHOP_PROMO_CAPTURE:
            cap_ptype = cb_hist_get_captured_piece(&old_ele.hist);
            cb_write_piece(board, from, GPU_PTYPE_PAWN, board->turn);
            cb_replace_piece(board, to, cap_ptype, !board->turn, GPU_PTYPE_BISHOP, board->turn);
            break;
        case GPU_MV_ROOK_PROMO_CAPTURE:
            cap_ptype = cb_hist_get_captured_piece(&old_ele.hist);
            cb_write_piece(board, from, GPU_PTYPE_PAWN, board->turn);
            cb_replace_piece(board, to, cap_ptype, !board->turn, GPU_PTYPE_ROOK, board->turn);
            break;
        case GPU_MV_QUEEN_PROMO_CAPTURE:
            cap_ptype = cb_hist_get_captured_piece(&old_ele.hist);
            cb_write_piece(board, from, GPU_PTYPE_PAWN, board->turn);
            cb_replace_piece(board, to, cap_ptype, !board->turn, GPU_PTYPE_QUEEN, board->turn);
            break;
    }
#endif
}

