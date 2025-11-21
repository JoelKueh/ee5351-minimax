
/* TODO: This file is pretty much just make-unmake. */

#include <string.h>
#include <errno.h>
#include <threads.h>

#include "gpu_lib.h"
#include "gpu_tables.cuh"
#include "gpu_const.cuh"
#include "gpu_move.cuh"
#include "gpu_board.cuh"
#include "gpu_history.cuh"

__device__ void gpu_make(gpu_board_t *board, const gpu_move_t mv)
{
    gpu_history_t old_state = board->hist.data[board->hist.count - 1].hist;
    gpu_hist_ele_t new_ele;
    gpu_mv_flag_t flag = (gpu_mv_flag_t)gpu_mv_get_flags(mv);
    uint8_t to = gpu_mv_get_to(mv);
    uint8_t from = gpu_mv_get_from(mv);

    gpu_ptype_t ptype;
    gpu_ptype_t cap_ptype;
    gpu_history_t new_state = old_state;

    /* Variables for castles. */
    uint8_t rook_from;
    uint8_t rook_to;

    /* Variables for enp. */
    int8_t direction;

    /* TODO: This code has massive divergence. Refactor. */
#if 0
    /* Make the move. */
    switch (flag)
    {
        case GPU_MV_QUIET:
            ptype = gpu_ptype_at_sq(board, from);
            gpu_state_set_captured_piece(&new_state, GPU_PTYPE_EMPTY);
            gpu_state_decay_castle_rights(&new_state, board->turn, to, from);
            gpu_write_piece(board, to, ptype, board->turn);
            gpu_delete_piece(board, from, ptype, board->turn);
            break;
        case GPU_MV_CAPTURE:
            ptype = cb_ptype_at_sq(board, from);
            cap_ptype = cb_ptype_at_sq(board, to);
            cb_hist_set_captured_piece(&new_state, cap_ptype);
            cb_hist_decay_castle_rights(&new_state, board->turn, to, from);
            cb_replace_piece(board, to, ptype, board->turn, cap_ptype, !board->turn);
            cb_delete_piece(board, from, ptype, board->turn);
            break;
        case GPU_MV_DOUBLE_PAWN_PUSH:
            cb_hist_set_enp(&new_state, to & 0b111);
            cb_write_piece(board, to, GPU_PTYPE_PAWN, board->turn);
            cb_delete_piece(board, from, GPU_PTYPE_PAWN, board->turn);
            break;
        case GPU_MV_KING_SIDE_CASTLE:
            rook_from = board->turn ? M_WHITE_KING_SIDE_ROOK_START :
                M_BLACK_KING_SIDE_ROOK_START;
            rook_to = board->turn ? M_WHITE_KING_SIDE_ROOK_TARGET :
                M_BLACK_KING_SIDE_ROOK_TARGET;
            cb_hist_set_captured_piece(&new_state, GPU_PTYPE_EMPTY);
            cb_hist_remove_castle(&new_state, board->turn);
            cb_delete_piece(board, from, GPU_PTYPE_KING, board->turn);
            cb_write_piece(board, to, GPU_PTYPE_KING, board->turn);
            cb_delete_piece(board, rook_from, GPU_PTYPE_ROOK, board->turn);
            cb_write_piece(board, rook_to, GPU_PTYPE_ROOK, board->turn);
            break;
        case GPU_MV_QUEEN_SIDE_CASTLE:
            rook_from = board->turn ? M_WHITE_QUEEN_SIDE_ROOK_START :
                M_BLACK_QUEEN_SIDE_ROOK_START;
            rook_to = board->turn ? M_WHITE_QUEEN_SIDE_ROOK_TARGET :
                M_BLACK_QUEEN_SIDE_ROOK_TARGET;
            cb_hist_set_captured_piece(&new_state, GPU_PTYPE_EMPTY);
            cb_hist_remove_castle(&new_state, board->turn);
            cb_delete_piece(board, from, GPU_PTYPE_KING, board->turn);
            cb_write_piece(board, to, GPU_PTYPE_KING, board->turn);
            cb_delete_piece(board, rook_from, GPU_PTYPE_ROOK, board->turn);
            cb_write_piece(board, rook_to, GPU_PTYPE_ROOK, board->turn);
            break;
        case GPU_MV_ENPASSANT:
            direction = board->turn == GPU_WHITE ? 8 : -8;
            cb_hist_set_captured_piece(&new_state, GPU_PTYPE_PAWN);
            cb_write_piece(board, to, GPU_PTYPE_PAWN, board->turn);
            cb_delete_piece(board, from, GPU_PTYPE_PAWN, board->turn);
            cb_delete_piece(board, to + direction, GPU_PTYPE_PAWN, !board->turn);
            break;
        case GPU_MV_KNIGHT_PROMO:
            cb_hist_set_captured_piece(&new_state, GPU_PTYPE_EMPTY);
            cb_write_piece(board, to, GPU_PTYPE_KNIGHT, board->turn);
            cb_delete_piece(board, from, GPU_PTYPE_PAWN, board->turn);
            break;
        case GPU_MV_BISHOP_PROMO:
            cb_hist_set_captured_piece(&new_state, GPU_PTYPE_EMPTY);
            cb_write_piece(board, to, GPU_PTYPE_BISHOP, board->turn);
            cb_delete_piece(board, from, GPU_PTYPE_PAWN, board->turn);
            break;
        case GPU_MV_ROOK_PROMO:
            cb_hist_set_captured_piece(&new_state, GPU_PTYPE_EMPTY);
            cb_write_piece(board, to, GPU_PTYPE_ROOK, board->turn);
            cb_delete_piece(board, from, GPU_PTYPE_PAWN, board->turn);
            break;
        case GPU_MV_QUEEN_PROMO:
            cb_hist_set_captured_piece(&new_state, GPU_PTYPE_EMPTY);
            cb_write_piece(board, to, GPU_PTYPE_QUEEN, board->turn);
            cb_delete_piece(board, from, GPU_PTYPE_PAWN, board->turn);
            break;
        case GPU_MV_KNIGHT_PROMO_CAPTURE:
            cap_ptype = cb_ptype_at_sq(board, to);
            cb_hist_set_captured_piece(&new_state, cap_ptype);
            cb_hist_decay_castle_rights(&new_state, board->turn, to, from);
            cb_replace_piece(board, to, GPU_PTYPE_KNIGHT, board->turn, cap_ptype, !board->turn);
            cb_delete_piece(board, from, GPU_PTYPE_PAWN, board->turn);
            break;
        case GPU_MV_BISHOP_PROMO_CAPTURE:
            cap_ptype = cb_ptype_at_sq(board, to);
            cb_hist_set_captured_piece(&new_state, cap_ptype);
            cb_hist_decay_castle_rights(&new_state, board->turn, to, from);
            cb_replace_piece(board, to, GPU_PTYPE_BISHOP, board->turn, cap_ptype, !board->turn);
            cb_delete_piece(board, from, GPU_PTYPE_PAWN, board->turn);
            break;
        case GPU_MV_ROOK_PROMO_CAPTURE:
            cap_ptype = cb_ptype_at_sq(board, to);
            cb_hist_set_captured_piece(&new_state, cap_ptype);
            cb_hist_decay_castle_rights(&new_state, board->turn, to, from);
            cb_replace_piece(board, to, GPU_PTYPE_ROOK, board->turn, cap_ptype, !board->turn);
            cb_delete_piece(board, from, GPU_PTYPE_PAWN, board->turn);
            break;
        case GPU_MV_QUEEN_PROMO_CAPTURE:
            cap_ptype = cb_ptype_at_sq(board, to);
            cb_hist_set_captured_piece(&new_state, cap_ptype);
            cb_hist_decay_castle_rights(&new_state, board->turn, to, from);
            cb_replace_piece(board, to, GPU_PTYPE_QUEEN, board->turn, cap_ptype, !board->turn);
            cb_delete_piece(board, from, GPU_PTYPE_PAWN, board->turn);
            break;
    }
#endif

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

