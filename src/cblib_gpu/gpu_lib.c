
/* TODO: This file is pretty much just make-unmake. */

#include <string.h>
#include <errno.h>
#include <threads.h>

#include "cb_lib.h"
#include "cb_tables.h"
#include "cb_const.h"
#include "cb_move.h"
#include "cb_board.h"
#include "cb_history.h"

void cb_make(cb_board_t *board, const cb_move_t mv)
{
    cb_history_t old_state = board->hist.data[board->hist.count - 1].hist;
    cb_hist_ele_t new_ele;
    cb_mv_flag_t flag = cb_mv_get_flags(mv);
    uint8_t to = cb_mv_get_to(mv);
    uint8_t from = cb_mv_get_from(mv);

    cb_ptype_t ptype;
    cb_ptype_t cap_ptype;
    cb_history_t new_state = old_state;

    /* Variables for castles. */
    uint8_t rook_from;
    uint8_t rook_to;

    /* Variables for enp. */
    int8_t direction;

    /* Make the move. */
    switch (flag)
    {
        case CB_MV_QUIET:
            ptype = cb_ptype_at_sq(board, from);
            cb_hist_set_captured_piece(&new_state, CB_PTYPE_EMPTY);
            cb_hist_decay_castle_rights(&new_state, board->turn, to, from);
            cb_write_piece(board, to, ptype, board->turn);
            cb_delete_piece(board, from, ptype, board->turn);
            break;
        case CB_MV_CAPTURE:
            ptype = cb_ptype_at_sq(board, from);
            cap_ptype = cb_ptype_at_sq(board, to);
            cb_hist_set_captured_piece(&new_state, cap_ptype);
            cb_hist_decay_castle_rights(&new_state, board->turn, to, from);
            cb_replace_piece(board, to, ptype, board->turn, cap_ptype, !board->turn);
            cb_delete_piece(board, from, ptype, board->turn);
            break;
        case CB_MV_DOUBLE_PAWN_PUSH:
            cb_hist_set_enp(&new_state, to & 0b111);
            cb_write_piece(board, to, CB_PTYPE_PAWN, board->turn);
            cb_delete_piece(board, from, CB_PTYPE_PAWN, board->turn);
            break;
        case CB_MV_KING_SIDE_CASTLE:
            rook_from = board->turn ? M_WHITE_KING_SIDE_ROOK_START :
                M_BLACK_KING_SIDE_ROOK_START;
            rook_to = board->turn ? M_WHITE_KING_SIDE_ROOK_TARGET :
                M_BLACK_KING_SIDE_ROOK_TARGET;
            cb_hist_set_captured_piece(&new_state, CB_PTYPE_EMPTY);
            cb_hist_remove_castle(&new_state, board->turn);
            cb_delete_piece(board, from, CB_PTYPE_KING, board->turn);
            cb_write_piece(board, to, CB_PTYPE_KING, board->turn);
            cb_delete_piece(board, rook_from, CB_PTYPE_ROOK, board->turn);
            cb_write_piece(board, rook_to, CB_PTYPE_ROOK, board->turn);
            break;
        case CB_MV_QUEEN_SIDE_CASTLE:
            rook_from = board->turn ? M_WHITE_QUEEN_SIDE_ROOK_START :
                M_BLACK_QUEEN_SIDE_ROOK_START;
            rook_to = board->turn ? M_WHITE_QUEEN_SIDE_ROOK_TARGET :
                M_BLACK_QUEEN_SIDE_ROOK_TARGET;
            cb_hist_set_captured_piece(&new_state, CB_PTYPE_EMPTY);
            cb_hist_remove_castle(&new_state, board->turn);
            cb_delete_piece(board, from, CB_PTYPE_KING, board->turn);
            cb_write_piece(board, to, CB_PTYPE_KING, board->turn);
            cb_delete_piece(board, rook_from, CB_PTYPE_ROOK, board->turn);
            cb_write_piece(board, rook_to, CB_PTYPE_ROOK, board->turn);
            break;
        case CB_MV_ENPASSANT:
            direction = board->turn == CB_WHITE ? 8 : -8;
            cb_hist_set_captured_piece(&new_state, CB_PTYPE_PAWN);
            cb_write_piece(board, to, CB_PTYPE_PAWN, board->turn);
            cb_delete_piece(board, from, CB_PTYPE_PAWN, board->turn);
            cb_delete_piece(board, to + direction, CB_PTYPE_PAWN, !board->turn);
            break;
        case CB_MV_KNIGHT_PROMO:
            cb_hist_set_captured_piece(&new_state, CB_PTYPE_EMPTY);
            cb_write_piece(board, to, CB_PTYPE_KNIGHT, board->turn);
            cb_delete_piece(board, from, CB_PTYPE_PAWN, board->turn);
            break;
        case CB_MV_BISHOP_PROMO:
            cb_hist_set_captured_piece(&new_state, CB_PTYPE_EMPTY);
            cb_write_piece(board, to, CB_PTYPE_BISHOP, board->turn);
            cb_delete_piece(board, from, CB_PTYPE_PAWN, board->turn);
            break;
        case CB_MV_ROOK_PROMO:
            cb_hist_set_captured_piece(&new_state, CB_PTYPE_EMPTY);
            cb_write_piece(board, to, CB_PTYPE_ROOK, board->turn);
            cb_delete_piece(board, from, CB_PTYPE_PAWN, board->turn);
            break;
        case CB_MV_QUEEN_PROMO:
            cb_hist_set_captured_piece(&new_state, CB_PTYPE_EMPTY);
            cb_write_piece(board, to, CB_PTYPE_QUEEN, board->turn);
            cb_delete_piece(board, from, CB_PTYPE_PAWN, board->turn);
            break;
        case CB_MV_KNIGHT_PROMO_CAPTURE:
            cap_ptype = cb_ptype_at_sq(board, to);
            cb_hist_set_captured_piece(&new_state, cap_ptype);
            cb_hist_decay_castle_rights(&new_state, board->turn, to, from);
            cb_replace_piece(board, to, CB_PTYPE_KNIGHT, board->turn, cap_ptype, !board->turn);
            cb_delete_piece(board, from, CB_PTYPE_PAWN, board->turn);
            break;
        case CB_MV_BISHOP_PROMO_CAPTURE:
            cap_ptype = cb_ptype_at_sq(board, to);
            cb_hist_set_captured_piece(&new_state, cap_ptype);
            cb_hist_decay_castle_rights(&new_state, board->turn, to, from);
            cb_replace_piece(board, to, CB_PTYPE_BISHOP, board->turn, cap_ptype, !board->turn);
            cb_delete_piece(board, from, CB_PTYPE_PAWN, board->turn);
            break;
        case CB_MV_ROOK_PROMO_CAPTURE:
            cap_ptype = cb_ptype_at_sq(board, to);
            cb_hist_set_captured_piece(&new_state, cap_ptype);
            cb_hist_decay_castle_rights(&new_state, board->turn, to, from);
            cb_replace_piece(board, to, CB_PTYPE_ROOK, board->turn, cap_ptype, !board->turn);
            cb_delete_piece(board, from, CB_PTYPE_PAWN, board->turn);
            break;
        case CB_MV_QUEEN_PROMO_CAPTURE:
            cap_ptype = cb_ptype_at_sq(board, to);
            cb_hist_set_captured_piece(&new_state, cap_ptype);
            cb_hist_decay_castle_rights(&new_state, board->turn, to, from);
            cb_replace_piece(board, to, CB_PTYPE_QUEEN, board->turn, cap_ptype, !board->turn);
            cb_delete_piece(board, from, CB_PTYPE_PAWN, board->turn);
            break;
    }

    /* Save the new state to the stack. */
    board->turn = !board->turn;
    new_ele.hist = new_state;
    new_ele.move = mv;
    cb_hist_stack_push(&board->hist, new_ele);
}

void cb_unmake(cb_board_t *board)
{
    cb_hist_ele_t old_ele = cb_hist_stack_pop(&board->hist);
    cb_hist_ele_t new_ele;
    cb_mv_flag_t flag = cb_mv_get_flags(old_ele.move);
    uint8_t to = cb_mv_get_to(old_ele.move);
    uint8_t from = cb_mv_get_from(old_ele.move);

    cb_ptype_t ptype;
    cb_ptype_t cap_ptype;

    /* Variables for castles. */
    uint8_t rook_from;
    uint8_t rook_to;

    /* Variables for enp. */
    int8_t direction;

    /* Unmake the move. */
    board->turn = !board->turn;
    switch (flag) {
        case CB_MV_QUIET:
        case CB_MV_DOUBLE_PAWN_PUSH:
            ptype = cb_ptype_at_sq(board, to);
            cb_write_piece(board, from, ptype, board->turn);
            cb_delete_piece(board, to, ptype, board->turn);
            break;
        case CB_MV_CAPTURE:
            ptype = cb_ptype_at_sq(board, to);
            cap_ptype = cb_hist_get_captured_piece(&old_ele.hist);
            cb_write_piece(board, from, ptype, board->turn);
            cb_replace_piece(board, to, cap_ptype, !board->turn, ptype, board->turn);
            break;
        case CB_MV_KING_SIDE_CASTLE:
            rook_from = board->turn ? M_WHITE_KING_SIDE_ROOK_START :
                M_BLACK_KING_SIDE_ROOK_START;
            rook_to = board->turn ? M_WHITE_KING_SIDE_ROOK_TARGET :
                M_BLACK_KING_SIDE_ROOK_TARGET;
            cb_write_piece(board, from, CB_PTYPE_KING, board->turn);
            cb_delete_piece(board, to, CB_PTYPE_KING, board->turn);
            cb_write_piece(board, rook_from, CB_PTYPE_ROOK, board->turn);
            cb_delete_piece(board, rook_to, CB_PTYPE_ROOK, board->turn);
            break;
        case CB_MV_QUEEN_SIDE_CASTLE:
            rook_from = board->turn ? M_WHITE_QUEEN_SIDE_ROOK_START :
                M_BLACK_QUEEN_SIDE_ROOK_START;
            rook_to = board->turn ? M_WHITE_QUEEN_SIDE_ROOK_TARGET :
                M_BLACK_QUEEN_SIDE_ROOK_TARGET;
            cb_write_piece(board, from, CB_PTYPE_KING, board->turn);
            cb_delete_piece(board, to, CB_PTYPE_KING, board->turn);
            cb_write_piece(board, rook_from, CB_PTYPE_ROOK, board->turn);
            cb_delete_piece(board, rook_to, CB_PTYPE_ROOK, board->turn);
            break;
        case CB_MV_ENPASSANT:
            direction = board->turn ? 8 : -8;
            cb_write_piece(board, from, CB_PTYPE_PAWN, board->turn);
            cb_delete_piece(board, to, CB_PTYPE_PAWN, board->turn);
            cb_write_piece(board, to + direction, CB_PTYPE_PAWN, !board->turn);
            break;
        case CB_MV_KNIGHT_PROMO:
            cb_write_piece(board, from, CB_PTYPE_PAWN, board->turn);
            cb_delete_piece(board, to, CB_PTYPE_KNIGHT, board->turn);
            break;
        case CB_MV_BISHOP_PROMO:
            cb_write_piece(board, from, CB_PTYPE_PAWN, board->turn);
            cb_delete_piece(board, to, CB_PTYPE_BISHOP, board->turn);
            break;
        case CB_MV_ROOK_PROMO:
            cb_write_piece(board, from, CB_PTYPE_PAWN, board->turn);
            cb_delete_piece(board, to, CB_PTYPE_ROOK, board->turn);
            break;
        case CB_MV_QUEEN_PROMO:
            cb_write_piece(board, from, CB_PTYPE_PAWN, board->turn);
            cb_delete_piece(board, to, CB_PTYPE_QUEEN, board->turn);
            break;
        case CB_MV_KNIGHT_PROMO_CAPTURE:
            cap_ptype = cb_hist_get_captured_piece(&old_ele.hist);
            cb_write_piece(board, from, CB_PTYPE_PAWN, board->turn);
            cb_replace_piece(board, to, cap_ptype, !board->turn, CB_PTYPE_KNIGHT, board->turn);
            break;
        case CB_MV_BISHOP_PROMO_CAPTURE:
            cap_ptype = cb_hist_get_captured_piece(&old_ele.hist);
            cb_write_piece(board, from, CB_PTYPE_PAWN, board->turn);
            cb_replace_piece(board, to, cap_ptype, !board->turn, CB_PTYPE_BISHOP, board->turn);
            break;
        case CB_MV_ROOK_PROMO_CAPTURE:
            cap_ptype = cb_hist_get_captured_piece(&old_ele.hist);
            cb_write_piece(board, from, CB_PTYPE_PAWN, board->turn);
            cb_replace_piece(board, to, cap_ptype, !board->turn, CB_PTYPE_ROOK, board->turn);
            break;
        case CB_MV_QUEEN_PROMO_CAPTURE:
            cap_ptype = cb_hist_get_captured_piece(&old_ele.hist);
            cb_write_piece(board, from, CB_PTYPE_PAWN, board->turn);
            cb_replace_piece(board, to, cap_ptype, !board->turn, CB_PTYPE_QUEEN, board->turn);
            break;
    }
}

