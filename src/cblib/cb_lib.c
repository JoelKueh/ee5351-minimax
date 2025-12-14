
#include <string.h>
#include <stdio.h>
#include <errno.h>

#include "cb_lib.h"
#include "cb_tables.h"
#include "cb_const.h"
#include "cb_move.h"
#include "cb_board.h"
#include "cb_history.h"

void cb_mv_to_uci_algbr(char *buf, cb_move_t move)
{
    buf[0] = cb_mv_get_from(move) % 8 + 'a';
    buf[1] = '8' - cb_mv_get_from(move) / 8;
    buf[2] = cb_mv_get_to(move) % 8 + 'a';
    buf[3] = '8' - cb_mv_get_to(move) / 8;

    switch (cb_mv_get_flags(move)) {
        case CB_MV_KNIGHT_PROMO:
        case CB_MV_KNIGHT_PROMO_CAPTURE:
            buf[4] = 'n';
            buf[5] = '\0';
            break;
        case CB_MV_BISHOP_PROMO:
        case CB_MV_BISHOP_PROMO_CAPTURE:
            buf[4] = 'b';
            buf[5] = '\0';
            break;
        case CB_MV_ROOK_PROMO:
        case CB_MV_ROOK_PROMO_CAPTURE:
            buf[4] = 'r';
            buf[5] = '\0';
            break;
        case CB_MV_QUEEN_PROMO:
        case CB_MV_QUEEN_PROMO_CAPTURE:
            buf[4] = 'q';
            buf[5] = '\0';
            break;
        default:
            buf[4] = '\0';
            break;
    }
}

cb_errno_t cb_board_init(cb_error_t *err, cb_board_t *board)
{
    cb_errno_t result;
    if ((result = cb_hist_stack_init(&board->hist)) != 0)
        return cb_mkerr(err, result, "malloc: %s\n", strerror(errno));

    return 0;
}

cb_errno_t cb_tables_init(cb_error_t *err)
{
    cb_errno_t result;
    cb_init_normal_tables();
    if ((result = cb_init_magic_tables()) != 0)
        return cb_mkerr(err, result, "malloc: %s\n", strerror(errno));
    return CB_EOK;
}

void cb_tables_free()
{
    cb_free_magic_tables();
}

void cb_board_free(cb_board_t *board)
{
    cb_hist_stack_free(&board->hist);
}

cb_errno_t cb_reserve_for_make(cb_error_t *err, cb_board_t *board, uint32_t added_depth)
{
    int result;
    if ((result = cb_hist_stack_reserve(&board->hist, added_depth)) != 0) {
        return cb_mkerr(err, result, "realloc: %s\n", strerror(errno));
    }
    return 0;
}

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

cb_errno_t cb_mv_from_short_algbr(cb_error_t *err, cb_move_t *mv, cb_board_t *board,
                                  const char *algbr)
{
    assert(false && "not yet implemented");
    return 0;
}

cb_errno_t cb_mv_from_uci_algbr(cb_error_t *err, cb_move_t *mv, cb_board_t *board,
                                const char *algbr)
{
    uint8_t to, from;
    uint16_t flag;
    cb_move_t temp_mv;
    cb_mvlst_t mvlst;
    cb_state_tables_t state;
    int i;

    /* Make sure that the length of the algebraic string is 4 or 5 characters. */
    if (strlen(algbr) != 4 && strlen(algbr) != 5) {
        return cb_mkerr(err, CB_EINVAL, "invalid move string length");
    }

    /* Get the information about the move itself. */
    from = algbr[0] - 'a';
    from += ('8' - algbr[1]) * 8;
    to = algbr[2] - 'a';
    to += ('8' - algbr[3]) * 8;

    /* Verify that the characters inputted were valid. */
    if (from < 0 || from >= 64 || to < 0 || to >= 64)
        return cb_mkerr(err, CB_EINVAL, "invalid character in move");

    /* Find the move in the list of generated moves and return the match. */
    cb_gen_board_tables(&state, board);
    cb_gen_moves(&mvlst, board, &state);
    *mv = CB_INVALID_MOVE;
    for (i = 0; i < cb_mvlst_size(&mvlst); i++) {
        temp_mv = cb_mvlst_at(&mvlst, i);
        if (cb_mv_get_from(temp_mv) == from && cb_mv_get_to(temp_mv) == to) {
            *mv = temp_mv;
            break;
        }
    }

    /* Check if the move was not valid. */
    if (*mv == CB_INVALID_MOVE) {
        return cb_mkerr(err, CB_EILLEGAL, "illegal move specified");
    }

    /* Adjust the move for information about promotions. */
    flag = cb_mv_get_flags(*mv);
    if (flag & (1 << 15)) { /* Check if this is a promotion. */
        flag &= ~(0b11 << 12); /* Remove the lower two bits from the flag. */

        /* Check what the piece type is for a promotion and add the offset to turn
         * the piece type for the capture into what we want. */
        switch (algbr[4]) {
            case 'n':
                flag += 0 << 12;
                break;
            case 'b':
                flag += 1 << 12;
                break;
            case 'r':
                flag += 2 << 12;
                break;
            case 'q':
                flag += 3 << 12;
                break;
            default:
                return cb_mkerr(err, CB_EINVAL, "invlaid character in move");
        }

        /* Recreate the move. */
        *mv = cb_mv_from_data(from, to, flag);
    }

    return 0;
}

cb_errno_t parse_fen_main(cb_error_t *err, cb_board_t *board, char *fen_main)
{
    uint8_t sq = 0;
    uint8_t row = 0;
    char c;
    int i = 0;
    cb_color_t pcolor;

    /* Error handling. */
    if (fen_main == NULL)
        return cb_mkerr(err, CB_EINVAL, "missing fen body");

    /* Parse the main portion of the fen string. */
    while ((c = fen_main[i++]) != '\0') {
        if ('1' <= c && c <= '8') {
            sq += c - '0';
            /* Check if we should have had a '/' in our input string by now. */
            if (sq < row * 8 || sq > row * 8 + 8)
                return cb_mkerr(err, CB_EINVAL, "row overrun without encountering '/'");
        } else if (c == 'p' || c == 'P') {
            cb_write_piece(board, sq, CB_PTYPE_PAWN, c < 'a' ? CB_WHITE : CB_BLACK);
            sq++;
        } else if (c == 'n' || c == 'N') {
            cb_write_piece(board, sq, CB_PTYPE_KNIGHT, c < 'a' ? CB_WHITE : CB_BLACK);
            sq++;
        } else if (c == 'b' || c == 'B') {
            cb_write_piece(board, sq, CB_PTYPE_BISHOP, c < 'a' ? CB_WHITE : CB_BLACK);
            sq++;
        } else if (c == 'r' || c == 'R') {
            cb_write_piece(board, sq, CB_PTYPE_ROOK, c < 'a' ? CB_WHITE : CB_BLACK);
            sq++;
        } else if (c == 'q' || c == 'Q') {
            cb_write_piece(board, sq, CB_PTYPE_QUEEN, c < 'a' ? CB_WHITE : CB_BLACK);
            sq++;
        } else if (c == 'k' || c == 'K') {
            cb_write_piece(board, sq, CB_PTYPE_KING, c < 'a' ? CB_WHITE : CB_BLACK);
            sq++;
        } else if (c == '/') {
            /* If we are not at the end of the row, we don't want to get a '/'. */
            if (sq % 8 != 0)
                return cb_mkerr(err, CB_EINVAL, "encountered '/' before the end of a row");
            row += 1;
        } else {
            /* Any invalid characters return an error. */
            return cb_mkerr(err, CB_EINVAL, "invalid character in fen body");
        }
    }

    /* Throw errors for the write head not being at the end of the board. */
    if (sq != 64)
        return cb_mkerr(err, CB_EINVAL, "unexpected end to fen body");

    return 0;
}

cb_errno_t parse_fen_turn(cb_error_t *err, cb_board_t *board, char *fen_turn)
{
    /* Error handling. */
    if (fen_turn == NULL)
        return cb_mkerr(err, CB_EINVAL, "missing fen turn");

    /* Set the turn in the board. */
    if (*fen_turn == 'w') {
        board->turn = CB_WHITE;
    } else if (*fen_turn == 'b') {
        board->turn = CB_BLACK;
    } else {
        return cb_mkerr(err, CB_EINVAL, "unexpected character in fen turn");
    }
    
    return 0;
}

cb_errno_t parse_fen_rights(cb_error_t *err, cb_board_t *board, char *fen_rights)
{
    int i = 0;
    char c;

    /* Error handling. */
    if (fen_rights == NULL)
        return cb_mkerr(err, CB_EINVAL, "missing fen rights");

    /* Set the rights based on the token. */
    while ((c = fen_rights[i++]) != '\0') {
        if (c == 'K') {
            cb_hist_add_ksc(&board->hist.data[board->hist.count - 1].hist, CB_WHITE);
        } else if (c == 'Q') {
            cb_hist_add_qsc(&board->hist.data[board->hist.count - 1].hist, CB_WHITE);
        } else if (c == 'k') {
            cb_hist_add_ksc(&board->hist.data[board->hist.count - 1].hist, CB_BLACK);
        } else if (c == 'q') {
            cb_hist_add_qsc(&board->hist.data[board->hist.count - 1].hist, CB_BLACK);
        } else if (c == '-') {
            return 0;
        } else {
            return cb_mkerr(err, CB_EINVAL, "unexpected character in fen rights");
        }
    }

    return 0;
}

cb_errno_t parse_fen_enp(cb_error_t *err, cb_board_t *board, char *fen_enp)
{
    /* Error handling. */
    if (fen_enp == NULL)
        return cb_mkerr(err, CB_EINVAL, "missing fen rights");
    if (strlen(fen_enp) == 1) {
        if (fen_enp[0] != '-')
            return cb_mkerr(err, CB_EINVAL, "invalid character in fen enp");
        return 0;
    }
    if (strlen(fen_enp) != 2)
        return cb_mkerr(err, CB_EINVAL, "invalid fen enpassant length");
    if (fen_enp[0] < 'a' || fen_enp[0] > 'h')
        return cb_mkerr(err, CB_EINVAL, "invalid character in fen enp");

    /* Set the square based on the token. */
    cb_hist_set_enp(&board->hist.data[board->hist.count - 1].hist, fen_enp[0] - 'a');

    return 0;
}

cb_errno_t parse_fen_hlfmv(cb_error_t *err, cb_board_t *board, char *fen_hlfmv)
{
    int hlfmv;
    char *endptr;

    if (fen_hlfmv != NULL) {
        errno = 0;
        hlfmv = strtol(fen_hlfmv, &endptr, 10);
        if (errno || *endptr != '\0')
            return cb_mkerr(err, CB_EINVAL, "invalid halfmove number");
        cb_hist_set_halfmove_clk(&board->hist.data[board->hist.count - 1].move, hlfmv);
    }

    return 0;
}

cb_errno_t cb_board_from_fen(cb_error_t *err, cb_board_t *board, char *fen)
{
    cb_errno_t result;

    char *fen_main = strtok(fen, " \n");
    char *fen_turn = strtok(NULL, " \n");
    char *fen_rights = strtok(NULL, " \n");
    char *fen_enp = strtok(NULL, " \n");
    char *fen_hlfmv = strtok(NULL, " \n");

    cb_wipe_board(board);
    if ((result = cb_reserve_for_make(err, board, 1)) != 0)
        return cb_mkerr(err, CB_EABORT, "realloc: %s", strerror(errno));
    cb_hist_stack_push(&board->hist, CB_INIT_STATE);
    if ((result = parse_fen_main(err, board, fen_main)) != 0)
        return result;
    if ((result = parse_fen_turn(err, board, fen_turn)) != 0)
        return result;
    if ((result = parse_fen_rights(err, board, fen_rights)) != 0)
        return result;
    if ((result = parse_fen_enp(err, board, fen_enp)) != 0)
        return result;
    if ((result = parse_fen_hlfmv(err, board, fen_hlfmv)) != 0)
        return result;

    return 0;
}

cb_errno_t cb_board_from_uci(cb_error_t *err, cb_board_t *board, char *uci)
{
    cb_errno_t result;
    char *moves;
    char *algbr = NULL;
    cb_move_t mv;

    /* Locate the beginning of the string of moves. */
    if ((moves = strstr(uci, "moves ")) != NULL) {
        *(moves - 1) = '\0';
        moves += 6;
    }

    /* Parse the fen main section. */
    if ((result = cb_board_from_fen(err, board, uci)) != 0)
        return result;

    /* If there is no moves string, then go on with your life. */
    if (moves == NULL)
        return 0;

    /* Parse the list of moves. */
    algbr = strtok(moves, " \n");
    while (algbr != NULL) {
        if ((result = cb_mv_from_uci_algbr(err, &mv, board, algbr)) != 0)
            return result;
        cb_make(board, mv);
        algbr = strtok(NULL, " \n");
    }

    return 0;
}

cb_errno_t cb_board_from_pgn(cb_error_t *err, cb_board_t *board, char *fen)
{
    assert(false && "not yet implemented");
    return 0;
}

