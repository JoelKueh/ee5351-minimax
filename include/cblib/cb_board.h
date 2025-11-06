
#ifndef CB_BOARD_H
#define CB_BOARD_H

#include <stdint.h>
#include <stdbool.h>
#include <assert.h>
#include <string.h>

#include "cb_types.h"

static inline char ptype_to_ascii(cb_ptype_t piece, cb_color_t color)
{
    switch (piece) {
        case CB_PTYPE_PAWN:
            return color == CB_WHITE ? 'P' : 'p';
        case CB_PTYPE_KNIGHT:
            return color == CB_WHITE ? 'N' : 'n';
        case CB_PTYPE_BISHOP:
            return color == CB_WHITE ? 'B' : 'b';
        case CB_PTYPE_ROOK:
            return color == CB_WHITE ? 'R' : 'r';
        case CB_PTYPE_QUEEN:
            return color == CB_WHITE ? 'Q' : 'q';
        case CB_PTYPE_KING:
            return color == CB_WHITE ? 'K' : 'k';
        case CB_PTYPE_EMPTY:
            return ' ';
    }
}

/* Functions for reading the board representation. */
static inline cb_ptype_t cb_ptype_at_sq(const cb_board_t *board, uint8_t sq)
{
    return board->mb.data[sq];
}

static inline cb_ptype_t cb_ptype_at(const cb_board_t *board, uint8_t row, uint8_t col)
{
    return cb_ptype_at_sq(board, row * 8 + col);
}

static inline cb_pid_t cb_pid_at_sq(const cb_board_t *board, uint8_t sq)
{
    assert(false && "not yet implemented");
    return board->mb.data[sq];
}

static inline cb_pid_t cb_pid_at(const cb_board_t *board, uint8_t row, uint8_t col)
{
    assert(false && "not yet implemented");
    return cb_pid_at_sq(board, row * 8 + col);
}

static inline cb_color_t cb_color_at_sq(const cb_board_t *board, uint8_t sq)
{
    return board->bb.color[CB_WHITE] & (UINT64_C(1) << sq) ? CB_WHITE : CB_BLACK;
}

static inline cb_color_t cb_color_at(const cb_board_t *board, uint8_t row, uint8_t col)
{
    return cb_color_at_sq(board, row * 8 + col);
}

/* Functions for manipulating the board representation. */
static inline void cb_replace_piece(cb_board_t *board, uint8_t sq, uint8_t ptype, uint8_t pcolor,
        uint8_t old_ptype, uint8_t old_pcolor)
{
    board->mb.data[sq] = ptype;
    board->bb.piece[pcolor][ptype] |= UINT64_C(1) << sq;
    board->bb.color[pcolor] |= UINT64_C(1) << sq;
    board->bb.piece[old_pcolor][old_ptype] &= ~(UINT64_C(1) << sq);
    board->bb.color[old_pcolor] &= ~(UINT64_C(1) << sq);
}

static inline void cb_write_piece(cb_board_t *board, uint8_t sq, uint8_t ptype, uint8_t pcolor)
{
    board->mb.data[sq] = ptype;
    board->bb.piece[pcolor][ptype] |= UINT64_C(1) << sq;
    board->bb.color[pcolor] |= UINT64_C(1) << sq;
    board->bb.occ |= UINT64_C(1) << sq;
}

static inline void cb_delete_piece(cb_board_t *board, uint8_t sq, uint8_t ptype, uint8_t pcolor)
{
    board->mb.data[sq] = CB_PTYPE_EMPTY;
    board->bb.piece[pcolor][ptype] &= ~(UINT64_C(1) << sq);
    board->bb.color[pcolor] &= ~(UINT64_C(1) << sq);
    board->bb.occ &= ~(UINT64_C(1) << sq);
}

static inline void cb_wipe_board(cb_board_t *board)
{
    int i;
    memset(&board->mb.data, CB_PTYPE_EMPTY, 64 * sizeof(uint8_t));
    memset(&board->bb.piece, 0, 12 * sizeof(uint64_t));
    memset(&board->bb.color, 0, 2 * sizeof(uint64_t));
    board->bb.occ = 0;
    board->hist.count = 0;
}

/* Functions for converting a board to a 2d array of characters. */
static inline void cb_board_to_str(char str_rep[8][8], const cb_board_t *board)
{
    int row, col;
    for (row = 0; row < 8; row++) {
        for (col = 0; col < 8; col++) {
            str_rep[row][col] = ptype_to_ascii(cb_ptype_at(board, row, col),
                    cb_color_at(board, row, col));
        }
    }
}

static inline void cb_board_to_utf8(char str_rep[8][8], const cb_board_t *board)
{

}

#endif /* CB_BOARD_H */
