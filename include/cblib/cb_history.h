
#ifndef CB_HISTORY_H
#define CB_HISTORY_H

#include <stdint.h>
#include <stdbool.h>
#include <stdlib.h>
#include <errno.h>

#include "cb_const.h"

/* The only way a game is longer than 1024 moves is if someone is doing something malicious.
 * Make strict guarantees about safety while also making things fast for default use. */
#define CB_STACK_INIT_SIZE 1024

/**
 * Format:
 * HLFMV_NUMBER : ENP_COL / CAP_PIECE : ENP_AVAIL : KQkq
 */

/**
 * Sets the size of *hist to hist->count + CB_STACK_INIT_SIZE elements.
 */
static inline int cb_hist_stack_reserve(cb_hist_stack_t *hist, uint32_t added_depth)
{
    if (hist->count + added_depth < hist->size)
        return 0;

    hist->data = (cb_hist_ele_t *)realloc(hist->data,
            (hist->count + added_depth) * sizeof(cb_hist_ele_t));
    if (hist->data == NULL)
        return ENOMEM;
    hist->size += CB_STACK_INIT_SIZE;
    return 0;
}

/**
 * Initializes a history stack.
 */
static inline int cb_hist_stack_init(cb_hist_stack_t *hist)
{
    if ((hist->data = (cb_hist_ele_t *)malloc(CB_STACK_INIT_SIZE * sizeof(cb_hist_ele_t))) == NULL)
        return ENOMEM;
    hist->count = 0;
    hist->size = CB_STACK_INIT_SIZE;
    return 0;
}

/**
 * Frees a history stack.
 */
static inline void cb_hist_stack_free(cb_hist_stack_t *hist)
{
    free(hist->data);
}

/**
 * Pushes a history element to the stack.
 * IT IS UP TO THE CALLER TO GUARANTEE THAT PROPER SPACE IS ALLOCATED ON THE HEAP.
 */
static inline void cb_hist_stack_push(cb_hist_stack_t *hist, cb_hist_ele_t ele)
{
    hist->data[hist->count++] = ele;
}

/**
 * Returns the top elment of the history stack and removes it.
 */
static inline cb_hist_ele_t cb_hist_stack_pop(cb_hist_stack_t *hist)
{
    return hist->data[--hist->count];
}

/**
 * Returns true if the player has the right to king side castle, false otherwise.
 */
static inline bool cb_hist_has_ksc(cb_history_t hist, cb_color_t color)
{
    return (hist & 0b10 << color * 2) != 0;
}

/**
 * Returns true if the player still has the right to queen side castle, false otherwise.
 */
static inline bool cb_hist_has_qsc(cb_history_t hist, cb_color_t color)
{
    return (hist & 0b1 << color * 2) != 0;
}


/**
 * Removes the right to king side castle.
 */
static inline void cb_hist_remove_ksc(cb_history_t *hist, cb_color_t color)
{
    *hist &= ~(0b10 << color * 2);
}

/**
 * Removes the right to queen side castle.
 */
static inline void cb_hist_remove_qsc(cb_history_t *hist, cb_color_t color)
{
    *hist &= ~(0b1 << color * 2);
}

/**
 * Removes all castling rights for a specified color.
 */
static inline void cb_hist_remove_castle(cb_history_t *hist, cb_color_t color)
{
    *hist &= ~(0b11 << color * 2);
}


/**
 * Adds king side castling right.
 */
static inline void cb_hist_add_ksc(cb_history_t *hist, cb_color_t color)
{
    *hist |= 0b10 << color * 2;
}

/**
 * Adds queen side castling right.
 */
static inline void cb_hist_add_qsc(cb_history_t *hist, cb_color_t color)
{
    *hist |= 0b1 << color * 2;
}

/**
 * Removes all castling rights for specified color.
 */
static inline void cb_hist_add_castle(cb_history_t *hist, cb_color_t color)
{
    *hist |= 0b11 << color * 2;
}


/**
 * Returns true if there is an enpassant availiable.
 */
static inline bool cb_hist_enp_availiable(cb_history_t hist)
{
    return (hist & HIST_ENP_AVAILABLE) != 0;
}

static inline uint8_t cb_hist_enp_col(cb_history_t hist)
{
    return (hist & HIST_ENP_COL) >> 5;
}

/**
 * Sets up this move state to open an enpassant square.
 */
static inline void cb_hist_set_enp(cb_history_t *hist, uint8_t enp_col)
{
    *hist = (*hist & ~HIST_ENP_COL) | (enp_col << 5);
    *hist |= HIST_ENP_AVAILABLE;
}

/**
 * Removes enpassant from the history state.
 */
static inline void cb_hist_decay_enp(cb_history_t *hist)
{
    *hist &= ~HIST_ENP_ALL;
}

/**
 * Sets up this move state to hold a captured piece.
 */
static inline void cb_hist_set_captured_piece(cb_history_t *hist, cb_ptype_t pid)
{
    *hist = (*hist & ~HIST_ENP_COL) | (pid << 5);
    *hist &= ~HIST_ENP_AVAILABLE;
}

static inline cb_ptype_t cb_hist_get_captured_piece(cb_history_t *hist)
{
    return (*hist & HIST_ENP_COL) >> 5;
}


/**
 * Returns true if the 50-move rule has been met, else returns false.
 */
static inline bool cb_hist_halfmove_clk_done(cb_history_t hist)
{
    return (hist & HIST_HALFMOVE_CLOCK) == HIST_HALFMOVE_FIFTY;
}

/**
 * Resets the halfmove clock.
 */
static inline void cb_hist_reset_halfmove_clk(cb_history_t *hist)
{
    *hist &= ~HIST_HALFMOVE_CLOCK;
}

/**
 * Increments the halfmove clock.
 */
static inline void cb_hist_inc_halfmove_clk(cb_history_t *hist)
{
    *hist += UINT16_C(1) << 8;
}

static inline void cb_hist_set_halfmove_clk(cb_history_t *hist, uint16_t val)
{
    *hist &= ~HIST_HALFMOVE_CLOCK;
    *hist |= val << 8;
}

/**
 * Decays castle rights after a move.
 */
static inline void cb_hist_decay_castle_rights(cb_history_t *hist, uint8_t color,
        uint8_t to, uint8_t from)
{
    /* Remove castling rights for moving a king or rook. */
    *hist &= from == M_WHITE_KING_START ? ~UINT16_C(0b1100) : 0xFFFF;
    *hist &= from == M_BLACK_KING_START ? ~UINT16_C(  0b11) : 0xFFFF;
    *hist &= from == M_WHITE_KING_SIDE_ROOK_START ? ~UINT16_C(0b1000) : 0xFFFF;
    *hist &= from == M_BLACK_KING_SIDE_ROOK_START ? ~UINT16_C(  0b10) : 0xFFFF;
    *hist &= from == M_WHITE_QUEEN_SIDE_ROOK_START ? ~UINT16_C(0b100) : 0xFFFF;
    *hist &= from == M_BLACK_QUEEN_SIDE_ROOK_START ? ~UINT16_C(  0b1) : 0xFFFF;

    /* Remove castling rights for taking a rook. */
    *hist &= to == M_WHITE_KING_SIDE_ROOK_START ? ~UINT16_C(0b1000) : 0xFFFF;
    *hist &= to == M_BLACK_KING_SIDE_ROOK_START ? ~UINT16_C(  0b10) : 0xFFFF;
    *hist &= to == M_WHITE_QUEEN_SIDE_ROOK_START ? ~UINT16_C(0b100) : 0xFFFF;
    *hist &= to == M_BLACK_QUEEN_SIDE_ROOK_START ? ~UINT16_C(  0b1) : 0xFFFF;
}

#endif /* CB_HISTORY_H */
