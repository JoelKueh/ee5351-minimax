
#ifndef CB_DEBUG_H
#define CB_DEBUG_H

#include <stdio.h>
#include "cb_types.h"

/**
 * @breif Prints the move history stack.
 * @param f The file pointer to print to.
 * @param board The board containing the movelist to print.
 */
void cb_print_mv_hist(FILE *f, cb_board_t *board);

/**
 * @breif Prints the board in ascii.
 * @param f The file pointer to print to.
 * @param board The board to print.
 */
void cb_print_board_ascii(FILE *f, cb_board_t *board);

/**
 * @breif Prints the board with fancy chess characters from the UTF8 set.
 * @param f The file pointer to print to.
 * @param board The board to print.
 */
void cb_print_board_utf8(FILE *f, cb_board_t *board);

/**
 * @breif Prints the board with fancy chess characters from the UTF8 set.
 * @param f The file pointer to print to.
 * @param board The board to print.
 */
void cb_print_bitboard(FILE *f, cb_board_t *board);

/**
 * @breif Prints specified state tables.
 * @param f The file pointer to print to.
 * @param board The board to print.
 */
void cb_print_state(FILE *f, cb_state_tables_t *state);

/**
 * @breif Prints the moves that can be made in the current position.
 * @param f The file pointer to print to.
 * @param board The board to print.
 */
void cb_print_moves(FILE *f, cb_mvlst_t *mvlst);

#endif /* CB_DEBUG_H */
