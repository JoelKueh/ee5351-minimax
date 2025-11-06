
#ifndef CB_CONST
#define CB_CONST

#include <stdint.h>

#include "cb_types.h"

#define ENEMY_COLOR(c) (!c)

extern const uint16_t HIST_INIT_BOARD_STATE;
extern const uint16_t HIST_ENP_COL;
extern const uint16_t HIST_PID_COL;
extern const uint16_t HIST_ENP_AVAILABLE;
extern const uint16_t HIST_ENP_ALL;
extern const uint16_t HIST_HALFMOVE_CLOCK;
extern const uint16_t HIST_HALFMOVE_FIFTY;

extern const uint8_t M_WHITE_KING_START;
extern const uint8_t M_WHITE_KING_SIDE_ROOK_START;
extern const uint8_t M_WHITE_QUEEN_SIDE_ROOK_START;
extern const uint8_t M_WHITE_KING_SIDE_CASTLE_TARGET;
extern const uint8_t M_WHITE_KING_SIDE_ROOK_TARGET;
extern const uint8_t M_WHITE_QUEEN_SIDE_CASTLE_TARGET;
extern const uint8_t M_WHITE_QUEEN_SIDE_ROOK_TARGET;

extern const uint8_t M_BLACK_KING_START;
extern const uint8_t M_BLACK_KING_SIDE_ROOK_START;
extern const uint8_t M_BLACK_QUEEN_SIDE_ROOK_START;
extern const uint8_t M_BLACK_KING_SIDE_CASTLE_TARGET;
extern const uint8_t M_BLACK_KING_SIDE_ROOK_TARGET;
extern const uint8_t M_BLACK_QUEEN_SIDE_CASTLE_TARGET;
extern const uint8_t M_BLACK_QUEEN_SIDE_ROOK_TARGET;

extern const uint8_t M_WHITE_MIN_ENPASSANT_TARGET;
extern const uint8_t M_BLACK_MIN_ENPASSANT_TARGET;

extern const uint64_t BB_RIGHT_COL;
extern const uint64_t BB_LEFT_COL;
extern const uint64_t BB_RIGHT_TWO_COLS;
extern const uint64_t BB_LEFT_TWO_COLS;
extern const uint64_t BB_TOP_ROW;
extern const uint64_t BB_BOTTOM_ROW;
extern const uint64_t BB_FULL;
extern const uint64_t BB_EMPTY;
extern const uint64_t BB_BLACK_PAWN_HOME;
extern const uint64_t BB_WHITE_PAWN_HOME;
extern const uint64_t BB_BLACK_PAWN_LINE;
extern const uint64_t BB_WHITE_PAWN_LINE;

extern const uint64_t BB_WHITE_KING_SIDE_CASTLE_TARGET;
extern const uint64_t BB_WHITE_QUEEN_SIDE_CASTLE_TARGET;

extern const uint64_t BB_WHITE_KING_SIDE_CASTLE_OCCUPANCY;
extern const uint64_t BB_WHITE_QUEEN_SIDE_CASTLE_OCCUPANCY;
extern const uint64_t BB_WHITE_KING_SIDE_CASTLE_CHECK;
extern const uint64_t BB_WHITE_QUEEN_SIDE_CASTLE_CHECK;
 
extern const uint64_t BB_BLACK_KING_SIDE_CASTLE_TARGET;
extern const uint64_t BB_BLACK_QUEEN_SIDE_CASTLE_TARGET;
 
extern const uint64_t BB_BLACK_KING_SIDE_CASTLE_OCCUPANCY;
extern const uint64_t BB_BLACK_QUEEN_SIDE_CASTLE_OCCUPANCY;
extern const uint64_t BB_BLACK_KING_SIDE_CASTLE_CHECK;
extern const uint64_t BB_BLACK_QUEEN_SIDE_CASTLE_CHECK;

extern const uint16_t CB_MV_TO_MASK;
extern const uint16_t CB_MV_FROM_MASK;
extern const uint16_t CB_MV_FLAG_MASK;

extern const cb_move_t CB_INVALID_MOVE;
extern const cb_hist_ele_t CB_INIT_STATE;

# endif /* CB_CONST */
