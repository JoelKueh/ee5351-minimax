
#ifndef GPU_TYPES_H
#define GPU_TYPES_H

#include <stdint.h>
#include <stdbool.h>

#define GPU_MAX_NUM_MOVES 218
#define GPU_MAX_SEARCH_DEPTH 8

/* Macros for reading pieces in the board. */
#define GPU_BB_COLOR(b, c)   (c ? b.color : b.occ & ~b.color)
#define GPU_BB_PAWNS(b, c)   (b.pawns & (c ? b.color : ~b.color))
#define GPU_BB_KNIGHTS(b, c) (b.knights & (c ? b.color : ~b.color))
#define GPU_BB_B_AND_Q(b, c) (b.bishops & (c ? b.color : ~b.color)) /* Bishops and Queens. */
#define GPU_BB_R_AND_Q(b, c) (b.rooks & (c ? b.color : ~b.color)) /* Rooks and Queens. */
#define GPU_BB_BISHOPS(b, c) (b.bishops & ~b.rooks & (c ? b.color : ~b.color))
#define GPU_BB_ROOKS(b, c)   (b.rooks & ~b.bishops & (c ? b.color : ~b.color))
#define GPU_BB_QUEENS(b, c)  (b.bishops & b.rooks & (c ? b.color : ~b.color))
#define GPU_BB_KINGS(b, c)   (b.kings & (c ? b.color : ~b.color))

/* Board Representation. */
typedef struct {
    uint64_t color;         /**< Bitmask of all white pieces. */
    uint64_t pawns;         /**< Bitmask of all pawns. */
    uint64_t knights;       /**< Bitmask of all knights. */
    uint64_t bishops;       /**< Bitmask of all bishops. */
    uint64_t rooks;         /**< Bitmask of all rooks. */
    uint64_t kings;         /**< Bitmask of all kings. */
    uint64_t occ;           /**< Occupancy bitmask. */
} gpu_bitboard_t;

/**
 * @breif Enumerates board piece and turn colors.
 */
enum {
    GPU_WHITE = 1,
    GPU_BLACK = 0
};
typedef bool gpu_color_t;

/**
 * @breif Enumerates piece types.
 *
 * A piece type contains only information about type, not about color.
 * This type can be used to index the bitboard array and is stored in the mailbox.
 */
enum {
    GPU_PTYPE_PAWN   = 0,
    GPU_PTYPE_KNIGHT = 1,
    GPU_PTYPE_BISHOP = 2,
    GPU_PTYPE_ROOK   = 3,
    GPU_PTYPE_QUEEN  = 4,
    GPU_PTYPE_KING   = 5,
    GPU_PTYPE_EMPTY  = 6
};
typedef uint8_t gpu_ptype_t;

/**
 * Enum holding the different flags that a move can contain
 */
enum {
    GPU_MV_QUIET                =  0 << 12,
    GPU_MV_DOUBLE_PAWN_PUSH     =  1 << 12,
    GPU_MV_KING_SIDE_CASTLE     =  2 << 12,
    GPU_MV_QUEEN_SIDE_CASTLE    =  3 << 12,
    GPU_MV_CAPTURE              =  4 << 12,
    GPU_MV_ENPASSANT            =  5 << 12,
    GPU_MV_KNIGHT_PROMO         =  8 << 12,
    GPU_MV_BISHOP_PROMO         =  9 << 12,
    GPU_MV_ROOK_PROMO           = 10 << 12,
    GPU_MV_QUEEN_PROMO          = 11 << 12,
    GPU_MV_KNIGHT_PROMO_CAPTURE = 12 << 12,
    GPU_MV_BISHOP_PROMO_CAPTURE = 13 << 12,
    GPU_MV_ROOK_PROMO_CAPTURE   = 14 << 12,
    GPU_MV_QUEEN_PROMO_CAPTURE  = 15 << 12
};
typedef uint16_t gpu_mv_flag_t;

/* Move representation. */
typedef uint16_t gpu_move_t;

/**
 * @breif Simple type to hold board state info not captured in the piece
 * organization. I'm sorry, this type should really be called board state
 * or something other than history. Bad naming scheme that I haven't fixed.
 */
typedef uint16_t gpu_history_t;

/* Full board representation. Flags packed into pawn bitboard as follows.
 * Credit to Ankan Banerjee for recommending this packing in his GPU chess
 * engine https://github.com/ankan-ban/perft_gpu.
 *
 *    q k Q K e e e e  <-- Unused bits. Pawns can never be here.
 *    0 0 0 0 0 0 0 0
 *    0 0 0 0 0 0 0 0
 *    0 0 0 0 0 0 0 0
 *    0 0 0 0 0 0 0 0
 *    0 0 0 0 0 0 0 0
 *    0 0 0 0 0 0 0 0
 *    h h h h h h h h  <-- Unused bits. Pawns can never be here.
 *
 * h -> Halfmove number. I don't actually use this. Ignoring 50 move rule.
 * e -> Enpassant column.
 * a -> Enpassant avaliable.
 * K -> White king side castle valid.
 * Q -> White queen side castle valid.
 * k -> White king side castle valid.
 * q -> White queen side castle valid.
 *
 * Definitions for working with this board with state bits is in gpu_state.h
 */
typedef struct {
    gpu_bitboard_t bb;  /**< Bitboard field in the union. */
    gpu_history_t state;
    gpu_color_t turn;
} gpu_board_t;

/**
 * @breif State table data structure that is useful in move generation.
 */
typedef struct {
    uint64_t threats;       /**< A bitmask for all pieces that threaten the king. */
    uint64_t checks;        /**< A bitmask for all pieces that check the king. */
    uint64_t check_blocks;  /**< A bitmask for all squares that can break a check. */
    uint64_t pinned;        /**< A bitmask for all pinned pieces. */
} gpu_state_tables_t;

/**
 * @brief Simple typedef for a move.
 */
typedef uint16_t gpu_move_t;

/**
 * @breif Stack element that holds the history of the board.
 */
typedef struct {
    gpu_history_t state;    /**< The history state at a given position. */
    gpu_move_t move;        /**< The last move played at a given position. */
} gpu_hist_ele_t;


/**
 * @breif Fixed size datastructure containing board history.
 * This holds the data for the search struct that is stored in global memory.
 *
 * This datastructure is shared among warps to get better coallescence
 * (history elements are interleaved among the warps).
 */
typedef struct {
    gpu_hist_ele_t history[GPU_MAX_SEARCH_DEPTH][32];  /* Move and state history. */
} gpu_history_list_t;

/**
 * @brief Holds information for one node in the search structure.
 * Shared among warps.
 */
typedef struct {
    gpu_move_t moves[GPU_MAX_NUM_MOVES][32];
    gpu_hist_ele_t hist_ele[32];
} gpu_search_struct_node_t;

/**
 * @brief Information for where moves are stored during the "search".
 */
typedef struct {
    gpu_move_t *moves;
    uint8_t count;
} gpu_search_struct_t;

#endif /* GPU_TYPES_H */

