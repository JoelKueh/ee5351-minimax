
#ifndef GPU_TYPES_H
#define GPU_TYPES_H

#include <stdint.h>
#include <stdbool.h>

#define GPU_MAX_NUM_MOVES 218
#define GPU_MAX_SEARCH_DEPTH 10

/* Board Representation. */
typedef struct {
    uint64_t color;         /**< Bitmasks for colored pieces. */
    uint64_t piece[5];      /**< A set of bitmasks for piece types. */
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
 * engine https://github.com/ankan-ban/perft_gpu. NOTE: Actually I'm changing
 * this. I think that the standard implementation will be easier to implement
 * first. We don't need this packing at the moment.
 *
 *    h h h h h h h h  <-- Unused bits. Pawns can never be here.
 *    0 0 0 0 0 0 0 0
 *    0 0 0 0 0 0 0 0
 *    0 0 0 0 0 0 0 0
 *    0 0 0 0 0 0 0 0
 *    0 0 0 0 0 0 0 0
 *    0 0 0 0 0 0 0 0
 *    e e e a K Q k q  <-- Unused bits. Pawns can never be here.
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
 * @breif Holds a list of moves.
 *
 * This data structure is not dynamically sized. A size of CB_MAX_NUM_MOVES was chosen as it
 * is the supposed maximum number of moves that can be played at any given chess position.
 */
typedef struct {
    gpu_move_t moves[GPU_MAX_NUM_MOVES];    /**< The list of moves. */
    uint8_t head;                           /**< The index of the top of the stack. */
} gpu_mvlst_t;

/**
 * @breif Stack element that holds the history of the board.
 */
typedef struct {
    gpu_history_t state;    /**< The history state at a given position. */
    gpu_move_t move;        /**< The last move played at a given position. */
} gpu_hist_ele_t;

/**
 * @breif Fixed size datastructure for the perft search on the GPU.
 * Each thread needs on of these search structures to hold the data
 * for the perft search.
 */
typedef struct {
    gpu_hist_ele_t hist[GPU_MAX_SEARCH_DEPTH]; /**< History. */
    gpu_mvlst_t moves[GPU_MAX_SEARCH_DEPTH]; /**< Move list. */
    int depth;              /**< The current depth on the stack. */
} gpu_search_struct_t;

#endif /* GPU_TYPES_H */
