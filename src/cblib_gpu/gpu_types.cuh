
#ifndef GPU_TYPES_H
#define GPU_TYPES_H

#include <stdint.h>
#include <stdbool.h>

#define GPU_MAX_NUM_MOVES 218
#define GPU_MAX_SEARCH_DEPTH 10

/* Macros for reading pieces in the board. */
#define GPU_BB_COLOR(b, c)   (c ? b.color : b.occ & ~b.color);
#define GPU_BB_PAWNS(b, c)   (b.piece[0] & (c ? b.color : ~b.color))
#define GPU_BB_KNIGHTS(b, c) (b.piece[1] & (c ? b.color : ~b.color))
#define GPU_BB_B_AND_Q(b, c) (b.piece[2] & (c ? b.color : ~b.color)) /* Bishops and Queens. */
#define GPU_BB_R_AND_Q(b, c) (b.piece[3] & (c ? b.color : ~b.color)) /* Rooks and Queens. */
#define GPU_BB_BISHOPS(b, c) (b.piece[2] & ~b.piece[3] & (c ? b.color : ~b.color))
#define GPU_BB_ROOKS(b, c)   (b.piece[3] & ~b.piece[2] & (c ? b.color : ~b.color))
#define GPU_BB_QUEENS(b, c)  (b.piece[2] & b.piece[3] & (c ? b.color : ~b.color))
#define GPU_BB_KINGS(b, c)   (b.piece[4] & (c ? b.color : ~b.color))

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

///**
// * @breif Holds a list of moves.
// *
// * This data structure is not dynamically sized. A size of CB_MAX_NUM_MOVES was chosen as it
// * is the supposed maximum number of moves that can be played at any given chess position.
// */
//typedef struct {
//    gpu_bitboard_t board;                   /**< Board for this position. */
//    gpu_move_t moves[GPU_MAX_NUM_MOVES];    /**< List of moves. */
//    uint8_t num_moves;      /**< Number of legal moves from this position. */
//} gpu_mvlst_t;

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
 * I think you could do something really cool with coallesced writes in some
 * sort of shared move buffer, but I don't know how that would work yet.
 *
 * My idea is something along these lines. There are 65536 bytes availiable to
 * us in shared memory, that means we can have 32 bytes for each thread.
 * This leaves us room for 16 moves per thread in a buffer (we'll decrease that
 * by one to save room for some metadata so 15 moves per thread with two bytes
 * extra, one for the index on each thread and one for the global write flag).
 * When one thread fills its buffer, it can set a bit that is readable by all
 * threads in the warp to signal that the buffer should be dumped to global
 * memory.
 *
 * We can make sure that all moves in the local write buffer are either present
 * or invalid. Threads 0-14 will copy moves generated by thread 0, then 16-311
 * for moves generated by thread 1. If the move is data is GPU_INVALID_MOVE
 * then the thread will skip its copy. The thread will then reset that data
 * in the local write buffer to GPU_INVALID_MOVE.
 *
 * Once the whole copy is done, all threads can reset their local indices and
 * keep on generating moves.
 *
 * This might not be necessary, we might not be global memory bottlenecked.
 * Alternatively, the better option might be to create a
 * make_move_and_count_child_moves function so that we don't have to store
 * any of the moves at depth d or d-1 (which dominate the work that we do as
 * there are ~35-38 moves per position on average).
 */
typedef struct {
    gpu_move_t *moves;
//  gpu_mv_wb_shared_t *buffer;
    uint8_t offset;
} gpu_mv_write_buf_t;

/* TODO: You could put the cool write coallescence structure here. */
typedef struct {
    
} gpu_mv_wb_shared_t;

#endif /* GPU_TYPES_H */

