
#ifndef GPU_TYPES_H
#define GPU_TYPES_H

#include <stdint.h>
#include <stdbool.h>
#include <cb_types.h>

/* Board Representation. */
typedef struct {
    uint64_t color;         /**< Bitmasks for colored pieces. */
    uint64_t piece[5];      /**< A set of bitmasks for piece types. */
} gpu_bitboard_t;

/* Move representation. */
typedef uint16_t gpu_move_t;

/* Full board representation. Flags packed into pawn bitboard as follows.
 * Credit to Ankan Banerjee for recommending this packing in his GPU chess
 * engine https://github.com/ankan-ban/perft_gpu.
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
typedef union {
    gpu_bitboard_t bb;  /**< Bitboard field in the union. */
    struct {
        uint64_t u0;    /**< Unused bits for the struct. */
        uint64_t raw;   /**< Raw bits of pawn bitboard. */
        uint64_t u1[4]; /**< Unnecessary bits in the struct. */
    } state;
} gpu_board_t;

/**
 * @breif State table data structure that is useful in move generation.
 *
 * NOTE: I've adjusted this
 */
typedef struct {
    uint64_t threats;       /**< A bitmask for all pieces that threaten the king. */
    uint64_t checks;        /**< A bitmask for all pieces that check the king. */
    uint64_t check_blocks;  /**< A bitmask for all squares that can break a check. */
    uint64_t pinned;        /**< A bitmask for all pinned pieces. */
} cb_state_tables_t;

#endif /* GPU_TYPES_H */
