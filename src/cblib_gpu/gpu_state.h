
#ifndef GPU_STATE_H
#define GPU_STATE_H

#include "gpu_types.h"
#include "cb_const.h"

/* Masks for working with enpassant. */
#define GPU_STATE_ENP_COL (UINT64_C(0b111) << 5)
#define GPU_STATE_HAS_ENP (UINT64_C(1) << 4)

/**
 * Returns true if the player has the right to king side castle, false otherwise.
 */
__device__ static inline bool gpu_state_has_ksc(gpu_board_t *restrict b, cb_color_t color)
{
    return (b->state.raw & 0b10 << color * 2) != 0;
}

/**
 * Returns true if the player still has the right to queen side castle, false otherwise.
 */
__device__ static inline bool gpu_state_has_qsc(gpu_board_t *restrict b, cb_color_t color)
{
    return (b->state.raw & 0b1 << color * 2) != 0;
}

/**
 * Removes the right to king side castle.
 */
__device__ static inline void gpu_state_remove_ksc(gpu_board_t *restrict b, cb_color_t color)
{
    b->state.raw &= ~(0b10 << color * 2);
}

/**
 * Removes the right to queen side castle.
 */
__device__ static inline void gpu_state_remove_qsc(gpu_board_t *restrict b, cb_color_t color)
{
    b->state.raw &= ~(0b1 << color * 2);
}

/**
 * Removes all castling rights for a specified color.
 */
__device__ static inline void gpu_state_remove_castle(gpu_board_t *restrict b, cb_color_t color)
{
    b->state.raw &= ~(0b11 << color * 2);
}


/**
 * Adds king side castling right.
 */
__device__ static inline void gpu_state_add_ksc(gpu_board_t *restrict b, cb_color_t color)
{
    b->state.raw |= 0b10 << color * 2;
}

/**
 * Adds queen side castling right.
 */
__device__ static inline void gpu_state_add_qsc(gpu_board_t *restrict b, cb_color_t color)
{
    b->state.raw |= 0b1 << color * 2;
}

/**
 * Removes all castling rights for specified color.
 */
__device__ static inline void gpu_state_add_castle(gpu_board_t *restrict b, cb_color_t color)
{
    b->state.raw |= 0b11 << color * 2;
}


/**
 * Returns true if there is an enpassant availiable.
 */
__device__ static inline bool gpu_state_enp_availiable(gpu_board_t *restrict b)
{
    return (b->state.raw & GPU_STATE_ENP_COL) != 0;
}

__device__ static inline uint8_t gpu_state_enp_col(gpu_board_t *restrict b)
{
    return (b->state.raw & GPU_STATE_ENP_COL) >> 5;
}

/**
 * Sets up this move state to open an enpassant square.
 */
__device__ static inline void gpu_state_set_enp(gpu_board_t *restrict b, uint8_t enp_col)
{
    b->state.raw = (b->state.raw & ~GPU_STATE_ENP_COL) | (enp_col << 5);
    b->state.raw |= GPU_STATE_HAS_ENP;
}

/**
 * Removes enpassant from the history state.
 */
__device__ static inline void gpu_state_decay_enp(gpu_board_t *restrict b)
{
    b->state.raw &= ~GPU_STATE_ENP_COL;
}

/**
 * Decays castle rights after a move.
 */
__device__ static inline void gpu_state_decay_castle_rights(
        gpu_board_t *restrict b, uint8_t color,
        uint8_t to, uint8_t from)
{
    /* Remove castling rights for moving a king or rook. */
    b->state.raw &= from == M_WHITE_KING_START ? ~UINT16_C(0b1100) : 0xFFFF;
    b->state.raw &= from == M_BLACK_KING_START ? ~UINT16_C(  0b11) : 0xFFFF;
    b->state.raw &= from == M_WHITE_KING_SIDE_ROOK_START ? ~UINT16_C(0b1000) : 0xFFFF;
    b->state.raw &= from == M_BLACK_KING_SIDE_ROOK_START ? ~UINT16_C(  0b10) : 0xFFFF;
    b->state.raw &= from == M_WHITE_QUEEN_SIDE_ROOK_START ? ~UINT16_C(0b100) : 0xFFFF;
    b->state.raw &= from == M_BLACK_QUEEN_SIDE_ROOK_START ? ~UINT16_C(  0b1) : 0xFFFF;

    /* Remove castling rights for taking a rook. */
    b->state.raw &= to == M_WHITE_KING_SIDE_ROOK_START ? ~UINT16_C(0b1000) : 0xFFFF;
    b->state.raw &= to == M_BLACK_KING_SIDE_ROOK_START ? ~UINT16_C(  0b10) : 0xFFFF;
    b->state.raw &= to == M_WHITE_QUEEN_SIDE_ROOK_START ? ~UINT16_C(0b100) : 0xFFFF;
    b->state.raw &= to == M_BLACK_QUEEN_SIDE_ROOK_START ? ~UINT16_C(  0b1) : 0xFFFF;
}

#endif /* GPU_STATE_H */

