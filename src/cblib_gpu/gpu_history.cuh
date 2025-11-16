
#ifndef GPU_STATE_H
#define GPU_STATE_H

#include "gpu_types.cuh"
#include "gpu_const.cuh"

/* Masks for working with enpassant. */
#define GPU_STATE_ENP_COL (UINT64_C(0b111) << 5)
#define GPU_STATE_HAS_ENP (UINT64_C(1) << 4)

/**
 * @breif Writes a history element to the stack and increases depth.
 */
__device__ static inline void gpu_search_struct_push(
        gpu_search_struct_t *restrict s, gpu_hist_ele_t ele)
{
    s->hist[s->depth++] = ele;
}

/**
 * @breif Pops a history element off the stack and decreases depth.
 */
__device__ static inline gpu_hist_ele_t gpu_search_struct_pop(
        gpu_search_struct_t *restrict s)
{
    return s->hist[--s->depth];
}

/**
 * Returns true if the player has the right to king side castle, false otherwise.
 */
__device__ static inline bool gpu_state_has_ksc(
        gpu_history_t hist, gpu_color_t color)
{
    return (hist & 0b10 << color * 2) != 0;
}

/**
 * Returns true if the player still has the right to queen side castle, false otherwise.
 */
__device__ static inline bool gpu_state_has_qsc(
        gpu_history_t hist, gpu_color_t color)
{
    return (hist & 0b1 << color * 2) != 0;
}

/**
 * Removes the right to king side castle.
 */
__device__ static inline void gpu_state_remove_ksc(
        gpu_history_t hist, gpu_color_t color)
{
    hist &= ~(0b10 << color * 2);
}

/**
 * Removes the right to queen side castle.
 */
__device__ static inline void gpu_state_remove_qsc(
        gpu_history_t hist, gpu_color_t color)
{
    hist &= ~(0b1 << color * 2);
}

/**
 * Removes all castling rights for a specified color.
 */
__device__ static inline void gpu_state_remove_castle(
        gpu_history_t hist, gpu_color_t color)
{
    hist &= ~(0b11 << color * 2);
}


/**
 * Adds king side castling right.
 */
__device__ static inline void gpu_state_add_ksc(
        gpu_history_t hist, gpu_color_t color)
{
    hist |= 0b10 << color * 2;
}

/**
 * Adds queen side castling right.
 */
__device__ static inline void gpu_state_add_qsc(
        gpu_history_t hist, gpu_color_t color)
{
    hist |= 0b1 << color * 2;
}

/**
 * Removes all castling rights for specified color.
 */
__device__ static inline void gpu_state_add_castle(
        gpu_history_t hist, gpu_color_t color)
{
    hist |= 0b11 << color * 2;
}


/**
 * Returns true if there is an enpassant availiable.
 */
__device__ static inline bool gpu_state_enp_availiable(gpu_history_t hist)
{
    return (hist & GPU_STATE_ENP_COL) != 0;
}

__device__ static inline uint8_t gpu_state_enp_col(gpu_history_t hist)
{
    return (hist & GPU_STATE_ENP_COL) >> 5;
}

/**
 * Sets up this move state to open an enpassant square.
 */
__device__ static inline void gpu_state_set_enp(
        gpu_history_t *restrict hist, uint8_t enp_col)
{
    *hist = (*hist & ~GPU_STATE_ENP_COL) | (enp_col << 5);
    *hist |= GPU_STATE_HAS_ENP;
}

/**
 * Removes enpassant from the history state.
 */
__device__ static inline void gpu_state_decay_enp(
        gpu_history_t *restrict hist)
{
    *hist &= ~GPU_STATE_ENP_COL;
}

/**
 * Decays castle rights after a move.
 */
__device__ static inline void gpu_state_decay_castle_rights(
        gpu_history_t *restrict hist, uint8_t color, uint8_t to, uint8_t from)
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

#endif /* GPU_STATE_H */

