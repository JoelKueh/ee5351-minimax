
#ifndef GPU_LIB_H
#define GPU_LIB_H

#include <threads.h>

#include "gpu_dbg.cuh"
#include "gpu_const.cuh"
#include "gpu_move.cuh"
#include "gpu_board.cuh"
#include "gpu_history.cuh"
#include "gpu_search_struct.cuh"

__device__ __forceinline__ void gpu_make(
        gpu_board_t *__restrict__ board, const gpu_move_t mv)
{
    /* Fields of the move and variables for board state. */
    gpu_history_t old_state = board->state;
    gpu_mv_flag_t flag = (gpu_mv_flag_t)gpu_mv_get_flags(mv);
    uint8_t to = gpu_mv_get_to(mv);
    uint8_t from = gpu_mv_get_from(mv);

    /* Variables for all moves. */
    gpu_ptype_t ptype;
    gpu_ptype_t new_ptype;
    gpu_ptype_t cap_ptype;
    gpu_history_t new_state = old_state;

    /* Variables for castles. */
    uint8_t rook_from;
    uint8_t rook_to;

    /* Variables for enp. */
    int8_t direction;

    /* Decay enpassant. */
    gpu_state_decay_enp(&new_state);

    /* Handle enpassant separately (it's rare so divergence is fine). */
    if (flag == GPU_MV_ENPASSANT) {
        direction = board->turn == GPU_WHITE ? 8 : -8;
        gpu_state_set_captured_piece(&new_state, GPU_PTYPE_PAWN);
        gpu_write_piece(board, to, GPU_PTYPE_PAWN, board->turn);
        gpu_delete_piece(board, from, GPU_PTYPE_PAWN, board->turn);
        gpu_delete_piece(board, to + direction, GPU_PTYPE_PAWN, !board->turn);
        goto out_save_new_state;
    }

    /* Read the piece type from the board. */
    ptype = gpu_ptype_at_sq(board, from);
    cap_ptype = gpu_ptype_at_sq(board, to);

    /* TODO: Remove me. */
    //if (cap_ptype == GPU_PTYPE_KING) {
    //    cap_ptype = GPU_PTYPE_KING;
    //    printf("Error\n");
    //}

    /* If a piece was captured, set it in the board state. */
    gpu_state_set_captured_piece(&new_state, cap_ptype);
    gpu_state_decay_castle_rights(&new_state, board->turn, to, from);

    /* Piece type changes if this is a promotion. Remember that
     *  - Flag types are sequential in lowest 2 bits.
     *  - Only promos have the 4th bit of the flag set. */
    new_ptype = ptype + 1 + ((flag & (0b11 << 12)) >> 12);
    new_ptype = (flag & (0b1000 << 12)) ? new_ptype : ptype;

    /* Move the piece from its previous position to its new position. */
    if (cap_ptype != GPU_PTYPE_EMPTY)
        gpu_delete_piece(board, to, cap_ptype, board->turn);
    gpu_write_piece(board, to, new_ptype, board->turn);
    gpu_delete_piece(board, from, ptype, board->turn);

    /* Set the enpassant field if this was a double pawn push. */
    if (flag == GPU_MV_DOUBLE_PAWN_PUSH)
        gpu_state_set_enp(&new_state, to & 0b111);

    /* Extra work for king side castling. */
    if (flag == GPU_MV_KING_SIDE_CASTLE) {
        rook_from = board->turn ? M_WHITE_KING_SIDE_ROOK_START :
            M_BLACK_KING_SIDE_ROOK_START;
        rook_to = board->turn ? M_WHITE_KING_SIDE_ROOK_TARGET :
            M_BLACK_KING_SIDE_ROOK_TARGET;
        gpu_delete_piece(board, rook_from, GPU_PTYPE_ROOK, board->turn);
        gpu_write_piece(board, rook_to, GPU_PTYPE_ROOK, board->turn);
    }

    /* Extra work for queen side castling. */
    if (flag == GPU_MV_QUEEN_SIDE_CASTLE) {
        rook_from = board->turn ? M_WHITE_QUEEN_SIDE_ROOK_START :
            M_BLACK_QUEEN_SIDE_ROOK_START;
        rook_to = board->turn ? M_WHITE_QUEEN_SIDE_ROOK_TARGET :
            M_BLACK_QUEEN_SIDE_ROOK_TARGET;
        gpu_delete_piece(board, rook_from, GPU_PTYPE_ROOK, board->turn);
        gpu_write_piece(board, rook_to, GPU_PTYPE_ROOK, board->turn);
    }

    /* Save the new state to the stack. */
out_save_new_state:
    board->turn = !board->turn;
    board->state = new_state;
}

#endif /* GPU_LIB_H */

