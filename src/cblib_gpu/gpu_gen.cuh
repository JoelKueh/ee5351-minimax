
#ifndef GPU_GEN_H
#define GPU_GEN_H

/* TODO: Remove me. */
#include <inttypes.h>

#include <string.h>

#include "gpu_tables.cuh"
#include "gpu_types.cuh"
#include "gpu_move.cuh"
#include "gpu_board.cuh"
#include "gpu_history.cuh"
#include "gpu_bitutil.cuh"
#include "gpu_search_struct.cuh"
#include "gpu_ray_gen.cuh"
#include "gpu_dbg.cuh"

/* TODO: Move generation task. This is likely the hardest task.
 *
 * I will take care of pawn moves and castling. You don't need to worry about
 * that. Additionally, most of the move generation code from the CPU will
 * translate nicely to the GPU. There are some changes that I have considered
 * that might require some wider changes, but I'll save them for when we
 * have a working model.
 *
 * All in all, you will need to implmement
 */

__device__ static inline uint64_t gpu_pin_adjust(gpu_board_t *__restrict__ board,
        gpu_state_tables_t *__restrict__ state, uint8_t sq, uint64_t moves)
{
    uint8_t king_sq = gpu_peek_rbit(GPU_BB_KINGS(board->bb, board->turn));
    /* TODO: Remove me. */
    //printf("king_sq: %d, sq: %d, pin_ray: %" PRIu64 "\n", king_sq, sq,
    //        gpu_ray_through_sq2(king_sq, sq));
    return gpu_ray_through_sq2(king_sq, sq) & moves;
}

__device__ static inline uint64_t gpu_pawn_smear(
        uint64_t pawns, gpu_color_t color)
{
    return color == GPU_WHITE ?
        (pawns >> 9 & ~BB_RIGHT_COL) | (pawns >> 7 & ~BB_LEFT_COL) :
        (pawns << 7 & ~BB_RIGHT_COL) | (pawns << 9 & ~BB_LEFT_COL);
}

__device__ static inline uint64_t gpu_pawn_smear_left(
        uint64_t pawns, gpu_color_t color)
{
    return color == GPU_WHITE ?
        (pawns >> 9 & ~BB_RIGHT_COL) :
        (pawns << 9 & ~BB_LEFT_COL);
}

__device__ static inline uint64_t gpu_pawn_smear_forward(
        uint64_t pawns, gpu_color_t color)
{
    return color == GPU_WHITE ? pawns >> 8 : pawns << 8;
}

__device__ static inline uint64_t gpu_pawn_smear_right(
        uint64_t pawns, gpu_color_t color)
{
    return color == GPU_WHITE ?
        (pawns >> 7 & ~BB_LEFT_COL) :
        (pawns << 7 & ~BB_RIGHT_COL);
}

__device__ static inline void gpu_append_pinned_pawn_moves(
        gpu_search_struct_t *__restrict__ ss, gpu_board_t *__restrict__ board,
        gpu_state_tables_t *__restrict__ state, uint64_t pinned)
{
    while (pinned) {
        uint8_t sq = gpu_pop_rbit(&pinned);
        uint64_t mvmsk;
        uint8_t target;
        gpu_mv_flag_t flags;

        /* Generate the move mask. */
        mvmsk = gpu_pawn_smear_forward(UINT64_C(1) << sq, board->turn) & ~board->bb.occ;
        mvmsk |= gpu_pawn_smear_forward(mvmsk, board->turn) &
            (board->turn == GPU_WHITE ? BB_WHITE_PAWN_LINE : BB_BLACK_PAWN_LINE);
        mvmsk |= gpu_pawn_smear(UINT64_C(1) << sq, board->turn) &
            (~board->bb.color & board->bb.occ);
        mvmsk = gpu_pin_adjust(board, state, sq, mvmsk);

        /* Append all of the moves to the list. */
        while (mvmsk) {
            target = gpu_pop_rbit(&mvmsk);
            flags = (UINT64_C(1) << target) & board->bb.occ ? GPU_MV_CAPTURE : GPU_MV_QUIET;
            flags = target == sq + 16 || target == sq - 16 ? GPU_MV_DOUBLE_PAWN_PUSH : flags;
            gpu_ss_push_move(ss, gpu_mv_from_data(sq, target, flags));
        }
    }
}

__device__ static inline void gpu_append_pushes(
        gpu_search_struct_t *__restrict__ ss, gpu_board_t *__restrict__ board, uint64_t pushes)
{
    uint8_t target;
    uint8_t sq;

    while (pushes != 0) {
        target = gpu_pop_rbit(&pushes);
        sq = target + (board->turn == GPU_WHITE ? 8 : -8);
	    gpu_ss_push_move(ss, gpu_mv_from_data(sq, target, GPU_MV_QUIET));
    }
}

__device__ static inline void gpu_append_doubles(
        gpu_search_struct_t *__restrict__ ss,
        gpu_board_t *__restrict__ board, uint64_t doubles)
{
    uint8_t target;
    uint8_t sq;

    while (doubles != 0) {
        target = gpu_pop_rbit(&doubles);
        sq = target + (board->turn == GPU_WHITE ? 16 : -16);
        gpu_ss_push_move(ss, gpu_mv_from_data(sq, target, GPU_MV_DOUBLE_PAWN_PUSH));
    }
}

__device__ static inline void gpu_append_left_attacks(
        gpu_search_struct_t *__restrict__ ss,
        gpu_board_t *__restrict__ board, uint64_t left_attacks)
{
    uint8_t target;
    uint8_t sq;

    while (left_attacks != 0) {
        target = gpu_pop_rbit(&left_attacks);
        sq = target + (board->turn == GPU_WHITE ? 9 : -9);
        gpu_ss_push_move(ss, gpu_mv_from_data(sq, target, GPU_MV_CAPTURE));
    }
}

__device__ static inline void gpu_append_right_attacks(
        gpu_search_struct_t *__restrict__ ss,
        gpu_board_t *__restrict__ board, uint64_t right_attacks)
{
    uint8_t target;
    uint8_t sq;

    while (right_attacks != 0) {
        target = gpu_pop_rbit(&right_attacks);
        sq = target + (board->turn == GPU_WHITE ? 7 : -7);
        gpu_ss_push_move(ss, gpu_mv_from_data(sq, target, GPU_MV_CAPTURE));
    }
}

__device__ static inline void gpu_append_left_promos(
        gpu_search_struct_t *__restrict__ ss,
        gpu_board_t *__restrict__ board, uint64_t left_promos)
{
    uint8_t target;
    uint8_t sq;

    while (left_promos != 0) {
        target = gpu_pop_rbit(&left_promos);
        sq = target + (board->turn == GPU_WHITE ? 9 : -9);
        gpu_ss_push_move(ss, gpu_mv_from_data(sq, target, GPU_MV_KNIGHT_PROMO_CAPTURE));
        gpu_ss_push_move(ss, gpu_mv_from_data(sq, target, GPU_MV_BISHOP_PROMO_CAPTURE));
        gpu_ss_push_move(ss, gpu_mv_from_data(sq, target, GPU_MV_ROOK_PROMO_CAPTURE));
        gpu_ss_push_move(ss, gpu_mv_from_data(sq, target, GPU_MV_QUEEN_PROMO_CAPTURE));
    }
}

__device__ static inline void gpu_append_forward_promos(
        gpu_search_struct_t *__restrict__ ss,
        gpu_board_t *__restrict__ board, uint64_t forward_promos)
{
    uint8_t target;
    uint8_t sq;

    while (forward_promos != 0) {
        target = gpu_pop_rbit(&forward_promos);
        sq = target + (board->turn == GPU_WHITE ? 8 : -8);
        gpu_ss_push_move(ss, gpu_mv_from_data(sq, target, GPU_MV_KNIGHT_PROMO));
        gpu_ss_push_move(ss, gpu_mv_from_data(sq, target, GPU_MV_BISHOP_PROMO));
        gpu_ss_push_move(ss, gpu_mv_from_data(sq, target, GPU_MV_ROOK_PROMO));
        gpu_ss_push_move(ss, gpu_mv_from_data(sq, target, GPU_MV_QUEEN_PROMO));
    }
}

__device__ static inline void gpu_append_right_promos(
        gpu_search_struct_t *__restrict__ ss,
        gpu_board_t *__restrict__ board, uint64_t right_promos)
{
    uint8_t target;
    uint8_t sq;

    while (right_promos != 0) {
        target = gpu_pop_rbit(&right_promos);
        sq = target + (board->turn == GPU_WHITE ? 7 : -7);
        gpu_ss_push_move(ss, gpu_mv_from_data(sq, target, GPU_MV_KNIGHT_PROMO_CAPTURE));
        gpu_ss_push_move(ss, gpu_mv_from_data(sq, target, GPU_MV_BISHOP_PROMO_CAPTURE));
        gpu_ss_push_move(ss, gpu_mv_from_data(sq, target, GPU_MV_ROOK_PROMO_CAPTURE));
        gpu_ss_push_move(ss, gpu_mv_from_data(sq, target, GPU_MV_QUEEN_PROMO_CAPTURE));
    }
}

/* TODO: Move pinned pawn logic elsewhere. */
__device__ static inline void gpu_append_pawn_moves(
        gpu_search_struct_t *__restrict__ ss,
        gpu_board_t *__restrict__ board, gpu_state_tables_t *__restrict__ state)
{
    /* Get the mask of pawns that we want to evaluate. */
    uint64_t pawns = GPU_BB_PAWNS(board->bb, board->turn);

    /* First handle all pinned pawns. */
    uint64_t pinned = pawns & state->pinned;
    pawns &= ~state->pinned;
    gpu_append_pinned_pawn_moves(ss, board, state, pinned);

    /* Generate left attacks for pawns. */
    uint64_t left_smear = gpu_pawn_smear_left(pawns, board->turn);
    uint64_t left_attacks = left_smear & GPU_BB_COLOR(board->bb, !board->turn);
    left_attacks &= state->check_blocks;
    uint64_t left_promos = left_attacks & (BB_TOP_ROW | BB_BOTTOM_ROW);
    left_attacks ^= left_promos;
    gpu_append_left_attacks(ss, board, left_attacks);
    gpu_append_left_promos(ss, board, left_promos);

    /* Generate right attacks for pawns. */
    uint64_t right_smear = gpu_pawn_smear_right(pawns, board->turn);
    uint64_t right_attacks = right_smear & GPU_BB_COLOR(board->bb, !board->turn);
    right_attacks &= state->check_blocks;
    uint64_t right_promos = right_attacks & (BB_TOP_ROW | BB_BOTTOM_ROW);
    right_attacks ^= right_promos;
    gpu_append_right_attacks(ss, board, right_attacks);
    gpu_append_right_promos(ss, board, right_promos);

    /* Generate masks for pushing pawns. */
    uint64_t forward_smear = gpu_pawn_smear_forward(pawns, board->turn);
    uint64_t forward_moves = forward_smear & ~board->bb.occ;
    uint64_t double_smear = gpu_pawn_smear_forward(forward_moves, board->turn);
    forward_moves &= state->check_blocks;
    uint64_t forward_promos = forward_moves & (BB_TOP_ROW | BB_BOTTOM_ROW);
    forward_moves ^= forward_promos;
    gpu_append_pushes(ss, board, forward_moves);
    gpu_append_forward_promos(ss, board, forward_promos);

    /* Smear the forward moves again to get the double pushes. */
    uint64_t double_moves = double_smear & ~board->bb.occ;
    double_moves &= board->turn == GPU_WHITE ? BB_WHITE_PAWN_LINE : BB_BLACK_PAWN_LINE;
    double_moves &= state->check_blocks;
    gpu_append_doubles(ss, board, double_moves);
}

/* TODO: Simple moves cover pretty much everything. */
__device__ static inline void gpu_append_simple_moves(
        gpu_search_struct_t *__restrict__ ss,
        gpu_board_t *__restrict__ board, gpu_state_tables_t *__restrict__ state)
{
    uint8_t sq, target;
    gpu_mv_flag_t flags;
    uint64_t mvmsk;
    uint64_t pinned = 0;

    /* Generate knight moves. */
    uint64_t knights = GPU_BB_KNIGHTS(board->bb, board->turn);
    pinned = knights & state->pinned;
    knights ^= pinned;
    while (pinned) {
        sq = gpu_pop_rbit(&pinned);
        mvmsk = gpu_read_knight_atk_msk(sq);
        mvmsk &= ~GPU_BB_COLOR(board->bb, board->turn);
        mvmsk &= state->check_blocks;
        mvmsk = gpu_pin_adjust(board, state, sq, mvmsk);
        while (mvmsk) {
            target = gpu_pop_rbit(&mvmsk);
            flags = (UINT64_C(1) << target) & board->bb.occ ? GPU_MV_CAPTURE : GPU_MV_QUIET;
            gpu_ss_push_move(ss, gpu_mv_from_data(sq, target, flags));
        }
    }
    while (knights) {
        sq = gpu_pop_rbit(&knights);
        mvmsk = gpu_read_knight_atk_msk(sq);
        mvmsk &= ~GPU_BB_COLOR(board->bb, board->turn);
        mvmsk &= state->check_blocks;
        while (mvmsk) {
            target = gpu_pop_rbit(&mvmsk);
            flags = (UINT64_C(1) << target) & board->bb.occ ? GPU_MV_CAPTURE : GPU_MV_QUIET;
            gpu_ss_push_move(ss, gpu_mv_from_data(sq, target, flags));
        }
    }

    /* Generate bishop and queen moves. */
    uint64_t bishops = GPU_BB_B_AND_Q(board->bb, board->turn);
    pinned = bishops & state->pinned;
    bishops ^= pinned;
    while (pinned) {
        sq = gpu_pop_rbit(&pinned);
        mvmsk = gpu_read_bishop_atk_msk(sq, board->bb.occ);
        mvmsk &= ~GPU_BB_COLOR(board->bb, board->turn);
        mvmsk &= state->check_blocks;
        mvmsk = gpu_pin_adjust(board, state, sq, mvmsk);
        while (mvmsk) {
            target = gpu_pop_rbit(&mvmsk);
            flags = (UINT64_C(1) << target) & board->bb.occ ? GPU_MV_CAPTURE : GPU_MV_QUIET;
            gpu_ss_push_move(ss, gpu_mv_from_data(sq, target, flags));
        }
    }
    while (bishops) {
        sq = gpu_pop_rbit(&bishops);
        mvmsk = gpu_read_bishop_atk_msk(sq, board->bb.occ);
        mvmsk &= ~GPU_BB_COLOR(board->bb, board->turn);
        mvmsk &= state->check_blocks;
        while (mvmsk) {
            target = gpu_pop_rbit(&mvmsk);
            flags = (UINT64_C(1) << target) & board->bb.occ ? GPU_MV_CAPTURE : GPU_MV_QUIET;
            gpu_ss_push_move(ss, gpu_mv_from_data(sq, target, flags));
        }
    }

    /* Generate rook and queen moves. */
    uint64_t rooks = GPU_BB_R_AND_Q(board->bb, board->turn);
    pinned = rooks & state->pinned;
    rooks ^= pinned;
    while (pinned) {
        sq = gpu_pop_rbit(&pinned);
        mvmsk = gpu_read_rook_atk_msk(sq, board->bb.occ);
        mvmsk &= ~GPU_BB_COLOR(board->bb, board->turn);
        mvmsk &= state->check_blocks;
        mvmsk = gpu_pin_adjust(board, state, sq, mvmsk);
        while (mvmsk) {
            target = gpu_pop_rbit(&mvmsk);
            flags = (UINT64_C(1) << target) & board->bb.occ ? GPU_MV_CAPTURE : GPU_MV_QUIET;
            gpu_ss_push_move(ss, gpu_mv_from_data(sq, target, flags));
        }
    }
    while (rooks) {
        sq = gpu_pop_rbit(&rooks);
        mvmsk = gpu_read_rook_atk_msk(sq, board->bb.occ);
        mvmsk &= ~GPU_BB_COLOR(board->bb, board->turn);
        mvmsk &= state->check_blocks;
        while (mvmsk) {
            target = gpu_pop_rbit(&mvmsk);
            flags = (UINT64_C(1) << target) & board->bb.occ ? GPU_MV_CAPTURE : GPU_MV_QUIET;
            gpu_ss_push_move(ss, gpu_mv_from_data(sq, target, flags));
        }
    }

    /* Generate king moves. */
    uint64_t kings = GPU_BB_KINGS(board->bb, board->turn);
    /* Kings can't be pinned of course. */
    while (kings) {
        sq = gpu_pop_rbit(&kings);
        mvmsk = gpu_read_king_atk_msk(sq);
        mvmsk &= ~GPU_BB_COLOR(board->bb, board->turn);
        mvmsk &= ~state->threats;
        while (mvmsk) {
            target = gpu_pop_rbit(&mvmsk);
            flags = (UINT64_C(1) << target) & board->bb.occ ? GPU_MV_CAPTURE : GPU_MV_QUIET;
            gpu_ss_push_move(ss, gpu_mv_from_data(sq, target, flags));
        }
    }
}

__device__ static inline bool ksc_legal(gpu_board_t *__restrict__ board,
        gpu_state_tables_t *__restrict__ state)
{
    uint64_t occ_mask = board->turn == GPU_WHITE ? BB_WHITE_KING_SIDE_CASTLE_OCCUPANCY :
        BB_BLACK_KING_SIDE_CASTLE_OCCUPANCY;
    uint64_t check_mask = board->turn == GPU_WHITE ? BB_WHITE_KING_SIDE_CASTLE_CHECK :
        BB_BLACK_KING_SIDE_CASTLE_CHECK;

    /* If the occupancy intersects occ_mask or the threats intersect ckeck_mask. No castling. */
    return ((board->bb.occ & occ_mask) | (state->threats & check_mask)) == 0
        && gpu_state_has_ksc(board->state, board->turn);
}

__device__ static inline bool qsc_legal(gpu_board_t *__restrict__ board,
        gpu_state_tables_t *__restrict__ state)
{
    uint64_t occ_mask = board->turn == GPU_WHITE ? BB_WHITE_QUEEN_SIDE_CASTLE_OCCUPANCY :
        BB_BLACK_QUEEN_SIDE_CASTLE_OCCUPANCY;
    uint64_t check_mask = board->turn == GPU_WHITE ? BB_WHITE_QUEEN_SIDE_CASTLE_CHECK :
        BB_BLACK_QUEEN_SIDE_CASTLE_CHECK;

    /* If the occupancy intersects occ_mask or the threats intersect ckeck_mask. No castling. */
    return ((board->bb.occ & occ_mask) | (state->threats & check_mask)) == 0
        && gpu_state_has_qsc(board->state, board->turn);
}

__device__ static inline void gpu_append_castle_moves(
        gpu_search_struct_t *__restrict__ ss,
        gpu_board_t *__restrict__ board, gpu_state_tables_t *__restrict__ state)
{
    uint8_t from = board->turn == GPU_WHITE ? M_WHITE_KING_START : M_BLACK_KING_START;
    uint8_t to;

    if (ksc_legal(board, state)) {
        to = board->turn == GPU_WHITE ? M_WHITE_KING_SIDE_CASTLE_TARGET :
            M_BLACK_KING_SIDE_CASTLE_TARGET;
        gpu_ss_push_move(ss, gpu_mv_from_data(from, to, GPU_MV_KING_SIDE_CASTLE));
    }

    if (qsc_legal(board, state)) {
        to = board->turn == GPU_WHITE ? M_WHITE_QUEEN_SIDE_CASTLE_TARGET :
            M_BLACK_QUEEN_SIDE_CASTLE_TARGET;
        gpu_ss_push_move(ss, gpu_mv_from_data(from, to, GPU_MV_QUEEN_SIDE_CASTLE));
    }
}

/* TODO: I got this one, it's about enpassant. */
__device__ void gpu_append_enp_moves(
        gpu_search_struct_t *__restrict__ ss,
        gpu_board_t *__restrict__ board, gpu_state_tables_t *__restrict__ state)
{
    /* NOTE: Enpassants are rare, so this function is allowed to be slow. */

    /* Exit early if there is not availiable enpassant. */
    if (!gpu_state_enp_available(board->state))
        return;

    /* Get the squares relavent to the piece that can enpassant. */
    uint8_t enp_row_start = board->turn == GPU_WHITE ? M_BLACK_MIN_ENPASSANT_TARGET :
        M_WHITE_MIN_ENPASSANT_TARGET;
    uint8_t enp_sq = enp_row_start + gpu_state_enp_col(board->state);
    uint8_t enemy_sq = enp_sq + (board->turn == GPU_WHITE ? 8 : -8);

    /* Get all of the pieces that can enpassnt. */
    uint64_t enp_sources = gpu_pawn_smear(UINT64_C(1) << enp_sq, !board->turn)
        & GPU_BB_PAWNS(board->bb, board->turn);

    /* Loop through the pieces that can enpassant and generate the moves. */
    uint8_t sq, king_sq;
    gpu_move_t mv;
    uint64_t new_occ, bishop_threats, rook_threats;
    while (enp_sources) {
        sq = gpu_pop_rbit(&enp_sources);
        mv = gpu_mv_from_data(sq, enp_sq, GPU_MV_ENPASSANT);

        /* Update the occupancy mask to what it will be after the move takes place. */
        new_occ = board->bb.occ;
        new_occ &= ~(UINT64_C(1) << sq);
        new_occ &= ~(UINT64_C(1) << enemy_sq);
        new_occ |= UINT64_C(1) << enp_sq;

        /* Check if the king is in check after the move is made.
         * This could be the case if some piece was pinned before the enpassant was made. */
        king_sq = gpu_peek_rbit(GPU_BB_KINGS(board->bb, board->turn));

        bishop_threats = gpu_read_bishop_atk_msk(king_sq, new_occ)
            & GPU_BB_B_AND_Q(board->bb, !board->turn);
        if (bishop_threats) continue;

        rook_threats = gpu_read_rook_atk_msk(king_sq, new_occ)
            & GPU_BB_R_AND_Q(board->bb, !board->turn);
        if (rook_threats) continue;

        /* Push the move if it doesn't cause any problems. */
        gpu_ss_push_move(ss, mv);
    }
}

/* TODO: This one is pretty simple to change. */
__device__ static inline void gpu_gen_moves(
        gpu_search_struct_t *__restrict__ ss,
        gpu_board_t *__restrict__ board, gpu_state_tables_t *__restrict__ state)
{
    gpu_append_pawn_moves(ss, board, state);
    gpu_append_simple_moves(ss, board, state);
    gpu_append_castle_moves(ss, board, state);
    gpu_append_enp_moves(ss, board, state);
}

__device__ static inline uint64_t gpu_gen_threats(
        gpu_board_t *__restrict__ board)
{
    uint64_t threats;

    /* Remove the king to allow pieces to "see through" it. */
    uint64_t occ = board->bb.occ ^ GPU_BB_KINGS(board->bb, board->turn);

    /* Generate pawn threats. */
    threats = gpu_pawn_smear(GPU_BB_PAWNS(board->bb, !board->turn), !board->turn);

    /* Generate knight threats. */
    uint64_t knights = GPU_BB_KNIGHTS(board->bb, !board->turn);
    /* TODO: Fast, non-lookup knight move generation?
     * https://www.chessprogramming.org/Knight_Pattern.
     */
    while (knights) {
        threats |= gpu_read_knight_atk_msk(gpu_pop_rbit(&knights));
    }

    /* Generate bishop threats. */
    uint64_t bishops = GPU_BB_B_AND_Q(board->bb, !board->turn);
    /* TODO: Fast, non-lookup bishop move generation (kogge-stone)?
     * https://www.chessprogramming.org/Kogge-Stone_Algorithm.
     */
    while (bishops) {
        threats |= gpu_read_bishop_atk_msk(gpu_pop_rbit(&bishops), occ);
    }

    /* Generate rook threats. */
    uint64_t rooks = GPU_BB_R_AND_Q(board->bb, !board->turn);
    /* TODO: Fast, non-lookup rook move generation (kogge-stone)?
     * https://www.chessprogramming.org/Kogge-Stone_Algorithm.
     */
    while (rooks) {
        threats |= gpu_read_rook_atk_msk(gpu_pop_rbit(&rooks), occ);
    }

    /* Generate king threats. */
    /* TODO: Fast, non-lookup king move generation.
     * https://www.chessprogramming.org/Kogge-Stone_Algorithm.
     */
    uint64_t kings = GPU_BB_KINGS(board->bb, !board->turn);
    threats |= gpu_read_king_atk_msk(gpu_peek_rbit(kings));

    return threats;
}

/* TODO: This one should be a straightforward translation. */
__device__ static inline uint64_t gpu_gen_checks(
        gpu_board_t *__restrict__ board, uint64_t threats)
{
    uint64_t king = GPU_BB_KINGS(board->bb, board->turn);

    /* Exit early if the king isn't threatened. */
    if ((king & threats) == 0)
        return 0;

    /* Build the list of pieces that check the king. */
    uint8_t king_sq = gpu_peek_rbit(king);
    uint64_t checks = gpu_pawn_smear(UINT64_C(1) << king_sq, board->turn)
        & GPU_BB_PAWNS(board->bb, !board->turn);
    checks |= gpu_read_knight_atk_msk(king_sq)
        & GPU_BB_KNIGHTS(board->bb, !board->turn);
    checks |= gpu_read_bishop_atk_msk(king_sq, board->bb.occ)
        & GPU_BB_B_AND_Q(board->bb, !board->turn);
    checks |= gpu_read_rook_atk_msk(king_sq, board->bb.occ)
        & GPU_BB_R_AND_Q(board->bb, !board->turn);
    /* Here's a helpful reminder that a king can never check another king. */

    return checks;
}

/* TODO: This one should be a straightforward translation. */
__device__ static inline uint64_t gpu_gen_check_blocks(
        gpu_board_t *__restrict__ board, uint64_t checks)
{
    if (checks == 0)
        return BB_FULL;
    else if (gpu_popcnt(checks) != 1)
        return BB_EMPTY;

    uint8_t king_sq = gpu_peek_rbit(GPU_BB_KINGS(board->bb, board->turn));
    uint8_t check_sq = gpu_peek_rbit(checks);
    return gpu_read_tf_table(check_sq, king_sq) | (UINT64_C(1) << check_sq);
}

/* TODO: This one should be a straightforward translation.
 *
 * I'd recommend looking this one up on Chess Programming Wiki.
 */
__device__ static inline uint64_t gpu_xray_bishop_attacks(
        uint64_t occ, uint64_t blockers, uint8_t sq)
{
    uint64_t attacks = gpu_read_bishop_atk_msk(sq, occ);
    blockers &= attacks;
    return attacks ^ gpu_read_bishop_atk_msk(sq, occ ^ blockers);
}

/* TODO: This one should be a straightforward translation.
 *
 * I'd recommend looking this one up on Chess Programming Wiki.
 */
__device__ static inline uint64_t gpu_xray_rook_attacks(
        uint64_t occ, uint64_t blockers, uint8_t sq)
{
    uint64_t attacks = gpu_read_rook_atk_msk(sq, occ);
    blockers &= attacks;
    return attacks ^ gpu_read_rook_atk_msk(sq, occ ^ blockers);
}

/* TODO: This fucntion needs serious adjustment. */
__device__ static inline uint64_t gpu_gen_pins(gpu_board_t *__restrict__ board)
{
    uint8_t king_sq = gpu_peek_rbit(GPU_BB_KINGS(board->bb, board->turn));
    uint64_t blockers = GPU_BB_COLOR(board->bb, board->turn);
    uint64_t pinned = 0;
    uint64_t pinner;
    uint8_t sq;

    /* Get all of the first pinners. */
    pinner = gpu_xray_bishop_attacks(board->bb.occ, blockers, king_sq)
        & GPU_BB_B_AND_Q(board->bb, !board->turn);
    while (pinner) {
        sq = gpu_pop_rbit(&pinner);
        pinned |= gpu_read_tf_table(sq, king_sq) & blockers;
    }

    /* Get all of the second pinners. */
    pinner = gpu_xray_rook_attacks(board->bb.occ, blockers, king_sq)
        & GPU_BB_R_AND_Q(board->bb, !board->turn);
    while (pinner) {
        sq = gpu_pop_rbit(&pinner);
        pinned |= gpu_read_tf_table(sq, king_sq) & blockers;
    }

    return pinned;
}

/* TODO: Move Generation Task.
 *
 * These functions will need to be adjusted for the GPU.
 * In particular, I think we will need to take a look at the way I handle
 * pins. It is kindof stupid. See notes above and in gpu_types.h.
 */
__device__ static inline void gpu_gen_board_tables(gpu_board_t *__restrict__ board,
        gpu_state_tables_t *__restrict__ state)
{
    /* TODO: Remove me */
    //gpu_print_bitboard(board);
    state->threats = gpu_gen_threats(board);
    state->checks = gpu_gen_checks(board, state->threats);
    state->check_blocks = gpu_gen_check_blocks(board, state->checks);
    state->pinned = gpu_gen_pins(board);
}

#endif /* GPU_GEN_H */
