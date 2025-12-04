
#ifndef GPU_COUNT_MOVES_H
#define GPU_COUNT_MOVES_H

#include "gpu_gen.cuh"

__device__ __forceinline__ uint8_t gpu_count_pinned_pawn_moves(
        gpu_board_t *__restrict__ board, gpu_state_tables_t *__restrict__ state,
        uint64_t pinned)
{
    uint8_t count = 0;

    while (pinned) {
        uint8_t sq = gpu_pop_rbit(&pinned);
        uint64_t mvmsk;
        uint8_t target;
        gpu_mv_flag_t flags;

        /* Generate the move mask. */
        mvmsk = gpu_pawn_smear_forward(UINT64_C(1) << sq, board->turn) & ~board->bb.occ;
        mvmsk |= gpu_pawn_smear_forward(mvmsk, board->turn) & ~board->bb.occ &
            (board->turn == GPU_WHITE ? BB_WHITE_PAWN_LINE : BB_BLACK_PAWN_LINE);
        mvmsk |= gpu_pawn_smear(UINT64_C(1) << sq, board->turn) &
            GPU_BB_COLOR(board->bb, !board->turn);
        mvmsk = gpu_pin_adjust(board, state, sq, mvmsk);
        mvmsk &= state->check_blocks;

        /* Append all of the moves to the list. */
        while (mvmsk) {
            target = gpu_pop_rbit(&mvmsk);
            flags = (UINT64_C(1) << target) & board->bb.occ ? GPU_MV_CAPTURE : GPU_MV_QUIET;
            flags = target == sq + 16 || target == sq - 16 ? GPU_MV_DOUBLE_PAWN_PUSH : flags;
            count += 1;
        }
    }

    return count;
}

__device__ __forceinline__ uint8_t gpu_count_pawn_moves(
        gpu_board_t *__restrict__ board, gpu_state_tables_t *__restrict__ state)
{
    uint8_t count = 0;

    /* Get the mask of pawns that we want to evaluate. */
    uint64_t pawns = GPU_BB_PAWNS(board->bb, board->turn);

    /* First handle all pinned pawns. */
    uint64_t pinned = pawns & state->pinned;
    pawns &= ~state->pinned;
    count += gpu_count_pinned_pawn_moves(board, state, pinned);

    /* Generate left attacks for pawns. */
    uint64_t left_smear = gpu_pawn_smear_left(pawns, board->turn);
    uint64_t left_attacks = left_smear & GPU_BB_COLOR(board->bb, !board->turn);
    left_attacks &= state->check_blocks;
    uint64_t left_promos = left_attacks & (BB_TOP_ROW | BB_BOTTOM_ROW);
    left_attacks ^= left_promos;
    count += gpu_popcnt(left_attacks);
    count += 4 * gpu_popcnt(left_promos);

    /* Generate right attacks for pawns. */
    uint64_t right_smear = gpu_pawn_smear_right(pawns, board->turn);
    uint64_t right_attacks = right_smear & GPU_BB_COLOR(board->bb, !board->turn);
    right_attacks &= state->check_blocks;
    uint64_t right_promos = right_attacks & (BB_TOP_ROW | BB_BOTTOM_ROW);
    right_attacks ^= right_promos;
    count += gpu_popcnt(right_attacks);
    count += 4 * gpu_popcnt(right_promos);

    /* Generate masks for pushing pawns. */
    uint64_t forward_smear = gpu_pawn_smear_forward(pawns, board->turn);
    uint64_t forward_moves = forward_smear & ~board->bb.occ;
    uint64_t double_smear = gpu_pawn_smear_forward(forward_moves, board->turn);
    forward_moves &= state->check_blocks;
    uint64_t forward_promos = forward_moves & (BB_TOP_ROW | BB_BOTTOM_ROW);
    forward_moves ^= forward_promos;
    count += gpu_popcnt(forward_moves);
    count += 4 * gpu_popcnt(forward_promos);

    /* Smear the forward moves again to get the double pushes. */
    uint64_t double_moves = double_smear & ~board->bb.occ;
    double_moves &= board->turn == GPU_WHITE ? BB_WHITE_PAWN_LINE : BB_BLACK_PAWN_LINE;
    double_moves &= state->check_blocks;
    count += gpu_popcnt(double_moves);

    return count;
}

__device__ __forceinline__ uint8_t gpu_count_simple_moves(
        gpu_board_t *__restrict__ board, gpu_state_tables_t *__restrict__ state)
{
    uint8_t count = 0;
    uint8_t sq, target;
    gpu_mv_flag_t flags;
    uint64_t mvmsk;
    uint64_t pinned = 0;

    /* Mask for what squares sliding pieces are allowed to move onto. */
    uint64_t slider_allow_mask =
        ~GPU_BB_COLOR(board->bb, board->turn) & state->check_blocks;

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
        count += gpu_popcnt(mvmsk);
    }
    while (knights) {
        sq = gpu_pop_rbit(&knights);
        mvmsk = gpu_read_knight_atk_msk(sq);
        mvmsk &= ~GPU_BB_COLOR(board->bb, board->turn);
        mvmsk &= state->check_blocks;
        count += gpu_popcnt(mvmsk);
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
        count += gpu_popcnt(mvmsk);
    }
    count += gpu_popcnt(gpu_north_east_atk(bishops, board->bb.occ) & slider_allow_mask);
    count += gpu_popcnt(gpu_north_west_atk(bishops, board->bb.occ) & slider_allow_mask);
    count += gpu_popcnt(gpu_south_west_atk(bishops, board->bb.occ) & slider_allow_mask);
    count += gpu_popcnt(gpu_south_east_atk(bishops, board->bb.occ) & slider_allow_mask);

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
        count += gpu_popcnt(mvmsk);
    }
    count += gpu_popcnt(gpu_east_atk(rooks, board->bb.occ) & slider_allow_mask);
    count += gpu_popcnt(gpu_north_atk(rooks, board->bb.occ) & slider_allow_mask);
    count += gpu_popcnt(gpu_west_atk(rooks, board->bb.occ) & slider_allow_mask);
    count += gpu_popcnt(gpu_south_atk(rooks, board->bb.occ) & slider_allow_mask);

    /* Generate king moves. */
    sq = gpu_peek_rbit(GPU_BB_KINGS(board->bb, board->turn));
    mvmsk = gpu_read_king_atk_msk(sq);
    mvmsk &= ~GPU_BB_COLOR(board->bb, board->turn);
    mvmsk &= ~state->threats;
    count += gpu_popcnt(mvmsk);

    return count;
}

__device__ __forceinline__ uint8_t gpu_count_castle_moves(
        gpu_board_t *__restrict__ board, gpu_state_tables_t *__restrict__ state)
{
    uint8_t count = 0;
    count += ksc_legal(board, state) ? 1 : 0;
    count += qsc_legal(board, state) ? 1 : 0;
    return count;
}

__device__ __forceinline__ uint8_t gpu_count_enp_moves(
        gpu_board_t *__restrict__ board, gpu_state_tables_t *__restrict__ state)
{
    /* Exit early if there is not availiable enpassant. */
    uint8_t count = 0;
    if (!gpu_state_enp_available(board->state))
        return 0;

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
    uint64_t new_occ, bishop_threats, rook_threats;
    while (enp_sources) {
        sq = gpu_pop_rbit(&enp_sources);

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

        count += 1;
    }

    return count;
}

__device__ __forceinline__ uint8_t gpu_count_moves(
        gpu_board_t *__restrict__ board, gpu_state_tables_t *__restrict__ state)
{
    uint8_t count = 0;
    count += gpu_count_pawn_moves(board, state);
    count += gpu_count_simple_moves(board, state);
    count += gpu_count_castle_moves(board, state);
    count += gpu_count_enp_moves(board, state);
    return count;
}

#endif /* GPU_COUNT_MOVES_H */
