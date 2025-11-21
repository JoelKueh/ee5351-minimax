
#include <string.h>

#include "gpu_types.cuh"
#include "gpu_move.cuh"
#include "gpu_board.cuh"
#include "gpu_history.cuh"
#include "gpu_bitutil.cuh"

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

__device__ static inline void gpu_append_pushes(
		gpu_move_t *__restrict__ moves, uint32_t *__restrict__ offset,
		gpu_board_t *__restrict__ board, uint64_t pushes)
{
    uint8_t target;
    uint8_t sq;

    while (pushes != 0) {
        target = gpu_pop_rbit(&pushes);
        sq = target + (board->turn == GPU_WHITE ? 8 : -8);
	    moves[(*offset)++] = gpu_mv_from_data(sq, target, GPU_MV_QUIET);
    }
}

__device__ static inline void gpu_append_doubles(
        gpu_move_t *__restrict__ moves, uint32_t *__restrict__ offset,
        gpu_board_t *__restrict__ board, uint64_t doubles)
{
    uint8_t target;
    uint8_t sq;

    while (doubles != 0) {
        target = gpu_pop_rbit(&doubles);
        sq = target + (board->turn == GPU_WHITE ? 16 : -16);
        moves[(*offset)++] = gpu_mv_from_data(sq, target, GPU_MV_DOUBLE_PAWN_PUSH);
    }
}

__device__ static inline void gpu_append_left_attacks(
        gpu_move_t *__restrict__ moves, uint32_t *__restrict__ offset,
        gpu_board_t *__restrict__ board, uint64_t left_attacks)
{
    uint8_t target;
    uint8_t sq;

    while (left_attacks != 0) {
        target = gpu_pop_rbit(&left_attacks);
        sq = target + (board->turn == GPU_WHITE ? 9 : -9);
        moves[(*offset)++] = gpu_mv_from_data(sq, target, GPU_MV_CAPTURE);
    }
}

__device__ static inline void gpu_append_right_attacks(
        gpu_move_t *__restrict__ moves, uint32_t *__restrict__ offset,
        gpu_board_t *__restrict__ board, uint64_t right_attacks)
{
    uint8_t target;
    uint8_t sq;

    while (right_attacks != 0) {
        target = gpu_pop_rbit(&right_attacks);
        sq = target + (board->turn == GPU_WHITE ? 7 : -7);
        moves[(*offset)++] = gpu_mv_from_data(sq, target, GPU_MV_CAPTURE);
    }
}

__device__ static inline void gpu_append_left_promos(
        gpu_move_t *__restrict__ moves, uint32_t *__restrict__ offset,
        gpu_board_t *__restrict__ board, uint64_t left_promos)
{
    uint8_t target;
    uint8_t sq;

    while (left_promos != 0) {
        target = gpu_pop_rbit(&left_promos);
        sq = target + (board->turn == GPU_WHITE ? 9 : -9);
        moves[(*offset)++] = gpu_mv_from_data(sq, target, GPU_MV_KNIGHT_PROMO_CAPTURE);
        moves[(*offset)++] = gpu_mv_from_data(sq, target, GPU_MV_BISHOP_PROMO_CAPTURE);
        moves[(*offset)++] = gpu_mv_from_data(sq, target, GPU_MV_ROOK_PROMO_CAPTURE);
        moves[(*offset)++] = gpu_mv_from_data(sq, target, GPU_MV_QUEEN_PROMO_CAPTURE);
    }
}

__device__ static inline void gpu_append_forward_promos(
        gpu_move_t *__restrict__ moves, uint32_t *__restrict__ offset,
        gpu_board_t *__restrict__ board, uint64_t forward_promos)
{
    uint8_t target;
    uint8_t sq;

    while (forward_promos != 0) {
        target = gpu_pop_rbit(&forward_promos);
        sq = target + (board->turn == GPU_WHITE ? 8 : -8);
        moves[(*offset)++] = gpu_mv_from_data(sq, target, GPU_MV_KNIGHT_PROMO);
        moves[(*offset)++] = gpu_mv_from_data(sq, target, GPU_MV_BISHOP_PROMO);
        moves[(*offset)++] = gpu_mv_from_data(sq, target, GPU_MV_ROOK_PROMO);
        moves[(*offset)++] = gpu_mv_from_data(sq, target, GPU_MV_QUEEN_PROMO);
    }
}

__device__ static inline void gpu_append_right_promos(
        gpu_move_t *__restrict__ moves, uint32_t *__restrict__ offset,
        gpu_board_t *__restrict__ board, uint64_t right_promos)
{
    uint8_t target;
    uint8_t sq;

    while (right_promos != 0) {
        target = gpu_pop_rbit(&right_promos);
        sq = target + (board->turn == GPU_WHITE ? 7 : -7);
        moves[(*offset)++] = gpu_mv_from_data(sq, target, GPU_MV_KNIGHT_PROMO_CAPTURE);
        moves[(*offset)++] = gpu_mv_from_data(sq, target, GPU_MV_BISHOP_PROMO_CAPTURE);
        moves[(*offset)++] = gpu_mv_from_data(sq, target, GPU_MV_ROOK_PROMO_CAPTURE);
        moves[(*offset)++] = gpu_mv_from_data(sq, target, GPU_MV_QUEEN_PROMO_CAPTURE);
    }
}

__device__ static inline void gpu_append_pawn_moves(
        gpu_move_t *__restrict__ moves, uint32_t *__restrict__ offset,
        gpu_board_t *__restrict__ board, gpu_state_tables_t *__restrict__ state)
{
    /* Get the mask of pawns that we want to evaluate. */
    uint64_t pawns = board->bb.piece[board->turn][GPU_PTYPE_PAWN];

    /* TODO: FIX ME PINS. */

    /* Remove all of the pinned pawns and add back those that lie on a left ray. */
    uint64_t left_pin_mask = state->pins[GPU_DIR_DR] | state->pins[GPU_DIR_UL];
    uint64_t left_pawns = (pawns & ~state->pins[8]) | (pawns & left_pin_mask);

    /* Remove all of the pinned pawns and add back those that lie on a forward ray. */
    uint64_t forward_pin_mask = state->pins[CB_DIR_D] | state->pins[CB_DIR_U];
    uint64_t forward_pawns = (pawns & ~state->pins[8]) | (pawns & forward_pin_mask);

    /* Remove all of the pinned pawns and add back those that lie on a right ray. */
    uint64_t right_pin_mask = state->pins[CB_DIR_DL] | state->pins[CB_DIR_UR];
    uint64_t right_pawns = (pawns & ~state->pins[8]) | (pawns & right_pin_mask);

    /* Generate masks for pawns moving left and right. */
    uint64_t left_smear = gpu_pawn_smear_left(left_pawns, board->turn);
    uint64_t left_attacks = left_smear & board->bb.color[!board->turn];
    uint64_t right_smear = gpu_pawn_smear_right(right_pawns, board->turn);
    uint64_t right_attacks = right_smear & board->bb.color[!board->turn];

    /* Generate masks for pushing pawns. */
    uint64_t forward_smear = gpu_pawn_smear_forward(forward_pawns, board->turn);
    uint64_t forward_moves = forward_smear & ~board->bb.occ;

    /* Smear the forward moves again to get the double pushes. */
    uint64_t double_smear = gpu_pawn_smear_forward(forward_moves, board->turn);
    uint64_t double_moves = double_smear & ~board->bb.occ;
    double_moves &= board->turn == GPU_WHITE ? BB_WHITE_PAWN_LINE : BB_BLACK_PAWN_LINE;

    /* Adjust for checks. */
    left_attacks &= state->check_blocks;
    right_attacks &= state->check_blocks;
    forward_moves &= state->check_blocks;
    double_moves &= state->check_blocks;

    /* Select the moves that cuase a promotion. */
    uint64_t left_promos = left_attacks & (BB_TOP_ROW | BB_BOTTOM_ROW);
    left_attacks ^= left_promos;
    uint64_t right_promos = right_attacks & (BB_TOP_ROW | BB_BOTTOM_ROW);
    right_attacks ^= right_promos;
    uint64_t forward_promos = forward_moves & (BB_TOP_ROW | BB_BOTTOM_ROW);
    forward_moves ^= forward_promos;

    /* Turn the masks into moves. */
    gpu_append_pushes(moves, offset, board, forward_moves);
    gpu_append_doubles(moves, offset, board, double_moves);
    gpu_append_left_attacks(moves, offset, board, left_attacks);
    gpu_append_right_attacks(moves, offset, board, right_attacks);
    gpu_append_forward_promos(moves, offset, board, forward_promos);
    gpu_append_left_promos(moves, offset, board, left_promos);
    gpu_append_right_promos(moves, offset, board, right_promos);
}

/* TODO: Simple moves cover pretty much everything. */
__device__ void append_simple_moves(
        gpu_mvlst_t *__restrict__ mvlst, gpu_board_t *__restrict__ board,
        gpu_state_tables_t *__restrict__ state)
{
    uint8_t sq, target;
    gpu_mv_flag_t flags;
    uint64_t mvmsk;
    uint64_t pieces = board->bb.color[board->turn];

    /* Generate pawn moves (and adjust for pins). */

    /* Generate knight moves (and adjust for pins). */

    /* Generate rook moves. */

    /* Generate bishop moves. */

    /* Generate queen moves. */

    /* Generate king moves. */

}

__device__ static inline bool ksc_legal(gpu_board_t *__restrict__ board,
        gpu_state_tables_t *__restrict__ state)
{
    cb_history_t hist = board->hist.data[board->hist.count - 1].hist;
    uint64_t occ_mask = board->turn == CB_WHITE ? BB_WHITE_KING_SIDE_CASTLE_OCCUPANCY :
        BB_BLACK_KING_SIDE_CASTLE_OCCUPANCY;
    uint64_t check_mask = board->turn == CB_WHITE ? BB_WHITE_KING_SIDE_CASTLE_CHECK :
        BB_BLACK_KING_SIDE_CASTLE_CHECK;

    /* If the occupancy intersects occ_mask or the threats intersect ckeck_mask. No castling. */
    return ((board->bb.occ & occ_mask) | (state->threats & check_mask)) == 0
        && cb_hist_has_ksc(hist, board->turn);
}

/* TODO: I got this one, it's about castling. */
__device__ static inline bool qsc_legal(gpu_board_t *__restrict__ board,
        gpu_state_tables_t *__restrict__ state)
{
    cb_history_t hist = board->hist.data[board->hist.count - 1].hist;
    uint64_t occ_mask = board->turn == CB_WHITE ? BB_WHITE_QUEEN_SIDE_CASTLE_OCCUPANCY :
        BB_BLACK_QUEEN_SIDE_CASTLE_OCCUPANCY;
    uint64_t check_mask = board->turn == CB_WHITE ? BB_WHITE_QUEEN_SIDE_CASTLE_CHECK :
        BB_BLACK_QUEEN_SIDE_CASTLE_CHECK;

    /* If the occupancy intersects occ_mask or the threats intersect ckeck_mask. No castling. */
    return ((board->bb.occ & occ_mask) | (state->threats & check_mask)) == 0
        && cb_hist_has_qsc(hist, board->turn);
}

/* TODO: I got this one, it's about castling. */
__device__ static inline void append_castle_moves(
        gpu_move_t *__restrict__ moves, uint32_t *__restrict__ offset,
        gpu_board_t *__restrict__ board, gpu_state_tables_t *__restrict__ state)
{
    uint8_t from = board->turn == CB_WHITE ? M_WHITE_KING_START : M_BLACK_KING_START;
    uint8_t to;

    if (ksc_legal(board, state)) {
        to = board->turn == CB_WHITE ? M_WHITE_KING_SIDE_CASTLE_TARGET :
            M_BLACK_KING_SIDE_CASTLE_TARGET;
        cb_mvlst_push(mvlst, cb_mv_from_data(from, to, CB_MV_KING_SIDE_CASTLE));
    }

    if (qsc_legal(board, state)) {
        to = board->turn == CB_WHITE ? M_WHITE_QUEEN_SIDE_CASTLE_TARGET :
            M_BLACK_QUEEN_SIDE_CASTLE_TARGET;
        cb_mvlst_push(mvlst, cb_mv_from_data(from, to, CB_MV_QUEEN_SIDE_CASTLE));
    }
}

/* TODO: I got this one, it's about enpassant. */
__device__ void append_enp_moves(
        gpu_move_t *__restrict__ moves, uint32_t *__restrict__ offset,
        gpu_board_t *__restrict__ board, gpu_state_tables_t *__restrict__ state)
{
    /* Exit early if there is not availiable enpassant. */
    if (!cb_hist_enp_availiable(board->hist.data[board->hist.count - 1].hist))
        return;

    /* Get the swares relavent to the piece that can enpassant. */
    gpu_history_t hist = board->hist.data[board->hist.count - 1].hist;
    uint8_t enp_row_start = board->turn == CB_WHITE ? M_BLACK_MIN_ENPASSANT_TARGET :
        M_WHITE_MIN_ENPASSANT_TARGET;
    uint8_t enp_sq = enp_row_start + cb_hist_enp_col(hist);
    uint8_t enemy_sq = enp_sq + (board->turn == CB_WHITE ? 8 : -8);

    /* Get all of the pieces that can enpassnt. */
    uint64_t enp_sources = cb_read_pawn_atk_msk(enp_sq, !board->turn)
        & board->bb.piece[board->turn][CB_PTYPE_PAWN];

    /* Loop through the pieces that can enpassant and generate the moves. */
    uint8_t sq, king_sq;
    gpu_move_t mv;
    uint64_t new_occ, bishop_threats, rook_threats;
    while (enp_sources) {
        sq = pop_rbit(&enp_sources);
        mv = cb_mv_from_data(sq, enp_sq, CB_MV_ENPASSANT);

        /* Update the occupancy mask to what it will be after the move takes place. */
        new_occ = board->bb.occ;
        new_occ &= ~(UINT64_C(1) << sq);
        new_occ &= ~(UINT64_C(1) << enemy_sq);
        new_occ |= UINT64_C(1) << enp_sq;

        /* Check if the king is in check after the move is made.
         * This could be the case if some piece was pinned before the enpassant was made. */
        king_sq = gpu_peek_rbit(board->bb.piece[board->turn][CB_PTYPE_KING]);

        bishop_threats = gpu_read_bishop_atk_msk(king_sq, new_occ)
            & (board->bb.piece[!board->turn][CB_PTYPE_BISHOP]
                | board->bb.piece[!board->turn][CB_PTYPE_QUEEN]);
        if (bishop_threats) continue;

        rook_threats = gpu_read_rook_atk_msk(king_sq, new_occ)
            & (board->bb.piece[!board->turn][CB_PTYPE_ROOK]
                | board->bb.piece[!board->turn][CB_PTYPE_QUEEN]);
        if (rook_threats) continue;

        /* Push the move if it doesn't cause any problems. */
        cb_mvlst_push(mvlst, mv);
    }
}

/* TODO: This one is pretty simple to change. */
__device__ void cb_gen_moves(
        gpu_move_t *__restrict__ moves, uint32_t *__restrict__ offset,
        gpu_board_t *__restrict__ board, gpu_state_tables_t *__restrict__ state)
{
    gpu_mvlst_clear(moves);
    gpu_append_pawn_moves(moves, board, state);
    gpu_append_simple_moves(moves, board, state);
    gpu_append_castle_moves(moves, board, state);
    gpu_append_enp_moves(moves, board, state);
}

/* TODO: This one should be a straightforward translation. */
__device__ static inline uint64_t gpu_gen_threats(
        gpu_board_t *__restrict__ board)
{
    uint64_t threats;
    uint8_t sq;
    uint64_t pawns = board->bb.piece[!board->turn][CB_PTYPE_PAWN];
    uint64_t king = board->bb.piece[board->turn][CB_PTYPE_KING];

    /* TODO: Need to reparadigm this. Loop over piece bitmasks. */

    /* Generate all of the threats. */
    uint64_t pieces = board->bb.color[!board->turn] ^ pawns;
    uint64_t occ = board->bb.occ ^ king; /* Remove the king to allow pieces to "see through" it. */
    cb_ptype_t ptype;
    cb_color_t pcolor;

    /* Generate all threats. */
    threats = gpu_pawn_smear(pawns, !board->turn);
    while (pieces) {
        sq = gpu_pop_rbit(&pieces);
        ptype = gpu_ptype_at_sq(board, sq);
        pcolor = gpu_color_at_sq(board, sq);
// gen_pseudo_mv_mask is not allowed because it causes divergence.
// Need to be generating moves for the same pieces across all boards.
//        threats |= gen_pseudo_mv_mask(ptype, pcolor, sq, occ);
    }

    return threats;
}

/* TODO: This one should be a straightforward translation. */
__device__ static inline uint64_t gpu_gen_checks(
        gpu_board_t *__restrict__ board, uint64_t threats)
{
    uint64_t *pieces = board->bb.piece[!board->turn];
    uint64_t king = board->bb.piece[board->turn][CB_PTYPE_KING];
    uint64_t occ = board->bb.occ;

    /* Exit early if the king isn't threatened. */
    if ((king & threats) == 0)
        return 0;

    /* Build the list of pieces that check the king. */
    uint64_t king_sq = peek_rbit(king);
    uint64_t checks = gpu_read_pawn_atk_msk(king_sq, board->turn) & pieces[CB_PTYPE_PAWN];
    checks |= gpu_read_knight_atk_msk(king_sq) & pieces[CB_PTYPE_KNIGHT];
    checks |= gpu_read_bishop_atk_msk(king_sq, occ)
        & (pieces[GPU_PTYPE_BISHOP] | pieces[GPU_PTYPE_QUEEN]);
    checks |= gpu_read_rook_atk_msk(king_sq, occ)
        & (pieces[GPU_PTYPE_ROOK] | pieces[GPU_PTYPE_QUEEN]);
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

    uint8_t king_sq = gpu_peek_rbit(board->bb.piece[board->turn][CB_PTYPE_KING]);
    uint8_t check_sq = gpu_peek_rbit(checks);
    return gpu_read_tf_table(check_sq, king_sq) | (UINT64_C(1) << check_sq);
}

/* TODO: This one should be a straightforward translation.
 *
 * I'd recommend looking this one up on Chess Programming Wiki.
 */
__device__ static inline uint64_t gpu_xray_bishop_attacks(
        uint64_t occ, uint64_t blockers, uint64_t sq)
{
    uint64_t attacks = cb_read_bishop_atk_msk(sq, occ);
    blockers &= attacks;
    return attacks ^ cb_read_bishop_atk_msk(sq, occ ^ blockers);
}

/* TODO: This one should be a straightforward translation.
 *
 * I'd recommend looking this one up on Chess Programming Wiki.
 */
__device__ static inline uint64_t gpu_xray_rook_attacks(
        uint64_t occ, uint64_t blockers, uint64_t sq)
{
    uint64_t attacks = cb_read_rook_atk_msk(sq, occ);
    blockers &= attacks;
    return attacks ^ cb_read_rook_atk_msk(sq, occ ^ blockers);
}

/* TODO: This fucntion needs serious adjustment. */
__device__ static inline void gpu_gen_pins(
        uint64_t pins[10], gpu_board_t *restrict board)
{
    uint64_t king = board->bb.piece[board->turn][CB_PTYPE_KING];
    uint64_t king_sq = peek_rbit(king);
    uint64_t occ = board->bb.occ;
    uint64_t blockers = board->bb.color[board->turn];
    uint64_t pinner;
    uint8_t sq, dir;

    /* Set all of the pins to full bitboards. */
    memset(pins, 0, 10 * sizeof(uint64_t));

    /* Get all of the first pinners. */
    pinner = gpu_xray_bishop_attacks(occ, blockers, king_sq)
        & (board->bb.piece[!board->turn][CB_PTYPE_BISHOP]
        | board->bb.piece[!board->turn][CB_PTYPE_QUEEN]);
    while (pinner) {
        sq = gpu_pop_rbit(&pinner);
        dir = gpu_get_ray_direction(king_sq, sq);

        /* NOTE: This is the dumb thing. I read the tf_table (to-from table)
         * right away to get the ray that the pinned piece lies on. If
         * we delay this computation, we don't need to use nearly as many
         * registers.
         */
        pins[dir] = gpu_read_tf_table(sq, king_sq);
        pins[8] ^= pins[dir];
    }

    /* Get all of the second pinners. */
    pinner = gpu_xray_rook_attacks(occ, blockers, king_sq)
        & (board->bb.piece[!board->turn][CB_PTYPE_ROOK]
        | board->bb.piece[!board->turn][CB_PTYPE_QUEEN]);
    while (pinner) {
        sq = gpu_pop_rbit(&pinner);
        dir = gpu_get_ray_direction(king_sq, sq);

        /* NOTE: Same thing here. */
        pins[dir] = gpu_read_tf_table(sq, king_sq);
        pins[8] ^= pins[dir];
    }
}

/* TODO: Move Generation Task.
 *
 * These functions will need to be adjusted for the GPU.
 * In particular, I think we will need to take a look at the way I handle
 * pins. It is kindof stupid. See notes above and in gpu_types.h.
 */
__device__ void gpu_gen_board_tables(gpu_state_tables_t *__restrict__ state,
        gpu_board_t *__restrict__ board)
{
    state->threats = gpu_gen_threats(board);
    state->checks = gpu_gen_checks(board, state->threats);
    state->check_blocks = gpu_gen_check_blocks(board, state->checks);
    gpu_gen_pins(state->pins, board);
}

