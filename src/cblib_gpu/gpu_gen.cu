
#include <string.h>

#include "gpu_types.cuh"
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
    return color == CB_WHITE ?
        (pawns >> 9 & ~BB_RIGHT_COL) | (pawns >> 7 & ~BB_LEFT_COL) :
        (pawns << 7 & ~BB_RIGHT_COL) | (pawns << 9 & ~BB_LEFT_COL);
}

__device__ static inline uint64_t gpu_pawn_smear_left(
        uint64_t pawns, gpu_color_t color)
{
    return color == CB_WHITE ?
        (pawns >> 9 & ~BB_RIGHT_COL) :
        (pawns << 9 & ~BB_LEFT_COL);
}

__device__ static inline uint64_t gpu_pawn_smear_forward(
        uint64_t pawns, gpu_color_t color)
{
    return color == CB_WHITE ? pawns >> 8 : pawns << 8;
}

__device__ static inline uint64_t gpu_pawn_smear_right(
        uint64_t pawns, gpu_color_t color)
{
    return color == CB_WHITE ?
        (pawns >> 7 & ~BB_LEFT_COL) :
        (pawns << 7 & ~BB_RIGHT_COL);
}

__device__ static inline void gpu_append_pushes(
		gpu_mvlst_t *restrict moves,
		uint32_t *restrict offset,
		uint64_t pushes, gpu_color_t color)
{
    uint8_t target;
    uint8_t sq;

    while (pushes != 0) {
        target = pop_rbit(&pushes);
        sq = target + (color == CB_WHITE ? 8 : -8);
	    moves[(*offset)++] = cb_mv_from_data(sq, target, CB_MV_QUIET);
        mvlst, cb_mv_from_data(sq, target, CB_MV_QUIET));
    }
}

__device__ static inline void append_doubles(
        gpu_mvlst_t *restrict mvlst, gpu_board_t *restrict board,
        uint64_t doubles)
{
    uint8_t target;
    uint8_t sq;

    while (doubles != 0) {
        target = pop_rbit(&doubles);
        sq = target + (board->turn == CB_WHITE ? 16 : -16);
        cb_mvlst_push(mvlst, cb_mv_from_data(sq, target, CB_MV_DOUBLE_PAWN_PUSH));
    }
}

__device__ static inline void append_left_attacks(
        gpu_mvlst_t *restrict mvlst, gpu_board_t *restrict board,
        uint64_t left_attacks)
{
    uint8_t target;
    uint8_t sq;

    while (left_attacks != 0) {
        target = pop_rbit(&left_attacks);
        sq = target + (board->turn == CB_WHITE ? 9 : -9);
        cb_mvlst_push(mvlst, cb_mv_from_data(sq, target, CB_MV_CAPTURE));
    }
}

__device__ static inline void append_right_attacks(
        gpu_mvlst_t *restrict mvlst, gpu_board_t *restrict board,
        uint64_t right_attacks)
{
    uint8_t target;
    uint8_t sq;

    while (right_attacks != 0) {
        target = pop_rbit(&right_attacks);
        sq = target + (board->turn == CB_WHITE ? 7 : -7);
        cb_mvlst_push(mvlst, cb_mv_from_data(sq, target, CB_MV_CAPTURE));
    }
}

__device__ static inline void append_left_promos(
        gpu_mvlst_t *restrict mvlst, gpu_board_t *restrict board,
        uint64_t left_promos)
{
    uint8_t target;
    uint8_t sq;

    while (left_promos != 0) {
        target = pop_rbit(&left_promos);
        sq = target + (board->turn == CB_WHITE ? 9 : -9);
        cb_mvlst_push(mvlst, cb_mv_from_data(sq, target, CB_MV_KNIGHT_PROMO_CAPTURE));
        cb_mvlst_push(mvlst, cb_mv_from_data(sq, target, CB_MV_BISHOP_PROMO_CAPTURE));
        cb_mvlst_push(mvlst, cb_mv_from_data(sq, target, CB_MV_ROOK_PROMO_CAPTURE));
        cb_mvlst_push(mvlst, cb_mv_from_data(sq, target, CB_MV_QUEEN_PROMO_CAPTURE));
    }
}

__device__ static inline void append_forward_promos(
        gpu_mvlst_t *restrict mvlst, gpu_board_t *restrict board,
        uint64_t forward_promos)
{
    uint8_t target;
    uint8_t sq;

    while (forward_promos != 0) {
        target = pop_rbit(&forward_promos);
        sq = target + (board->turn == CB_WHITE ? 8 : -8);
        cb_mvlst_push(mvlst, cb_mv_from_data(sq, target, CB_MV_KNIGHT_PROMO));
        cb_mvlst_push(mvlst, cb_mv_from_data(sq, target, CB_MV_BISHOP_PROMO));
        cb_mvlst_push(mvlst, cb_mv_from_data(sq, target, CB_MV_ROOK_PROMO));
        cb_mvlst_push(mvlst, cb_mv_from_data(sq, target, CB_MV_QUEEN_PROMO));
    }
}

__device__ static inline void append_right_promos(
        gpu_mvlst_t *restrict mvlst, gpu_board_t *restrict board,
        uint64_t right_promos)
{
    uint8_t target;
    uint8_t sq;

    while (right_promos != 0) {
        target = pop_rbit(&right_promos);
        sq = target + (board->turn == CB_WHITE ? 7 : -7);
        cb_mvlst_push(mvlst, cb_mv_from_data(sq, target, CB_MV_KNIGHT_PROMO_CAPTURE));
        cb_mvlst_push(mvlst, cb_mv_from_data(sq, target, CB_MV_BISHOP_PROMO_CAPTURE));
        cb_mvlst_push(mvlst, cb_mv_from_data(sq, target, CB_MV_ROOK_PROMO_CAPTURE));
        cb_mvlst_push(mvlst, cb_mv_from_data(sq, target, CB_MV_QUEEN_PROMO_CAPTURE));
    }
}

__device__ static inline void append_pawn_moves(
        gpu_mvlst_t *restrict mvlst, gpu_board_t *restrict board,
        gpu_state_tables_t *restrict state)
{
    /* Get the mask of pawns that we want to evaluate. */
    uint64_t pawns = board->bb.piece[board->turn][CB_PTYPE_PAWN];

    /* Remove all of the pinned pawns and add back those that lie on a left ray. */
    uint64_t left_pin_mask = state->pins[CB_DIR_DR] | state->pins[CB_DIR_UL];
    uint64_t left_pawns = (pawns & ~state->pins[8]) | (pawns & left_pin_mask);

    /* Remove all of the pinned pawns and add back those that lie on a forward ray. */
    uint64_t forward_pin_mask = state->pins[CB_DIR_D] | state->pins[CB_DIR_U];
    uint64_t forward_pawns = (pawns & ~state->pins[8]) | (pawns & forward_pin_mask);

    /* Remove all of the pinned pawns and add back those that lie on a right ray. */
    uint64_t right_pin_mask = state->pins[CB_DIR_DL] | state->pins[CB_DIR_UR];
    uint64_t right_pawns = (pawns & ~state->pins[8]) | (pawns & right_pin_mask);

    /* Generate masks for pawns moving left and right. */
    uint64_t left_smear = pawn_smear_left(left_pawns, board->turn);
    uint64_t left_attacks = left_smear & board->bb.color[!board->turn];
    uint64_t right_smear = pawn_smear_right(right_pawns, board->turn);
    uint64_t right_attacks = right_smear & board->bb.color[!board->turn];

    /* Generate masks for pushing pawns. */
    uint64_t forward_smear = pawn_smear_forward(forward_pawns, board->turn);
    uint64_t forward_moves = forward_smear & ~board->bb.occ;

    /* Smear the forward moves again to get the double pushes. */
    uint64_t double_smear = pawn_smear_forward(forward_moves, board->turn);
    uint64_t double_moves = double_smear & ~board->bb.occ;
    double_moves &= board->turn == CB_WHITE ? BB_WHITE_PAWN_LINE : BB_BLACK_PAWN_LINE;

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
    append_pushes(mvlst, board, forward_moves);
    append_doubles(mvlst, board, double_moves);
    append_left_attacks(mvlst, board, left_attacks);
    append_right_attacks(mvlst, board, right_attacks);
    append_forward_promos(mvlst, board, forward_promos);
    append_left_promos(mvlst, board, left_promos);
    append_right_promos(mvlst, board, right_promos);
}

__device__ uint64_t gen_pseudo_mv_mask(
        cb_ptype_t ptype, cb_color_t pcolor, uint8_t sq, uint64_t occ)
{
    switch (ptype) {
        case CB_PTYPE_PAWN:
            return cb_read_pawn_atk_msk(sq, pcolor);
        case CB_PTYPE_KNIGHT:
            return cb_read_knight_atk_msk(sq);
        case CB_PTYPE_BISHOP:
            return cb_read_bishop_atk_msk(sq, occ);
        case CB_PTYPE_ROOK:
            return cb_read_rook_atk_msk(sq, occ);
        case CB_PTYPE_QUEEN:
            return cb_read_bishop_atk_msk(sq, occ)
                | cb_read_rook_atk_msk(sq, occ);
        case CB_PTYPE_KING:
            return cb_read_king_atk_msk(sq);
        case CB_PTYPE_EMPTY:
            assert(false && "invalid piece type for pseudo legal move generation");
            return 0;
    }
}

/* TODO: Move generation work really begins here.
 *
 * Pin adjustment needs a serious rework to save on register space, good luck.
 */
__device__ static inline uint64_t pin_adjust(
        gpu_board_t *restrict board, gpu_state_tables_t *restrict state,
        uint8_t sq, uint64_t moves)
{
    uint64_t mask;
    uint8_t king_sq = peek_rbit(board->bb.piece[board->turn][CB_PTYPE_KING]);
    uint8_t dir = cb_get_ray_direction(king_sq, sq);
    return (state->pins[dir] & (UINT64_C(1) << sq)) == 0 ? moves : (moves & state->pins[dir]);
}

/* TODO: This should look mostly the same as the CPU version. */
__device__ uint64_t cb_gen_legal_mv_mask(
        gpu_board_t *restrict board, gpu_state_tables_t *restrict state,
        uint8_t sq)
{
    /* Generate the pseudo moves. */
    cb_ptype_t ptype = cb_ptype_at_sq(board, sq);
    cb_color_t pcolor = cb_color_at_sq(board, sq);
    uint64_t moves = gen_pseudo_mv_mask(ptype, pcolor, sq, board->bb.occ);
    moves &= ~board->bb.color[board->turn];

    /* Adjust moves for pins and checks. */
    moves &= ptype == CB_PTYPE_KING ? ~state->threats : state->check_blocks;
    moves = pin_adjust(board, state, sq, moves);

    return moves;
}

/* TODO: Simple moves cover pretty much everything. */
__device__ void append_simple_moves(
        gpu_mvlst_t *restrict mvlst, gpu_board_t *restrict board,
        gpu_state_tables_t *restrict state)
{
    uint8_t sq, target;
    cb_mv_flag_t flags;
    uint64_t mvmsk;
    uint64_t pieces = board->bb.color[board->turn];

    /* Append all of the moves to the list. */
    pieces ^= board->bb.piece[board->turn][CB_PTYPE_PAWN];
    while (pieces) {
        sq = pop_rbit(&pieces);
        mvmsk = cb_gen_legal_mv_mask(board, state, sq);
        while (mvmsk) {
            target = pop_rbit(&mvmsk);
            flags = (UINT64_C(1) << target) & board->bb.occ ? CB_MV_CAPTURE : CB_MV_QUIET;
            cb_mvlst_push(mvlst, cb_mv_from_data(sq, target, flags));
        }
    }
}

/* TODO: I got this one, it's about castling. */
__device__ static inline bool ksc_legal(
        gpu_board_t *restrict board, gpu_state_tables_t *restrict state)
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
__device__ static inline bool qsc_legal(
        gpu_board_t *restrict board, gpu_state_tables_t *restrict state)
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
__device__ void append_castle_moves(
        cb_mvlst_t *restrict mvlst, cb_board_t *restrict board,
        cb_state_tables_t *restrict state)
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
        gpu_mvlst_t *restrict mvlst, gpu_board_t *restrict board,
        gpu_state_tables_t *restrict state)
{
    /* Exit early if there is not availiable enpassant. */
    if (!cb_hist_enp_availiable(board->hist.data[board->hist.count - 1].hist))
        return;

    /* Get the swares relavent to the piece that can enpassant. */
    cb_history_t hist = board->hist.data[board->hist.count - 1].hist;
    uint8_t enp_row_start = board->turn == CB_WHITE ? M_BLACK_MIN_ENPASSANT_TARGET :
        M_WHITE_MIN_ENPASSANT_TARGET;
    uint8_t enp_sq = enp_row_start + cb_hist_enp_col(hist);
    uint8_t enemy_sq = enp_sq + (board->turn == CB_WHITE ? 8 : -8);

    /* Get all of the pieces that can enpassnt. */
    uint64_t enp_sources = cb_read_pawn_atk_msk(enp_sq, !board->turn)
        & board->bb.piece[board->turn][CB_PTYPE_PAWN];

    /* Loop through the pieces that can enpassant and generate the moves. */
    uint8_t sq, king_sq;
    cb_move_t mv;
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
        king_sq = peek_rbit(board->bb.piece[board->turn][CB_PTYPE_KING]);

        bishop_threats = cb_read_bishop_atk_msk(king_sq, new_occ)
            & (board->bb.piece[!board->turn][CB_PTYPE_BISHOP]
                | board->bb.piece[!board->turn][CB_PTYPE_QUEEN]);
        if (bishop_threats) continue;

        rook_threats = cb_read_rook_atk_msk(king_sq, new_occ)
            & (board->bb.piece[!board->turn][CB_PTYPE_ROOK]
                | board->bb.piece[!board->turn][CB_PTYPE_QUEEN]);
        if (rook_threats) continue;

        /* Push the move if it doesn't cause any problems. */
        cb_mvlst_push(mvlst, mv);
    }
}

/* TODO: This one is pretty simple to change. */
__device__ void cb_gen_moves(
        gpu_mvlst_t *restrict mvlst, gpu_board_t *restrict board,
        gpu_state_tables_t *restrict state)
{
    cb_mvlst_clear(mvlst);
    append_pawn_moves(mvlst, board, state);
    append_simple_moves(mvlst, board, state);
    append_castle_moves(mvlst, board, state);
    append_enp_moves(mvlst, board, state);
}

/* TODO: This one should be a straightforward translation. */
__device__ static inline uint64_t gen_threats(gpu_board_t *restrict board)
{
    uint64_t threats;
    uint8_t sq;
    uint64_t pawns = board->bb.piece[!board->turn][CB_PTYPE_PAWN];
    uint64_t king = board->bb.piece[board->turn][CB_PTYPE_KING];

    /* Generate all of the threats. */
    uint64_t pieces = board->bb.color[!board->turn] ^ pawns;
    uint64_t occ = board->bb.occ ^ king; /* Remove the king to allow pieces to "see through" it. */
    cb_ptype_t ptype;
    cb_color_t pcolor;

    /* Generate all threats. */
    threats = pawn_smear(pawns, !board->turn);
    while (pieces) {
        sq = pop_rbit(&pieces);
        ptype = cb_ptype_at_sq(board, sq);
        pcolor = cb_color_at_sq(board, sq);
        threats |= gen_pseudo_mv_mask(ptype, pcolor, sq, occ);
    }

    return threats;
}

/* TODO: This one should be a straightforward translation. */
__device__ static inline uint64_t gen_checks(
        gpu_board_t *restrict board, uint64_t threats)
{
    uint64_t *pieces = board->bb.piece[!board->turn];
    uint64_t king = board->bb.piece[board->turn][CB_PTYPE_KING];
    uint64_t occ = board->bb.occ;

    /* Exit early if the king isn't threatened. */
    if ((king & threats) == 0)
        return 0;

    /* Build the list of pieces that check the king. */
    uint64_t king_sq = peek_rbit(king);
    uint64_t checks = cb_read_pawn_atk_msk(king_sq, board->turn) & pieces[CB_PTYPE_PAWN];
    checks |= cb_read_knight_atk_msk(king_sq) & pieces[CB_PTYPE_KNIGHT];
    checks |= cb_read_bishop_atk_msk(king_sq, occ)
        & (pieces[CB_PTYPE_BISHOP] | pieces[CB_PTYPE_QUEEN]);
    checks |= cb_read_rook_atk_msk(king_sq, occ)
        & (pieces[CB_PTYPE_ROOK] | pieces[CB_PTYPE_QUEEN]);
    /* Here's a helpful reminder that a king can never check another king. */

    return checks;
}

/* TODO: This one should be a straightforward translation. */
__device__ static inline uint64_t gen_check_blocks(
        gpu_board_t *restrict board, uint64_t checks)
{
    if (checks == 0)
        return BB_FULL;
    else if (popcnt(checks) != 1)
        return BB_EMPTY;

    uint8_t king_sq = peek_rbit(board->bb.piece[board->turn][CB_PTYPE_KING]);
    uint8_t check_sq = peek_rbit(checks);
    return cb_read_tf_table(check_sq, king_sq) | (UINT64_C(1) << check_sq);
}

/* TODO: This one should be a straightforward translation.
 *
 * I'd recommend looking this one up on Chess Programming Wiki.
 */
__device__ static inline uint64_t xray_bishop_attacks(
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
__device__ static inline uint64_t xray_rook_attacks(
        uint64_t occ, uint64_t blockers, uint64_t sq)
{
    uint64_t attacks = cb_read_rook_atk_msk(sq, occ);
    blockers &= attacks;
    return attacks ^ cb_read_rook_atk_msk(sq, occ ^ blockers);
}

/* TODO: This fucntion needs serious adjustment. */
__device__ static inline void gen_pins(
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
    pinner = xray_bishop_attacks(occ, blockers, king_sq)
        & (board->bb.piece[!board->turn][CB_PTYPE_BISHOP]
        | board->bb.piece[!board->turn][CB_PTYPE_QUEEN]);
    while (pinner) {
        sq = pop_rbit(&pinner);
        dir = cb_get_ray_direction(king_sq, sq);

        /* NOTE: This is the dumb thing. I read the tf_table (to-from table)
         * right away to get the ray that the pinned piece lies on. If
         * we delay this computation, we don't need to use nearly as many
         * registers.
         */
        pins[dir] = cb_read_tf_table(sq, king_sq);
        pins[8] ^= pins[dir];
    }

    /* Get all of the second pinners. */
    pinner = xray_rook_attacks(occ, blockers, king_sq)
        & (board->bb.piece[!board->turn][CB_PTYPE_ROOK]
        | board->bb.piece[!board->turn][CB_PTYPE_QUEEN]);
    while (pinner) {
        sq = pop_rbit(&pinner);
        dir = cb_get_ray_direction(king_sq, sq);

        /* NOTE: Same thing here. */
        pins[dir] = cb_read_tf_table(sq, king_sq);
        pins[8] ^= pins[dir];
    }
}

/* TODO: Move Generation Task.
 *
 * These functions will need to be adjusted for the GPU.
 * In particular, I think we will need to take a look at the way I handle
 * pins. It is kindof stupid. See notes above and in gpu_types.h.
 */
__device__ void gpu_gen_board_tables(
        gpu_state_tables_t *restrict state, gpu_board_t *restrict board)
{
    state->threats = gen_threats(board);
    state->checks = gen_checks(board, state->threats);
    state->check_blocks = gen_check_blocks(board, state->checks);
    gen_pins(state->pins, board);
}

