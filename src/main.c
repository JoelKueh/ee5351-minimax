
#include "crosstime.h"
#include "cblib_gpu/gpu_lib.h"
#include <cb_lib.h>
#include <cb_move.h>
#include <inttypes.h>

/* TODO: We will have to implement the switch over to the GPU somewhere
 * in this function.
 *
 * If you don't understand, lookup the perft routines on chess programming
 * wiki. Other than that, I think the host code should work right away.
 * We just need to do something different when we reach the desired depth.
 */

int main()
{
    return 0;
}

/* TODO: To be clear, almost all of the necessary changes can be made
 * in this function. */
uint64_t perfting(cb_board_t *board, cb_state_tables_t *state, int depth)
{
    /* TODO: Remove this if directive. It's just to allow stuff to compile. */
    uint64_t cnt = 0;
    int i;
    cb_move_t mv;
    cb_mvlst_t mvlst;

    /* Base case. */
    if (depth <= 0)
        return 1;

    /* Generate the moves. */
    cb_gen_board_tables(state, board);
    cb_gen_moves(&mvlst, board, state);

    /* Make moves and move down the tree. */
    for (i = 0; i < cb_mvlst_size(&mvlst); i++) {
        mv = cb_mvlst_at(&mvlst, i);
        /* This function can fail, but only when a reservation is needed.
         * As perft does a manual reservation, there is no need to reserve here and no error. */
        cb_make(board, mv);

        /*cb_print_bitboard(stdout, board);
        cb_print_mv_hist(stdout, board);*/

        cnt += perfting(board, state, depth - 1);
        cb_unmake(board);
    }

    return cnt;
}

int perft(gpu_board_t *board, int depth)
{
    cb_errno_t result;
    cb_error_t err;
    cb_mvlst_t mvlst;
    cb_move_t mv;
    uint64_t perft_results[CB_MAX_NUM_MOVES];
    cb_state_tables_t state;
    uint64_t cnt = 0;
    uint64_t total = 0;
    char buf[6];
    int i;

    uint64_t start_time;
    uint64_t end_time;

    /* Exit early if depth is less than 1. */
    if (depth < 1) {
        printf("Perft with a depth below 1 doesn't do anything silly :-)\n");
        return 0;
    }

    /* Reserve the board history. This line guarantees that make will never write
     * past its proper bounds. */
    if ((result = cb_reserve_for_make(&err, board, depth)) != 0) {
        fprintf(stderr, "cb_reserve_for_make: %s\n", err.desc);
        return result;
    }

    /* Loop through all of the first levels and calculate the number of moves. */
    start_time = time_ns();
    cb_gen_board_tables(&state, board);
    cb_gen_moves(&mvlst, board, &state);
    for (i = 0; i < cb_mvlst_size(&mvlst); i++) {
        mv = cb_mvlst_at(&mvlst, i);
        cb_make(board, mv);
        cnt = perfting(board, &state, depth - 1);
        total += cnt;
        cb_mv_to_uci_algbr(buf, mv);
        printf("%s: %" PRIu64 "\n", buf, cnt);
        cb_unmake(board);
    }
    end_time = time_ns();
    printf("\n");
    printf("Nodes searched: %" PRIu64 "\n", total);
    printf("Time: %.3fms\n", (end_time - start_time) / 1000000.0);
    printf("\n");

    return 0;
}

