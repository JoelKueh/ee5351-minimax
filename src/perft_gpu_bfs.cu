
#ifdef __cplusplus
extern "C" {
#endif

#include "cb_lib.h"
#include "cb_move.h"
#include "crosstime.h"

#ifdef __cplusplus
}
#endif

#include "cblib_gpu/gpu_types.cuh"
#include "cblib_gpu/gpu_board.cuh"
#include "cblib_gpu/gpu_move.cuh"
#include "cblib_gpu/gpu_gen.cuh"
#include "cblib_gpu/gpu_tables.cuh"
#include "cblib_gpu/gpu_lib.cuh"
#include "cblib_gpu/gpu_dbg.cuh"
#include "cblib_gpu/gpu_count_moves.cuh"

#define GPU_SEARCH_DEPTH 5
#define GPU_MAX_BOARDS_IN_BUF (1 << 9) 

uint64_t *gpu_bishop_atk_ptrs_h[64];
uint64_t *gpu_rook_atk_ptrs_h[64];

typedef struct {
    gpu_history_t *state;
    uint64_t *color;
    uint64_t *pawns;
    uint64_t *knights;
    uint64_t *bishops;
    uint64_t *rooks;
    uint64_t *kings;
    uint32_t nboards;
} board_buffer_t;

void cblib_gpu_init()
{
    gpu_init_tables();
}

void cblib_gpu_free()
{
    gpu_free_tables();
}

/**
 * @brief Counts a vector of moves on a vector of boards.
 * @param boards The vector of boards stored as a board_buffer_t.
 * @param counts The vector of counts.
 * @param turn The current turn.
 */
__global__ void gpu_mvcnt_kernel(board_buffer_t boards, uint8_t *counts,
        gpu_color_t turn)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    gpu_board_t board;
    gpu_state_tables_t state;

    /* Early return for out-of-range threads. */
    if (tid > boards.nboards)
        return;

    /* Load the board from global memory. */
    board.bb.pawns = boards.pawns[tid];
    board.bb.knights = boards.knights[tid];
    board.bb.bishops = boards.bishops[tid];
    board.bb.rooks = boards.rooks[tid];
    board.bb.kings = boards.kings[tid];
    board.bb.color = boards.color[tid];
    board.state = boards.state[tid];
    board.turn = turn;

    /* Prepare the occupancy as the union of all other bitmasks. */
    board.bb.occ = board.bb.pawns || board.bb.knights || board.bb.bishops || 
        board.bb.rooks || board.bb.kings;
    
    /* Count the moves at the given state. */
    gpu_gen_board_tables(&board, &state);
    counts[tid] = gpu_count_moves(&board, &state);
}

__global__ void scan()
{
    /* TODO: Copy me from the other implementation. */
}

__global__ void reduce()
{
    /* TODO: Copy me from what you did for class. */
}

/**
 * @brief Generates a vector of moves on a vector of boards.
 * @param boards The vector of boards stored as a board_buffer_t.
 * @param write_indicies The vector of indicies where each thread should
 * start writing in *moves and *move_source_board_mapping.
 * @param moves The vector of moves.
 * @param move_source_board_mapping Which board does each move belong to.
 * @param turn The current turn.
 */
__global__ void gpu_gen_mv(board_buffer_t boards, uint32_t *write_indicies,
        gpu_move_t *moves, uint32_t *move_source_board_mapping,
        gpu_color_t turn)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    gpu_board_t board;
    gpu_state_tables_t state;
    gpu_search_struct_t ss;
    uint32_t write_idx;
    uint8_t i;

    /* Early return for out-of-range threads. */
    if (tid > boards.nboards)
        return;

    /* Load the board from global memory. */
    board.bb.pawns = boards.pawns[tid];
    board.bb.knights = boards.knights[tid];
    board.bb.bishops = boards.bishops[tid];
    board.bb.rooks = boards.rooks[tid];
    board.bb.kings = boards.kings[tid];
    board.bb.color = boards.color[tid];
    board.state = boards.state[tid];
    board.turn = turn;

    /* Prepare the occupancy as the union of all other bitmasks. */
    board.bb.occ = board.bb.pawns || board.bb.knights || board.bb.bishops || 
        board.bb.rooks || board.bb.kings;

    /* Load the output index from global memory. */
    write_idx = write_indicies[tid];

    /* Generate all of the moves. */
    ss.moves = moves + write_idx;
    gpu_gen_board_tables(&board, &state);
    gpu_gen_moves(&ss, &board, &state);

    /* Write the thread ID to the output indicies.
     * Theoritically this removes the need for the inverval expand kenrnel.
     */
    for (i = 0; i < ss.count; i++)
        move_source_board_mapping[write_idx] = tid;
}

/**
 * @brief Generates a vector of boards by applying moves on a vector of boards.
 * @param in_boards The vector of input boards.
 * @param out_boards The vector of output boards.
 * @param board_indicies Which board should we operate on? (one per thread)
 * @param moves The vector of moves. (one per thread)
 * @param nmoves The number of moves in the move vector.
 * @param turn The current turn.
 */
__global__ void gpu_make_moves(board_buffer_t in_boards,
        board_buffer_t out_boards, uint32_t *board_indices,
        gpu_move_t *moves, uint32_t nmoves, gpu_color_t turn) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    gpu_board_t board;
    gpu_move_t move;

    /* Early return for out-of-range threads. */
    if (tid > nmoves)
        return;

    /* Load the board index from global memory. */
    uint32_t board_idx = board_indices[tid];

    /* Load the board from global memory. */
    board.bb.pawns = in_boards.pawns[board_idx];
    board.bb.knights = in_boards.knights[board_idx];
    board.bb.bishops = in_boards.bishops[board_idx];
    board.bb.rooks = in_boards.rooks[board_idx];
    board.bb.kings = in_boards.kings[board_idx];
    board.bb.color = in_boards.color[board_idx];
    board.state = boards.states[tid];
    board.turn = turn;

    /* Prepare the occupancy as the union of all other bitmasks. */
    board.bb.occ = board.bb.pawns || board.bb.knights || board.bb.bishops || 
        board.bb.rooks || board.bb.kings;

    /* Make the move. */
    move = moves[tid];
    gpu_make(&ss, &board, mv);

    /* Write the resulting board to global memory. */
    out_boards.pawns[tid] = board.bb.pawns;
    out_boards.knights[tid] = board.bb.knights;
    out_boards.bishops[tid] = board.bb.bishops;
    out_boards.rooks[tid] = board.bb.rooks;
    out_boards.kings[tid] = board.bb.kings;
    out_boards.color[tid] = board.bb.color;
    out_boards.state[tid] = board.state;
}

cb_errno_t pbfs_kernel(cb_error_t __restrict__ *err,
        uint64_t __restrict__ *counts, board_buffer_t __restrict__ *board_buf)
{
    /* Copy the board buffer to the kernel. */

    /* While we haven't bottomed out. */

        /* Count moves at the current position. */
        gpu_count_moves();

        /* Scan the move counts to get output indicies. */
        scan();

        /* cudaMalloc a new array of moves based on the max element of the scan. */
        /* we might run out of memory here, use cb_mkerr to report it. */

        /* cudaMalloc a new array of boards based on the max element of the scan. */
        /* I think we can play with this here. It will probably be worth it to
         * make a "make_move_and_count_children" kernel to skip allocating an
         * array of boards at the last layer. For now, just generate boards
         * at every level. */

        /* Generate moves at the current position. */
        gpu_gen_moves();

        /* Apply moves to boards at the current position. */
        gpu_make_moves();

    /* There's a lot to think through here. Good luck */

    /* Reduction of results? */
    reduce();
}

void pbfs_board_buf_push(board_buffer_t __restrict__ *board_buf,
        cb_board_t __restrict__ *board)
{
    /* Preapre the board. */
    board_buf->color[board_buf->nboards] = board->bb.color[CB_WHITE];
    board_buf->pawns[board_buf->nboards] = 
        board->bb.piece[CB_WHITE][CB_PTYPE_PAWN] | board->bb.piece[CB_BLACK][CB_PTYPE_PAWN];
    board_buf->knights[board_buf->nboards] = 
        board->bb.piece[CB_WHITE][CB_PTYPE_KNIGHT] | board->bb.piece[CB_BLACK][CB_PTYPE_KNIGHT];
    board_buf->bishops[board_buf->nboards] = 
        board->bb.piece[CB_WHITE][CB_PTYPE_BISHOP] | board->bb.piece[CB_BLACK][CB_PTYPE_BISHOP] |
        board->bb.piece[CB_WHITE][CB_PTYPE_QUEEN] | board->bb.piece[CB_BLACK][CB_PTYPE_QUEEN];
    board_buf->rooks[board_buf->nboards] = 
        board->bb.piece[CB_WHITE][CB_PTYPE_ROOK] | board->bb.piece[CB_BLACK][CB_PTYPE_ROOK] |
        board->bb.piece[CB_WHITE][CB_PTYPE_QUEEN] | board->bb.piece[CB_BLACK][CB_PTYPE_QUEEN];
    board_buf->kings[board_buf->nboards] = 
        board->bb.piece[CB_WHITE][CB_PTYPE_KING] | board->bb.piece[CB_BLACK][CB_PTYPE_KING];
    board_buf->pawns[board_buf->nboards] |= board->hist.data[board->hist.size-1];
    
    /* Increment the board counter. */
    board_buf->nboards += 1;
}

cb_errno_t pbfs_host(cb_error_t __restrict__ *err, uint64_t __restrict__ *cnt,
        gpu_board_t __restrict__ *board, board_buffer_t __restrict__ *board_buf,
        int depth)
{
    /* Variables for making moves on the host. */
    cb_errno_t result = CB_EOK;
    cb_mvlst_t mvlst;
    cb_move_t mv;
    cb_state_tables_t state;

    /* Base case. */
    if (depth - GPU_SEARCH_DEPTH == 0) {
        pbfs_board_buf_push(board_buf, board);

        /* Launch the kernel if our buffer is full. */
        if (board_buf.nboards == GPU_MAX_BOARDS_IN_BUF) {
            result = pbfs_kernel(err, cnt, board_buf);
            board_buf->nboards == 0;
        }
        return result;
    }

    /* Generate the moves. */
    cb_gen_board_tables(state, board);
    cb_gen_moves(&mvlst, board, state);

    /* Make moves and move down the tree. */
    for (i = 0; i < cb_mvlst_size(&mvlst); i++) {
        mv = cb_mvlst_at(&mvlst, i);
        cb_make(board, mv);
        cnt += pbfs_host(board, state, depth - 1);
        cb_unmake(board);
    }

    return CB_EOK;
}

cb_errno_t perft_gpu_bfs(cb_board_t *board, int depth)
{
    /* Variables for printing the results. */
    cb_error_t err;
    cb_errno_t result;
    uint64_t total = 0;
    char buf[6];
    uint8_t mv_idx;

    /* Variables for making moves on the host. */
    cb_mvlst_t mvlst;
    cb_move_t mv;
    uint64_t perft_results[CB_MAX_NUM_MOVES];
    cb_state_tables_t state;
    uint64_t cnt = 0;
    int i;

    /* Variables for timing the kernel. */
    uint64_t start_time;
    uint64_t end_time;

    /* Exit early if depth is less than 1. */
    if (depth < 1) {
        printf("No perft with a depth below 1\n");
        return CB_EOK;
    }

    /* Reserve space in the board history. */
    if ((result = cb_reserve_for_make(&err, board,
                    depth - GPU_SEARCH_DEPTH)) != 0) {
        fprintf(stderr, "cb_reserve_for_make: %s\n", err.desc);
        return result;
    }

    /* Loop through all of the first levels and calculate the number of moves. */
    start_time = time_ns();
    cb_gen_board_tables(&state, board);
    cb_gen_moves(&mvlst, board, &state);
    for (i = 0; i < cb_mvlst_size(&mvlst); i++) {
        /* Make the move for the current position. */
        mv = cb_mvlst_at(&mvlst, i);
        cb_make(board, mv);

        /* Search the subtree on the GPU. */
        if ((result = pbfs_host(&err, &cnt, board, depth - 1)) != 0) {
            fprintf(stderr, "pbfs: %s\n", err.desc);
            return result;
        }

        /* Need one more kernel for the remaining moves. */
        if ((result = pbfs_kernel(&err, &cnt, board, depth - 1)) != 0) {
            fprintf(stderr, "pbfs: %s\n", err.desc);
            return result;
        }

        /* Update the total and write the subtree count to the console. */
        total += cnt;
        cb_mv_to_uci_algbr(buf, mv);
        printf("%s: %" PRIu64 "\n", buf, cnt);
        cb_unmake(board);
    }

    /* Check the end time and write the total to the console. */
    end_time = time_ns();
    printf("\n");
    printf("Nodes searched: %" PRIu64 "\n", total);
    printf("Time: %" PRIu64 "ms\n", (end_time - start_time) / 1000000);
    printf("\n");

    return CB_EOK;
}

