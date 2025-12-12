
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

#define GPU_SEARCH_DEPTH 4
#define GPU_MAX_BOARDS_IN_BUF (1 << 10)

uint64_t *gpu_bishop_atk_ptrs_h[64];
uint64_t *gpu_rook_atk_ptrs_h[64];

typedef struct {
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

__global__ void gpu_mvcnt_kernel(board_buffer_t boards, uint8_t *counts,
        gpu_color_t turn)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    gpu_board_t board;
    gpu_state_tables_t state;

    /* Load the board from global memory. */
    board.bb.pawns = boards.pawns[tid];
    board.bb.knights = boards.knights[tid];
    board.bb.bishops = boards.bishops[tid];
    board.bb.rooks = boards.rooks[tid];
    board.bb.kings = boards.kings[tid];
    board.bb.color = boards.color[tid];
    board.turn = turn;

    /* Prepare the occupancy as the union of all other bitmasks. */
    board.bb.occ = board.bb.pawns || board.bb.knights || board.bb.bishops || 
        board.bb.rooks || board.bb.kings;
    
    /* Count the moves at the given state. */
    gpu_gen_board_tables(&board, &state);
    counts[tid] = gpu_count_moves(&board, &state);
}

__global__ void scan(){
    //TODO
}

__global__ void gpu_gen_mv(board_buffer_t boards, uint32_t *in_indicies,
        gpu_move_t *moves, uint32_t *out_indicies, gpu_color_t turn)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    gpu_board_t board;
    gpu_state_tables_t state;
    gpu_search_struct_t ss;
    uint32_t out_idx;
    uint8_t i;

    /* Load the board from global memory. */
    board.bb.pawns = boards.pawns[tid];
    board.bb.knights = boards.knights[tid];
    board.bb.bishops = boards.bishops[tid];
    board.bb.rooks = boards.rooks[tid];
    board.bb.kings = boards.kings[tid];
    board.bb.color = boards.color[tid];
    board.turn = turn;

    /* Prepare the occupancy as the union of all other bitmasks. */
    board.bb.occ = board.bb.pawns || board.bb.knights || board.bb.bishops || 
        board.bb.rooks || board.bb.kings;

    /* Load the output index from global memory. */
    out_idx = in_indicies[tid];

    /* Generate all of the moves. */
    ss.moves = moves + out_idx;
    gpu_gen_board_tables(&board, &state);
    gpu_gen_moves(&ss, &board, &state);

    /* Write the thread ID to the output indicies.
     * Theoritically this removes the need for the inverval expand kenrnel.
     */
    for (i = 0; i < ss.count; i++)
        out_indicies[out_idx] = tid;
}

/**
 * Inputs:
 *  - in_boards: Vector of boards.
 *  - moves: Vector of moves (index by thread index).
 *  - board_idx: Mapping of thread index to index in in_boards.
 *    e.g. start_board = in_boards[board_idx[tx + bx * bdim.x]]
 *
 * Outputs:
 *  - out_boards: Vector of boards (index by thread index)
 *    out_boards[tx + bx * bdim.x] = end_board
 */
__global__ void gpu_make_moves(board_buffer_t in_boards,
        board_buffer_t out_boards, uint32_t *board_indices,
        gpu_move_t *moves, gpu_color_t turn) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    gpu_board_t board;
    gpu_move_t move;

    /* Load the board index from global memory. */
    uint32_t board_idx = board_indices[tid];

    /* Load the board from global memory. */
    board.bb.pawns = in_boards.pawns[board_idx];
    board.bb.knights = in_boards.knights[board_idx];
    board.bb.bishops = in_boards.bishops[board_idx];
    board.bb.rooks = in_boards.rooks[board_idx];
    board.bb.kings = in_boards.kings[board_idx];
    board.bb.color = in_boards.color[board_idx];
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
}

cb_errno_t pbfs_kernel(cb_error_t __restrict__ *err,
        uint64_t __restrict__ *counts, gpu_board_t __restrict__ *boards)
{
    
}

uint32_t pbfs_board_buf_push(board_buffer_t __restrict__ *board_buf,
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

    return board_idx;
}

cb_errno_t pbfs_host(cb_error_t __restrict__ *err, uint64_t __restrict__ *cnt,
        gpu_board_t __restrict__ *board, board_buffer_t __restrict__ *board_buf,
        int depth)
{
    /* Variables for making moves on the host. */
    cb_mvlst_t mvlst;
    cb_move_t mv;
    cb_state_tables_t state;

    /* Base case. */
    if (depth - GPU_SEARCH_DEPTH == 0) {
        pbfs_board_buf_push(board_buf, board);
        if (board_buf.nboards == GPU_
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

    /* Reserve the board history. This line guarantees that make will never write
     * past its proper bounds. */
    /* TODO: Investigate cpu gpu depth split. */
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
        mv = cb_mvlst_at(&mvlst, i);
        cb_make(board, mv);
        if ((result = pbfs_host(&err, &cnt, board, depth - 1)) != 0) {
            fprintf(stderr, "cb_reserve_for_make: %s\n", err.desc);
            return result;
        }
        total += cnt;
        cb_mv_to_uci_algbr(buf, mv);
        printf("%s: %" PRIu64 "\n", buf, cnt);
        cb_unmake(board);
    }
    end_time = time_ns();
    printf("\n");
    printf("Nodes searched: %" PRIu64 "\n", total);
    printf("Time: %" PRIu64 "ms\n", (end_time - start_time) / 1000000);
    printf("\n");

    return CB_EOK;
}

