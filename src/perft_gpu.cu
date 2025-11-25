
#include <inttypes.h>

#include "crosstime.h"
#include "cb_lib.h"
#include "cb_move.h"

#include "perft_gpu.h"
#include "cblib_gpu/gpu_types.cuh"
#include "cblib_gpu/gpu_board.cuh"
#include "cblib_gpu/gpu_move.cuh"
#include "cblib_gpu/gpu_gen.cuh"

__global__ void perft_gpu_slow_kernel(
        gpu_search_struct_node_t *ss_nodes, gpu_board_t *boards,
        uint64_t *counts, gpu_move_t *moves, uint8_t *num_moves_from_root,
        int depth)
{
    gpu_search_struct_t ss;
    gpu_board_t board;
    gpu_move_t mv;

    gpu_state_tables_t state;
    uint64_t perft_results[GPU_MAX_NUM_MOVES];
    uint64_t cnt = 0;
    uint64_t total = 0;
    char buf[6];
    int i;

    /* Prepare the output for the search struct. */
    ss.positions = ss_nodes;

    /* Load the board from memory. */
    board = *boards;

    /* Search through the tree. */
    for (int d = 0; d < depth; d++) {
        gpu_gen_board_tables(&board, &state);
        gpu_gen_moves(&ss, &board, &state);
    }

    /* Loop through the generated moves and add them to the output. */
    for (int i = 0; i < ss.offset; i++) {
        moves[i] = ss.positions[0].moves[0][i];
        counts[i] = 1;
    }

    /* Hooray, we're done! */
    return;
}

int perft_gpu_slow(cb_board_t *board, int depth)
{
    /* Error handling and printing variables. */
    cb_errno_t result;
    cb_error_t err;

    /* Variables for printing the results. */
    uint64_t total;
    cb_move_t mv;
    char buf[6];
    int i;

    /* Variables for interfacing with the kernel. */
    gpu_board_t h_board;
    uint64_t h_perft_counts[CB_MAX_NUM_MOVES];
    cb_move_t h_perft_moves[CB_MAX_NUM_MOVES];
    uint64_t h_num_moves_from_root;
    gpu_search_struct_node_t *d_ss_nodes;
    gpu_board_t *d_board;
    uint64_t *d_perft_counts;
    gpu_move_t *d_perft_moves;
    uint8_t *d_num_moves_from_root;

    /* Variables for timing the kernel. */
    uint64_t start_time;
    uint64_t end_time;

    /* Exit early if depth is less than 1. */
    if (depth < 1) {
        printf("No perft with a depth below 1\n");
        return 0;
    }

    /* TODO: Remove me. */
    depth = 1;

    /* Allocate space in device memory for the board and results. */
    cudaMalloc((void**)&d_ss_nodes,
            GPU_MAX_SEARCH_DEPTH * sizeof(gpu_search_struct_node_t));
    cudaMalloc((void**)&d_board, sizeof(gpu_board_t));
    cudaMalloc((void**)&d_perft_counts, GPU_MAX_NUM_MOVES * sizeof(uint64_t));
    cudaMalloc((void**)&d_perft_moves, GPU_MAX_NUM_MOVES * sizeof(uint64_t));
    cudaMalloc((void**)&d_num_moves_from_root, sizeof(uint8_t));

    /* Build the GPU board from the CPU board. */
    h_board.state = board->hist.data[board->hist.count-1].hist;
    h_board.turn = board->turn;
    h_board.bb.color = board->bb.color[CB_WHITE];
    h_board.bb.occ = board->bb.occ;
    h_board.bb.piece[GPU_PTYPE_PAWN] =
        board->bb.piece[CB_WHITE][CB_PTYPE_PAWN] | board->bb.piece[CB_BLACK][CB_PTYPE_PAWN];
    h_board.bb.piece[GPU_PTYPE_KNIGHT] =
        board->bb.piece[CB_WHITE][CB_PTYPE_KNIGHT] | board->bb.piece[CB_BLACK][CB_PTYPE_KNIGHT];
    h_board.bb.piece[GPU_PTYPE_BISHOP] =
        board->bb.piece[CB_WHITE][CB_PTYPE_BISHOP] | board->bb.piece[CB_BLACK][CB_PTYPE_BISHOP] |
        board->bb.piece[CB_WHITE][CB_PTYPE_QUEEN] | board->bb.piece[CB_BLACK][CB_PTYPE_QUEEN];
    h_board.bb.piece[GPU_PTYPE_ROOK] =
        board->bb.piece[CB_WHITE][CB_PTYPE_ROOK] | board->bb.piece[CB_BLACK][CB_PTYPE_ROOK] |
        board->bb.piece[CB_WHITE][CB_PTYPE_QUEEN] | board->bb.piece[CB_BLACK][CB_PTYPE_QUEEN];

    /* Copy the data to the GPU. */
    cudaMemcpy(d_board, &h_board, sizeof(gpu_board_t), cudaMemcpyHostToDevice);

    /* Loop through all of the first levels and calculate the number of moves. */
    start_time = time_ns();
    perft_gpu_slow_kernel<<<1, 1, 1>>>(d_ss_nodes, d_board,
            d_perft_counts, d_perft_moves, d_num_moves_from_root, depth);
    cudaMemcpy(&h_perft_counts, d_perft_counts,
            CB_MAX_NUM_MOVES * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_perft_moves, d_perft_moves,
            CB_MAX_NUM_MOVES * sizeof(gpu_move_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_num_moves_from_root, d_num_moves_from_root,
            sizeof(uint8_t), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    /* Loop over the output. */
    for (i = 0; i < h_num_moves_from_root; i++) {
        mv = h_perft_moves[i];
        total += h_perft_counts[i];
        cb_mv_to_uci_algbr(buf, mv);
        printf("%s: %" PRIu64 "\n", buf, h_perft_counts[i]);
        cb_unmake(board);
    }
    end_time = time_ns();

    /* Print out the results. */
    printf("\n");
    printf("Nodes searched: %" PRIu64 "\n", total);
    printf("Time: %" PRIu64 "ms\n", (end_time - start_time) / 1000000);
    printf("\n");

    return 0;
}
