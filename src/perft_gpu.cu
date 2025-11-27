
#include <inttypes.h>

#include "crosstime.h"

#ifdef __cplusplus
extern "C" {
#endif

#include "cb_lib.h"
#include "cb_move.h"

#ifdef __cplusplus
}
#endif

#include "perft_gpu.h"
#include "cblib_gpu/gpu_types.cuh"
#include "cblib_gpu/gpu_board.cuh"
#include "cblib_gpu/gpu_move.cuh"
#include "cblib_gpu/gpu_gen.cuh"
#include "cblib_gpu/gpu_tables.cuh"
#include "cblib_gpu/gpu_lib.cuh"
#include "cblib_gpu/gpu_dbg.cuh"

uint64_t *gpu_bishop_atk_ptrs_h[64];
uint64_t *gpu_rook_atk_ptrs_h[64];

void cblib_gpu_init()
{
    gpu_init_tables();
}

void cblib_gpu_free()
{
    gpu_free_tables();
}

__global__ void perft_gpu_slow_kernel(
        gpu_search_struct_node_t *ss_nodes, gpu_board_t *boards,
        uint64_t *counts, gpu_move_t *moves, uint8_t *num_moves_from_root,
        int depth)
{
    /* Variables containing search and board state information. */
    gpu_search_struct_t ss;
    gpu_state_tables_t state;
    gpu_board_t board;
    uint8_t mv_from_rt;

    /* Prepare the output for the search struct. */
    ss.positions = ss_nodes;

    /* Load the board from memory. */
    board = *boards;

    /* TODO: Remove me. Print out initial board. */
    gpu_print_bitboard(&board);

    /* Search through the tree. */
    ss.move_counts[0] = 0;
    ss.move_idx[0] = 0;
    ss.depth = 0;
    gpu_gen_board_tables(&board, &state);
    gpu_gen_moves(&ss, &board, &state);

    /* Loop through the generated moves and add them to the output. */
    mv_from_rt = ss.move_counts[0];
    for (int i = 0; i < mv_from_rt; i++) {
        moves[i] = gpu_ss_get_move(&ss, i);
        counts[i] = 0;
    }

    /* Search through the tree. DFS in a while loop. */
    while (true) {
        /* We have traversed an entire subtree, back out. */
        if (gpu_all_nodes_traversed(&ss)) {
            if (ss.depth == 0)
                break;
            gpu_unmake(&ss, &board);
            continue;
        }

        /* Make a child move and traverse its subtree. */
        gpu_traverse_to_next_child(&ss, &board);

        /* If we have bottomed out, then add to our count. */
        if (ss.depth == depth) {
            counts[ss.move_idx[0]-1] += 1;
            gpu_unmake(&ss, &board);
            continue;
        }

        /* In all other cases, generate moves after the traversal. */
        gpu_gen_board_tables(&board, &state);
        gpu_gen_moves(&ss, &board, &state);
    }

    /* Set output move count. */
    *num_moves_from_root = mv_from_rt;

    /* TODO: Remove me. Print out final board. */
    gpu_print_bitboard(&board);

    return;
}

int perft_gpu_slow(cb_board_t *board, int depth)
{
    /* Variables for printing the results. */
    uint64_t total;
    cb_move_t mv;
    char buf[6];
    uint8_t mv_idx;

    /* Variables for interfacing with the kernel. */
    gpu_board_t h_board;
    uint64_t h_perft_counts[CB_MAX_NUM_MOVES];
    cb_move_t h_perft_moves[CB_MAX_NUM_MOVES];
    uint8_t h_num_moves_from_root;
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
    h_board.bb.pawns = 
        board->bb.piece[CB_WHITE][CB_PTYPE_PAWN] | board->bb.piece[CB_BLACK][CB_PTYPE_PAWN];
    h_board.bb.knights = 
        board->bb.piece[CB_WHITE][CB_PTYPE_KNIGHT] | board->bb.piece[CB_BLACK][CB_PTYPE_KNIGHT];
    h_board.bb.bishops = 
        board->bb.piece[CB_WHITE][CB_PTYPE_BISHOP] | board->bb.piece[CB_BLACK][CB_PTYPE_BISHOP] |
        board->bb.piece[CB_WHITE][CB_PTYPE_QUEEN] | board->bb.piece[CB_BLACK][CB_PTYPE_QUEEN];
    h_board.bb.rooks = 
        board->bb.piece[CB_WHITE][CB_PTYPE_ROOK] | board->bb.piece[CB_BLACK][CB_PTYPE_ROOK] |
        board->bb.piece[CB_WHITE][CB_PTYPE_QUEEN] | board->bb.piece[CB_BLACK][CB_PTYPE_QUEEN];
    h_board.bb.kings = 
        board->bb.piece[CB_WHITE][CB_PTYPE_KING] | board->bb.piece[CB_BLACK][CB_PTYPE_KING];

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
    printf("Number of moves from root: %d\n", h_num_moves_from_root);

    /* Loop over the output. */
    total = 0;
    for (mv_idx = 0; mv_idx < h_num_moves_from_root; mv_idx++) {
        mv = h_perft_moves[mv_idx];
        total += h_perft_counts[mv_idx];
        cb_mv_to_uci_algbr(buf, mv);
        printf("%s: %" PRIu64 "\n", buf, h_perft_counts[mv_idx]);
    }
    end_time = time_ns();

    /* Allocate space in device memory for the board and results. */
    cudaFree(d_ss_nodes);
    cudaFree(d_board);
    cudaFree(d_perft_counts);
    cudaFree(d_perft_moves);
    cudaFree(d_num_moves_from_root);

    /* Print out the results. */
    printf("\n");
    printf("Nodes searched: %" PRIu64 "\n", total);
    printf("Time: %" PRIu64 "ms\n", (end_time - start_time) / 1000000);
    printf("\n");

    return 0;
}
