
#ifdef __cplusplus
extern "C" {
#endif

#include "cb_lib.h"
#include "cb_move.h"

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

uint64_t *gpu_bishop_atk_ptrs_h[64];
uint64_t *gpu_rook_atk_ptrs_h[64];

typedef struct {
    uint64_t *color;
    uint64_t *pawns;
    uint64_t *knights;
    uint64_t *bishops;
    uint64_t *rooks;
    uint64_t *kings;
} board_buffer_t;

void cblib_gpu_init()
{
    gpu_init_tables();
}

void cblib_gpu_free()
{
    gpu_free_tables();
}

__global__ void gpu_mvcnt_kernel(board_buffer_t boards, uint8_t *counts, gpu_color_t turn)
{
    gpu_board_t board;
    gpu_state_tables_t state;

    board.bb.pawns = boards.pawns[threadIdx.x + blockIdx.x * blockDim.x];
    board.bb.knights = boards.knights[threadIdx.x + blockIdx.x * blockDim.x];
    board.bb.bishops = boards.bishops[threadIdx.x + blockIdx.x * blockDim.x];
    board.bb.rooks = boards.rooks[threadIdx.x + blockIdx.x * blockDim.x];
    board.bb.kings = boards.kings[threadIdx.x + blockIdx.x * blockDim.x];
    board.bb.color = boards.color[threadIdx.x + blockIdx.x * blockDim.x];
    board.turn = turn;

    board.bb.occ = board.bb.pawns || board.bb.knights || board.bb.bishops || 
        board.bb.rooks || board.bb.kings;
    
    gpu_gen_board_tables(&board, &state);
    counts = gpu_count_moves(&board, &state);
}

__global__ void scan(){
    //TODO
}

__global__ void gpu_gen_mv(board_buffer_t boards, uint32_t *indicies, gpu_move_t *move, gpu_color_t turn) {
    gpu_board_t board;
    gpu_state_tables_t state;
    gpu_search_struct_t ss;

    board.bb.pawns = boards.pawns[threadIdx.x + blockIdx.x * blockDim.x];
    board.bb.knights = boards.knights[threadIdx.x + blockIdx.x * blockDim.x];
    board.bb.bishops = boards.bishops[threadIdx.x + blockIdx.x * blockDim.x];
    board.bb.rooks = boards.rooks[threadIdx.x + blockIdx.x * blockDim.x];
    board.bb.kings = boards.kings[threadIdx.x + blockIdx.x * blockDim.x];
    board.bb.color = boards.color[threadIdx.x + blockIdx.x * blockDim.x];
    board.turn = turn;

    board.bb.occ = board.bb.pawns || board.bb.knights || board.bb.bishops || 
        board.bb.rooks || board.bb.kings;

    ss.moves = moves + indicies[threadIdx.x];
    gpu_gen_board_tables(&board, &state);
    indicies = gpu_gen_moves(&ss, &board, &state);
}

__global__ void internal_expand(){
    //TODO
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
__global__ void gpu_make_moves(board_buffer_t in_boards, board_buffer_t out_boards,
    uint32_t *board_indices, gpu_move_t *moves, gpu_color_t turn){
        int tid = threadIdx.x + blockIdx.x * blockDim.x;
        gpu_board_t board;
        gpu_move_t move;

        uint32_t board_idx = in_board_indices[tid];
        board.bb.pawns = in_board.pawns[board_idx];
        board.bb.knights = in_board.knights[board_idx];
        board.bb.bishops = in_board.bishops[board_idx];
        board.bb.rooks = in_board.rooks[board_idx];
        board.bb.kings = in_board.kings[board_idx];
        board.bb.color = in_board.color[board_idx];
        board.turn = turn;

        move = moves[threadIdx.x + blockIdx.x * blockDim.x];

        board.bb.occ = board.bb.pawns || board.bb.knights || board.bb.bishops || 
            board.bb.rooks || board.bb.kings;

        gpu_make(&ss, &board, mv);

        out_boards.pawns[tid] = board.bb.pawns;
        out_boards.knights[tid] = board.bb.knights;
        out_boards.bishops[tid] = board.bb.bishops;
        out_boards.rooks[tid] = board.bb.rooks;
        out_boards.kings[tid] = board.bb.kings;
        out_boards.color[tid] = board.bb.color;
    
}




