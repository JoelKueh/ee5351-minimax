
#ifdef __cplusplus
extern "C" {
#endif

#include "cb_lib.h"
#include "cb_move.h"
#include "crosstime.h"

#ifdef __cplusplus
}
#endif

#include "./scan.cuh"
#include "./reduce.cuh"
#include "perft_gpu.h"
#include "cblib_gpu/gpu_types.cuh"
#include "cblib_gpu/gpu_board.cuh"
#include "cblib_gpu/gpu_move.cuh"
#include "cblib_gpu/gpu_gen.cuh"
#include "cblib_gpu/gpu_tables.cuh"
#include "cblib_gpu/gpu_lib.cuh"
#include "cblib_gpu/gpu_dbg.cuh"
#include "cblib_gpu/gpu_count_moves.cuh"

/* Block dim parameters. */
#define CHESS_BLOCK_DIM 64
#define REDUCTION_BLOCK_DIM 1024
#define SCAN_BLOCK_DIM 1024

/* Gpu search depth and launch parameters. */
#define GPU_SEARCH_DEPTH 3
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

__host__ __device__ void bbuf_alloc(board_buffer_t *__restrict__ bbuf)
{
    bbuf->state = (gpu_history_t*)malloc(GPU_MAX_BOARDS_IN_BUF * sizeof(gpu_history_t));
    bbuf->color = (uint64_t*)malloc(GPU_MAX_BOARDS_IN_BUF * sizeof(uint64_t));
    bbuf->pawns = (uint64_t*)malloc(GPU_MAX_BOARDS_IN_BUF * sizeof(uint64_t));
    bbuf->knights = (uint64_t*)malloc(GPU_MAX_BOARDS_IN_BUF * sizeof(uint64_t));
    bbuf->bishops = (uint64_t*)malloc(GPU_MAX_BOARDS_IN_BUF * sizeof(uint64_t));
    bbuf->rooks = (uint64_t*)malloc(GPU_MAX_BOARDS_IN_BUF * sizeof(uint64_t));
    bbuf->kings = (uint64_t*)malloc(GPU_MAX_BOARDS_IN_BUF * sizeof(uint64_t));
    bbuf->nboards = 0;
}

__host__ __device__ void bbuf_free(board_buffer_t *__restrict__ bbuf)
{
    free(bbuf->state);
    free(bbuf->color);
    free(bbuf->pawns);
    free(bbuf->knights);
    free(bbuf->bishops);
    free(bbuf->rooks);
    free(bbuf->kings);
}

__host__ __device__ void bbuf_read(board_buffer_t *__restrict__ board_buf,
        gpu_board_t *__restrict__ board, uint32_t idx)
{
    board->bb.pawns = board_buf->pawns[idx];
    board->bb.knights = board_buf->knights[idx];
    board->bb.bishops = board_buf->bishops[idx];
    board->bb.rooks = board_buf->rooks[idx];
    board->bb.kings = board_buf->kings[idx];
    board->bb.color = board_buf->color[idx];
    board->state = board_buf->state[idx];

    board->bb.occ = board->bb.pawns | board->bb.knights | board->bb.bishops
        | board->bb.rooks | board->bb.kings;
}

__host__ __device__ void bbuf_write(board_buffer_t *__restrict__ board_buf,
        gpu_board_t *__restrict__ board, uint32_t idx)
{
    board_buf->pawns[idx] = board->bb.pawns;
    board_buf->knights[idx] = board->bb.knights;
    board_buf->bishops[idx] = board->bb.bishops;
    board_buf->rooks[idx] = board->bb.rooks;
    board_buf->kings[idx] = board->bb.kings;
    board_buf->color[idx] = board->bb.color;
    board_buf->state[idx] = board->state;
}

__host__ void bbuf_push(board_buffer_t *__restrict__ board_buf,
        cb_board_t *__restrict__ board)
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
    board_buf->state[board_buf->nboards] = board->hist.data[board->hist.count-1].hist;
    
    /* Increment the board counter. */
    board_buf->nboards += 1;
}


__host__ void cuda_bbuf_alloc(board_buffer_t *__restrict__ d_bbuf,
        cudaStream_t s)
{
    cudaMallocAsync((void**)&d_bbuf->state, d_bbuf->nboards * sizeof(gpu_history_t), s);
    cudaMallocAsync((void**)&d_bbuf->color, d_bbuf->nboards * sizeof(uint64_t), s);
    cudaMallocAsync((void**)&d_bbuf->pawns, d_bbuf->nboards * sizeof(uint64_t), s);
    cudaMallocAsync((void**)&d_bbuf->knights, d_bbuf->nboards * sizeof(uint64_t), s);
    cudaMallocAsync((void**)&d_bbuf->bishops, d_bbuf->nboards * sizeof(uint64_t), s);
    cudaMallocAsync((void**)&d_bbuf->rooks, d_bbuf->nboards * sizeof(uint64_t), s);
    cudaMallocAsync((void**)&d_bbuf->kings, d_bbuf->nboards * sizeof(uint64_t), s);
}

__host__ void cuda_bbuf_free(board_buffer_t *__restrict__ d_bbuf,
        cudaStream_t s)
{
    cudaFreeAsync(d_bbuf->state, s);
    cudaFreeAsync(d_bbuf->color, s);
    cudaFreeAsync(d_bbuf->pawns, s);
    cudaFreeAsync(d_bbuf->knights, s);
    cudaFreeAsync(d_bbuf->bishops, s);
    cudaFreeAsync(d_bbuf->rooks, s);
    cudaFreeAsync(d_bbuf->kings, s);
}

__host__ void cuda_bbuf_memcpy(
        board_buffer_t *__restrict__ d_bbuf,
        board_buffer_t *__restrict__ bbuf, cudaStream_t s)
{
    cudaMemcpyAsync(d_bbuf->state, bbuf->state, d_bbuf->nboards * sizeof(gpu_history_t), cudaMemcpyHostToDevice);
    cudaMemcpyAsync(d_bbuf->color, bbuf->color, d_bbuf->nboards * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpyAsync(d_bbuf->pawns, bbuf->pawns, d_bbuf->nboards * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpyAsync(d_bbuf->knights, bbuf->knights, d_bbuf->nboards * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpyAsync(d_bbuf->bishops, bbuf->bishops, d_bbuf->nboards * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpyAsync(d_bbuf->rooks, bbuf->rooks, d_bbuf->nboards * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpyAsync(d_bbuf->kings, bbuf->kings, d_bbuf->nboards * sizeof(uint64_t), cudaMemcpyHostToDevice);
}

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
__global__ void gpu_mvcnt_kernel(board_buffer_t boards, uint32_t *counts,
        gpu_color_t turn)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    gpu_board_t board;
    gpu_state_tables_t state;

    /* Early return for out-of-range threads. */
    if (tid >= boards.nboards)
        return;

    /* Load the board from global memory. */
    bbuf_read(&boards, &board, tid);
    board.turn = turn;
    
    /* Count the moves at the given state. */
    gpu_gen_board_tables(&board, &state);
    counts[tid] = gpu_count_moves(&board, &state);
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
__global__ void gpu_mvgen_kernel(board_buffer_t boards, uint32_t *write_indicies,
        gpu_move_t *moves, uint32_t *source_board_mapping,
        gpu_color_t turn)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    gpu_board_t board;
    gpu_state_tables_t state;
    gpu_search_struct_t ss;
    uint32_t write_idx;
    uint8_t i;

    /* Early return for out-of-range threads. */
    if (tid >= boards.nboards)
        return;

    /* Load the board from global memory. */
    bbuf_read(&boards, &board, tid);
    board.turn = turn;

    /* Load the output index from global memory. */
    write_idx = write_indicies[tid];

    /* Generate all of the moves. */
    ss.moves = moves + write_idx;
    ss.count = 0;
    gpu_gen_board_tables(&board, &state);
    gpu_gen_moves(&ss, &board, &state);

    /* Write the thread ID to the output indicies.
     * Theoritically this removes the need for the inverval expand kenrnel.
     */
    for (i = 0; i < ss.count; i++)
        source_board_mapping[write_idx+i] = tid;
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
__global__ void gpu_mvmake_kernel(board_buffer_t in_boards,
        board_buffer_t out_boards, uint32_t *source_board_indicies,
        gpu_move_t *moves, uint32_t nmoves, gpu_color_t turn)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    gpu_board_t board;
    gpu_move_t move;

    /* Early return for out-of-range threads. */
    if (tid >= nmoves)
        return;

    /* Load the board index from global memory. */
    uint32_t board_idx = source_board_indicies[tid];

    /* Load the board from global memory. */
    bbuf_read(&in_boards, &board, board_idx);
    board.turn = turn;

    /* Make the move. */
    move = moves[tid];
    gpu_make(&board, move);

    /* Write the resulting board to global memory. */
    bbuf_write(&out_boards, &board, tid);
}

__global__ void gpu_mvmake_and_count_kernel(board_buffer_t boards,
        uint32_t *source_board_indices, gpu_move_t *moves, uint8_t *counts,
        uint32_t nmoves, gpu_color_t turn)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    gpu_state_tables_t state;
    gpu_board_t board;
    gpu_move_t move;

    /* Early return for out-of-range threads. */
    if (tid >= nmoves)
        return;

    /* Load the board index from global memory. */
    uint32_t board_idx = source_board_indices[tid];

    /* Load the board from global memory. */
    bbuf_read(&boards, &board, board_idx);
    board.turn = turn;

    /* Make the move. */
    move = moves[tid];
    gpu_make(&board, move);

    /* Count the moves at the new state. */
    gpu_gen_board_tables(&board, &state);
    counts[tid] = gpu_count_moves(&board, &state);
}

__global__ void pbfs_kernel_cdp(uint64_t *__restrict__ count,
        board_buffer_t boards, gpu_color_t turn)
{
#if 0
    /* Variables for the search. */
    board_buffer_t new_boards;
    board_buffer_t *p_new_boards = &new_boards;
    board_buffer_t *p_boards = &boards;
    gpu_move_t *moves;
    uint32_t *move_indicies;
    uint32_t *source_board_indicies;
    uint8_t *counts;
    uint32_t nmoves;

    /* Create stream for device kernel calls. */
    cudaStream_t s;
    cudaStreamCreate(&s);

    /* Perform the search to generate the top level board vector. */
    for (int i = 0; i < GPU_SEARCH_DEPTH; i++) {
        /* Allocate memory for the counts. */
        cudaMalloc((void**)&counts, boards->nboards * sizeof(uint32_t));

        /* Count the moves at the current level. */
        dim3 blockDim(CHESS_BLOCK_DIM, 1, 1);
        dim3 gridDim(ceil((float)boards->nboards / CHESS_BLOCK_DIM), 1, 1);
        gpu_mvcnt_kernel<<<gridDim, blockDim, s>>>(boards, counts, turn);

        /* Break out of the loop after generating counts if we are done. */
        if (i == i - GPU_SEARCH_DEPTH)
            break;

        /* Allocate memory for the move indicies that we will scan. */
        cudaMalloc((void**)&move_indicies, (boards->nboards + 1) * sizeof(uint32_t));

        /* Scan the counts returned from the previous kernel. */
        dim3 blockDim(SCAN_BLOCK_DIM, 1, 1);
        dim3 gridDim(ceil((float)boards->nboards / SCAN_BLOCK_DIM), 1, 1);
        scan<<<gridDim, blockDim, s>>>((move_indicies + 1), counts, boards->nboards);
        move_indicies[0] = 0;

        /* Synchronize with the stream to get the results of the scan. */
        cudaStreamSynchronize(s);

        /* Allocate memory for the moves. */
        nmoves = move_indicies[boards->nboards];
        cudaMalloc((void**)&moves, nmoves * sizeof(gpu_move_t));
        cudaMalloc((void**)&source_board_indicies, nmoves * sizeof(uint32_t));

        /* Generate the moves. */
        dim3 blockDim(CHESS_BLOCK_DIM, 1, 1);
        dim3 gridDim(ceil((float) / CHESS_BLOCK_DIM), 1, 1);
        gpu_mvgen_kernel<<<gridDim, blockDim, s>>>(boards, write_indicies,
                moves, source_board_indicies, turn);

        /* Allocate memory for the boards. */
        bbuf_alloc(p_new_boards, s);

        /* Make the moves on the boards. */
        dim3 blockDim(CHESS_BLOCK_DIM, 1, 1);
        dim3 gridDim(ceil((float)boards->nboards / CHESS_BLOCK_DIM), 1, 1);
        gpu_make_kernel<<<gridDim, blockDim, s>>>(p_boards, p_new_boards,
                source_board_indicies, moves, turn);

        /* Synchronize with the stream before performing this rounds frees. */
        cudaStreamSynchronize(s);
        cudaFree(counts);
        cudaFree(move_indices);
        cudaFree(moves);
        cudaFree(source_board_indicies);
        bbuf_free(p_boards, s);
        p_boards = p_new_boards;

        /* We have just made a move, change the current turn. */
        turn = !turn;
    }

    /* Reduce the count vector. */
    dim3 blockDim(REDUCTION_BLOCK_DIM, 1, 1);
    dim3 gridDim(ceil((float)boards->nboards / REDUCTION_BLOCK_DIM), 1, 1);
    reduce<<<gridDim, blockDim, s>>>(count, counts);

    /* Destroy the child stream. */
    cudaStreamDestroy(s);
#endif
}

cb_errno_t pbfs_kernel(cb_error_t *__restrict__ err,
        uint64_t *__restrict__ count, board_buffer_t *__restrict__ board_buf,
        gpu_color_t in_turn)
{
    /* Variables for the search. */
	board_buffer_t boards = *board_buf;
    board_buffer_t new_boards;
    gpu_move_t *moves;
    uint32_t *move_counts;
    uint32_t *move_indicies;
    uint32_t *source_board_indicies;
    uint32_t nmoves;
    uint8_t *last_layer_counts;
    gpu_color_t turn = in_turn;

    /* Create stream for device kernel calls. */
    cudaStream_t s;
    cudaStreamCreate(&s);

    /* Perform the search to generate the top level board vector. */
    for (int i = 0; i < GPU_SEARCH_DEPTH; i++) {
        /* Allocate memory for the move counts that we will scan. This
         * must be padded to support the scan that we will do later. */
        int move_counts_size = ceil(boards.nboards / (float)SCAN_TILE_SIZE) * SCAN_TILE_SIZE;
        int move_indicies_size = ceil((boards.nboards + 1) / (float)SCAN_TILE_SIZE) * SCAN_TILE_SIZE;
        cudaMalloc((void**)&move_counts, move_counts_size * sizeof(uint32_t));
        cudaMalloc((void**)&move_indicies, move_indicies_size * sizeof(uint32_t));

        /* Count the moves at the current level. */
        dim3 blockDim(CHESS_BLOCK_DIM, 1, 1);
        dim3 gridDim(ceil((float)boards.nboards / CHESS_BLOCK_DIM), 1, 1);
        gpu_mvcnt_kernel<<<gridDim, blockDim, 0, s>>>(boards, move_counts, turn);

        /* Scan the counts returned from the previous kernel. */
        blockDim = dim3(SCAN_BLOCK_DIM, 1, 1);
        gridDim = dim3(ceil((float)boards.nboards / SCAN_BLOCK_DIM), 1, 1);
        launch_scan(move_indicies + 1, move_counts, boards.nboards);
        cudaMemset(move_indicies, 0, sizeof(uint32_t));

        /* Synchronize with the stream to get the results of the scan. */
        cudaStreamSynchronize(s);

        /* Allocate memory for the moves. */
        cudaMemcpy(&nmoves, &(move_indicies[boards.nboards]),
                sizeof(uint32_t), cudaMemcpyDeviceToHost);
        cudaMalloc((void**)&moves, nmoves * sizeof(gpu_move_t));
        cudaMalloc((void**)&source_board_indicies, nmoves * sizeof(uint32_t));

        /* Generate the moves. */
        blockDim = dim3(CHESS_BLOCK_DIM, 1, 1);
        gridDim = dim3(ceil((float) boards.nboards / CHESS_BLOCK_DIM), 1, 1);
        gpu_mvgen_kernel<<<gridDim, blockDim, 0, s>>>(boards, move_indicies,
                moves, source_board_indicies, turn);

        /* Break out of the loop after generating counts if we are done. */
        if (i >= GPU_SEARCH_DEPTH - 2) {
            turn = !turn;
            break;
        }

        /* Allocate memory for the boards. */
        new_boards.nboards = nmoves;
        cuda_bbuf_alloc(&new_boards, s);

        /* Make the moves on the boards. */
        blockDim = dim3(CHESS_BLOCK_DIM, 1, 1);
        gridDim = dim3(ceil((float)boards.nboards / CHESS_BLOCK_DIM), 1, 1);
        gpu_mvmake_kernel<<<gridDim, blockDim, 0, s>>>(boards, new_boards,
                source_board_indicies, moves, nmoves, turn);

        /* Synchronize with the stream before performing this rounds frees. */
        cudaStreamSynchronize(s);
        cudaFree(move_counts);
        cudaFree(move_indicies);
        cudaFree(moves);
        cudaFree(source_board_indicies);
        cuda_bbuf_free(&boards, s);
        boards = new_boards;

        /* We have just made a move, change the current turn. */
        turn = !turn;
    }
    
    /* Synchronize with the stream. */
    cudaStreamSynchronize(s);

    /* Count the moves with the given position and move vectors. */
    cudaMalloc((void**)&last_layer_counts, nmoves * sizeof(uint8_t));
    dim3 blockDim(CHESS_BLOCK_DIM, 1, 1);
    dim3 gridDim(ceil((float)nmoves / CHESS_BLOCK_DIM), 1, 1);
    gpu_mvmake_and_count_kernel<<<gridDim, blockDim, 0, s>>>(boards,
            source_board_indicies, moves, last_layer_counts, nmoves, turn);

    /* Reduce the count vector. */
    blockDim = dim3(REDUCTION_BLOCK_DIM, 1, 1);
    gridDim = dim3(ceil((float)boards.nboards / REDUCTION_BLOCK_DIM), 1, 1);
    launch_reduction(count, last_layer_counts, nmoves, s);
    cudaStreamSynchronize(s);

    /* Free up resources. */
    cuda_bbuf_free(&new_boards, s);
    cudaFree(last_layer_counts);
    cudaStreamDestroy(s);

    /* Destroy the child stream. */

    return CB_EOK;
}

cb_errno_t pbfs_kernel_launch(cb_error_t *__restrict__ err,
        uint64_t *__restrict__ count, board_buffer_t *__restrict__ bbuf,
        gpu_color_t turn)
{
    /* NOTE: We use cudaMallocAsync because we need device ordered malloc. */

    board_buffer_t d_bbuf;
    uint64_t *d_count;
    uint64_t result;

    /* Allocate device memory. */
    d_bbuf.nboards = bbuf->nboards;
    cuda_bbuf_alloc(&d_bbuf, NULL);
    cudaMallocAsync((void**)&d_count, sizeof(uint64_t), NULL);

    /* Copy the board buffer to device memory. */
    cuda_bbuf_memcpy(&d_bbuf, bbuf, NULL);
    cudaDeviceSynchronize();

    /* Launch the cuda dynamic parallelism kernel on the boards. */
#if 0
    dim3 blockDim(1, 1, 1);
    dim3 gridDim(1, 1, 1);
    pbfs_kernel_cdp<<<gridDim, blockDim>>>(err, d_count, d_bbuf, turn);
#endif
    pbfs_kernel(err, d_count, &d_bbuf, turn);

    /* Copy the result back to the host. */
    cudaMemcpyAsync(&result, d_count, sizeof(uint64_t), cudaMemcpyDeviceToHost);
    *count = result;

    /* Free up the board vector and kernel results. */
    //cuda_bbuf_free(&d_bbuf, NULL); /* This is freed in the kernel. */
    cudaFree(d_count);

    return CB_EOK;
}


cb_errno_t pbfs_host(cb_error_t *err, uint64_t *cnt, cb_board_t *board,
        board_buffer_t *board_buf, int depth)
{
    /* Variables for making moves on the host. */
    cb_errno_t result = CB_EOK;
    cb_mvlst_t mvlst;
    cb_move_t mv;
    cb_state_tables_t state;
    uint8_t i;

    /* Base case. */
    if (depth < GPU_SEARCH_DEPTH) {
        bbuf_push(board_buf, board);

        /* Launch the kernel if our buffer is full. */
        if (board_buf->nboards == GPU_MAX_BOARDS_IN_BUF) {
            result = pbfs_kernel_launch(err, cnt, board_buf, board->turn);
            board_buf->nboards = 0;
        }
        return result;
    }

    /* Generate the moves. */
    cb_gen_board_tables(&state, board);
    cb_gen_moves(&mvlst, board, &state);

    /* Make moves and move down the tree. */
    for (i = 0; i < cb_mvlst_size(&mvlst); i++) {
        mv = cb_mvlst_at(&mvlst, i);
        cb_make(board, mv);
        cnt += pbfs_host(err, cnt, board, board_buf, depth - 1);
        cb_unmake(board);
    }

    return CB_EOK;
}

cb_errno_t perft_gpu_bfs(cb_board_t *board, int depth)
{
    /* Variables for printing the results. */
    board_buffer_t h_bbuf;
    cb_error_t err;
    cb_errno_t result;
    uint64_t total = 0;
    char buf[6];

    /* Variables for making moves on the host. */
    cb_mvlst_t mvlst;
    cb_move_t mv;
    cb_state_tables_t state;
    uint64_t cnt = 0;
    int i;

    /* Variables for timing the kernel. */
    uint64_t start_time;
    uint64_t end_time;

    /* Allocate space in the board buffer. */
    bbuf_alloc(&h_bbuf);

    /* Exit early if depth is less than 1. */
    if (depth < GPU_SEARCH_DEPTH) {
        printf("No perft with a depth below %d\n", GPU_SEARCH_DEPTH);
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
        if ((result = pbfs_host(&err, &cnt, board, &h_bbuf, depth - 1)) != 0) {
            fprintf(stderr, "pbfs: %s\n", err.desc);
            return result;
        }

        /* Need one more kernel for the remaining moves. */
        if ((result = pbfs_kernel_launch(&err, &cnt, &h_bbuf, board->turn)) != 0) {
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

    /* Free the allocated space in the board buffer. */
    bbuf_free(&h_bbuf);

    return CB_EOK;
}

