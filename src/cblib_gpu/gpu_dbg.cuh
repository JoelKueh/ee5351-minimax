
#ifndef GPU_DBG_H
#define GPU_DBG_H

#include <stdio.h>
#include "gpu_types.cuh"

#define ANSI_COLOR_RED     "\x1b[31m"
#define ANSI_COLOR_GREEN   "\x1b[32m"
#define ANSI_COLOR_YELLOW  "\x1b[33m"
#define ANSI_COLOR_BLUE    "\x1b[34m"
#define ANSI_COLOR_MAGENTA "\x1b[35m"
#define ANSI_COLOR_CYAN    "\x1b[36m"
#define ANSI_COLOR_RESET   "\x1b[0m"

#define PRINT_BUF_LEN 1024

__device__ void gpu_prep_bb_byte(char *buf, uint64_t bb, uint64_t rank) {
    for (int i = 0; i < 14; i++) {
        buf[i] = ' ';
    }
    buf[ 0] = (bb & (UINT64_C(1) << (rank * 8 + 0))) ? '1' : '0';
    buf[ 2] = (bb & (UINT64_C(1) << (rank * 8 + 1))) ? '1' : '0';
    buf[ 4] = (bb & (UINT64_C(1) << (rank * 8 + 2))) ? '1' : '0';
    buf[ 6] = (bb & (UINT64_C(1) << (rank * 8 + 3))) ? '1' : '0';
    buf[ 8] = (bb & (UINT64_C(1) << (rank * 8 + 4))) ? '1' : '0';
    buf[10] = (bb & (UINT64_C(1) << (rank * 8 + 5))) ? '1' : '0';
    buf[12] = (bb & (UINT64_C(1) << (rank * 8 + 6))) ? '1' : '0';
    buf[14] = (bb & (UINT64_C(1) << (rank * 8 + 7))) ? '1' : '0';
}

__device__ void gpu_print_bitboard(gpu_board_t *board)
{
    const char *wheaders[] = { "PAWN", "KNIGHT", "BISHOP", "ROOK", "KING", "COLOR", "OCC" };
    int i, j;
    char byte[PRINT_BUF_LEN];

    /* Print white pieces. */
    printf("\n");
    for (i = 0; i < 7; i++)
        printf("%-17s", wheaders[i]);
    printf("\n");
    for (i = 0; i < 7; i++)
        printf("===============  ");
    printf("\n");
    for (i = 0; i < 8; i++) {
        for (j = 0; j < 5; j++) {
            gpu_prep_bb_byte(byte, board->bb.piece[j], i);
            printf("%s  ", byte);
        }
        gpu_prep_bb_byte(byte, board->bb.color, i);
        printf("%s  ", byte);
        gpu_prep_bb_byte(byte, board->bb.occ, i);
        printf("%s  ", byte);
        printf("\n");
    }
}

__device__ void gpu_print_state(gpu_state_tables_t *state)
{
    const char *headers[] = { "PINS", "THREATS", "CHECKS", "CHECK_BLOCKS" };
    char byte[PRINT_BUF_LEN];
    int i;

    printf("\n");
    for (i = 0; i < 4; i++)
        printf("%-17s", headers[i]);
    printf("\n");
    for (i = 0; i < 4; i++)
        printf("===============  ");
    printf("\n");
    for (i = 0; i < 8; i++) {
        gpu_prep_bb_byte(byte, state->pinned, i);
        printf("%s  ", byte);
        gpu_prep_bb_byte(byte, state->threats, i);
        printf("%s  ", byte);
        gpu_prep_bb_byte(byte, state->checks, i);
        printf("%s  ", byte);
        gpu_prep_bb_byte(byte, state->check_blocks, i);
        printf("%s  ", byte);
        printf("\n");
    }
    printf("\n");
}

#endif /* GPU_DBG_H */

