
#include <string.h>
#include <stdio.h>
#include <errno.h>
#include <stdbool.h>
#include <stdlib.h>

#include "cb_lib.h"
#include "cb_dbg.h"
#include "perft_cpu.h"
#include "perft_gpu.h"

#define MAX_COMMAND_LEN 512
#define DEFAULT_FEN "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
#define MODDED_FEN "rnbqkbnr/pppppppp/p7/1p6/2p5/3p4/PPPPpPPP/RNBQKpNR w KQkq - 0 1"

typedef enum {
    PERFT_CPU,
    PERFT_GPU
} perft_types_t;

int handle_position(cb_board_t *board)
{
    /* Slice the next word off the command. */
    char *token = strtok(NULL, " \n");
    char *game_str = strtok(NULL, "\n");
    cb_error_t err;
    cb_errno_t result;

    /* Error handling. */
    if (token == NULL) {
        printf("Invalid position command. Usage:\n"
               "position <fen/uci/pgn>\n");
        return 1;
    }

    /* Execute the command. */
    if (strcmp(token, "fen") == 0) {
        if ((result = cb_board_from_uci(&err, board, game_str)) != 0) {
            fprintf(stderr, "cb_board_from_fen: %s\n", err.desc);
            return result;
        }
    } else if (strcmp(token, "uci") == 0) {
        if ((result = cb_board_from_uci(&err, board, game_str)) != 0) {
            fprintf(stderr, "cb_board_from_uci: %s\n", err.desc);
            return result;
        }
    } else if (strcmp(token, "pgn") == 0) {
        if ((result = cb_board_from_pgn(&err, board, game_str)) != 0) {
            fprintf(stderr, "cb_board_from_pgn: %s\n", err.desc);
            return result;
        }
    } else {
        printf("Invalid position command. Usage:\n"
               "position <fen/uci/pgn>\n");
        return 1;
    }

    /* Display information about what happened in the command. */
    if (result == CB_EABORT) {
        return -1;
    } else if (result == CB_EINVAL) {
        printf("Invalid game string\n");
        return 1;
    } else if (result == CB_EILLEGAL) {
        printf("Illegal move in game string\n");
        return 1;
    }

    return 0;
}

int handle_move(cb_board_t *board)
{
    /* Slice off the algebraic part of the move. */
    char *token = strtok(NULL, " \n");
    cb_errno_t result;
    cb_error_t err;
    cb_move_t mv;

    /* Error handling. */
    if (token == NULL) {
        printf("Invalid move command. Usage:\n"
               "m <uci_algbr>\n");
        return 1;
    }

    /* Parse the token into a move. */
    if ((result = cb_mv_from_uci_algbr(&err, &mv, board, token)) != 0) {
        fprintf(stderr, "cb_mv_from_uci_algbr: %s\n", err.desc);
        return result;
    }

    /* Reserve space for the move. */
    if ((result = cb_reserve_for_make(&err, board, 1)) != 0) {
        fprintf(stderr, "cb_reserve_for_make: %s\n", err.desc);
        return result;
    }
    cb_make(board, mv);

    return 0;
}

int handle_undo(cb_board_t *board)
{
    cb_unmake(board);
    return 0;
}

int handle_debug(cb_board_t *board)
{
    /* Slice one token off of the command. */
    char *token = strtok(NULL, " \n");
    cb_state_tables_t state;
    cb_mvlst_t mvlst;

    /* Error handling. */
    if (token == NULL) {
        printf("Invalid move command. Usage:\n"
               "dbg <state/board/moves>\n");
        return 1;
    }

    /* Parse the first token in the command. */
    cb_gen_board_tables(&state, board);
    cb_gen_moves(&mvlst, board, &state);
    if (strcmp(token, "state") == 0)
        cb_print_state(stdout, &state);
    else if (strcmp(token, "board") == 0)
        cb_print_bitboard(stdout, board);
    else if (strcmp(token, "moves") == 0)
        cb_print_moves(stdout, &mvlst);
    else 
        printf("Invalid command\n");
    
    return 0;
}

int handle_board(cb_board_t *board)
{
    cb_print_board_ascii(stdout, board);
    return 0;
}

int handle_perft(cb_board_t *board, perft_types_t type)
{
    /* Slice off the algebraic part of the move. */
    char *token = strtok(NULL, " \n");
    char *endptr;
    int depth;

    /* Verify the command. */
    if (token == NULL) {
        printf("Must provide depth for command perft");
        return 1;
    }

    /* Convert the depth to an integer. */
    errno = 0;
    depth = strtol(token, &endptr, 10);
    if (errno || *endptr != '\0') {
        printf("Depth must be a base 10 integer");
    }

    /* Perform the perft. */
    if (type == PERFT_CPU) {
        return perft_cheat(board, depth);
    } else if (type == PERFT_GPU) {
        return perft_gpu_slow(board, depth);
    }
}

int handle_go(cb_board_t *board)
{
    /* Slice off the algebraic part of the move. */
    char *token = strtok(NULL, " \n");

    if (strcmp(token, "perft") == 0)
        return handle_perft(board, PERFT_CPU);
    else if (strcmp(token, "pgpu") == 0)
        return handle_perft(board, PERFT_GPU);
    
    printf("Invalid go command\n");
    return 0;
}

int parse_input(char *command, cb_board_t *board)
{
    /* Slice one token off of the command. */
    char *token = strtok(command, " \n");

    /* Parse the first token in the command. */
    if (strcmp(token, "position") == 0)
        return handle_position(board);
    if (strcmp(token, "m") == 0)
        return handle_move(board);
    if (strcmp(token, "u") == 0)
        return handle_undo(board);
    if (strcmp(token, "dbg") == 0)
        return handle_debug(board);
    if (strcmp(token, "d") == 0)
        return handle_board(board);
    if (strcmp(token, "go") == 0)
        return handle_go(board);
    if (strcmp(token, "quit") == 0)
        return -1;
    
    printf("Invalid command\n");
    return 0;
}

int main()
{
    cb_mvlst_t moves;
    cb_board_t board;
    cb_state_tables_t state;
    char *command = NULL;
    size_t len = 0;
    ssize_t nread;
    int result = 0;
    bool run_program = true;
    cb_error_t err;
    char default_fen[] = DEFAULT_FEN;

    /* Print version information. */
    printf("kchess debug version 0.0.1 by Joel Kuehne\n");

    /* Set up board tables. */
    err.num = CB_EOK;
    cb_tables_init(&err);
    if (err.num != 0) {
        fprintf(stderr, "cb_table_init_once: %s\n", err.desc);
        result = 1;
        goto out;
    }

    /* Set up the gpu tables. */
    cblib_gpu_init();

    /* Set up the initial board state. */
    if (cb_board_init(&err, &board)) {
        fprintf(stderr, "cb_board_init: %s\n", err.desc);
        result = 1;
        goto out_free_tables;
    }

    if (cb_board_from_fen(&err, &board, default_fen)) {
        fprintf(stderr, "cb_board_from_fen: %s\n", err.desc);
        result = 1;
        goto out_free_board;
    }

    /* Accept commands from the user. */
    while (run_program) {
        if ((nread = getline(&command, &len, stdin)) < 0) {
            perror("getline");
            result = 1;
            goto out_free_command;
        }
        run_program = parse_input(command, &board) >= 0;
    }

out_free_command:
    free(command);
out_free_board:
    cb_board_free(&board);
out_free_tables:
    cblib_gpu_free();
    cb_tables_free();
out:
    return result;
}
