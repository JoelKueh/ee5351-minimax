
#include "cb_dbg.h"
#include "cb_types.h"
#include "cb_lib.h"
#include "cb_board.h"
#include "cb_move.h"

#define ANSI_COLOR_RED     "\x1b[31m"
#define ANSI_COLOR_GREEN   "\x1b[32m"
#define ANSI_COLOR_YELLOW  "\x1b[33m"
#define ANSI_COLOR_BLUE    "\x1b[34m"
#define ANSI_COLOR_MAGENTA "\x1b[35m"
#define ANSI_COLOR_CYAN    "\x1b[36m"
#define ANSI_COLOR_RESET   "\x1b[0m"

#define PRINT_BUF_LEN 1024

void cb_print_mv_hist(FILE *f, cb_board_t *board)
{
    int i = 0;
    cb_move_t mv;
    char buf[PRINT_BUF_LEN];
    cb_hist_stack_t *hist = &board->hist;

    /* First move isn't valid. */
    for (i = 1; i < (hist->count - 1); i++) {
        cb_mv_to_uci_algbr(buf, hist->data[i].move);
        fprintf(f, "%s -> ", buf);
    }
    cb_mv_to_uci_algbr(buf, hist->data[i].move);
    fprintf(f, "%s\n", buf);
}

void cb_print_board_ascii(FILE *f, cb_board_t *board)
{
    const char SEPERATOR[] = " +---+---+---+---+---+---+---+---+";
    const char PIECE_SEP[] = " | ";
    const char FILE_LINE[] = "   A   B   C   D   E   F   G   H";

    int row, col;
    char str_rep[8][8];

    cb_board_to_str(str_rep, board);
    fprintf(f, "%s\n", SEPERATOR);
    for (row = 0; row < 8; row++) {
        fprintf(f, "%s", PIECE_SEP);
        for (col = 0; col < 8; col++) {
            fprintf(f, "%c%s", str_rep[row][col], PIECE_SEP);
        }
        fprintf(f, "%d\n", 8 - row);
        fprintf(f, "%s\n", SEPERATOR);
    }
    fprintf(f, "%s\n", FILE_LINE);
}

void cb_print_board_utf8(FILE *f, cb_board_t *board)
{

}

void prep_bb_byte(char *buf, uint64_t bb, uint64_t rank) {
    sprintf(buf, "%s %s %s %s %s %s %s %s" ANSI_COLOR_RESET,
            (bb & (UINT64_C(1) << (rank * 8 + 0))) ? ANSI_COLOR_GREEN "1" : ANSI_COLOR_RED "0",
            (bb & (UINT64_C(1) << (rank * 8 + 1))) ? ANSI_COLOR_GREEN "1" : ANSI_COLOR_RED "0",
            (bb & (UINT64_C(1) << (rank * 8 + 2))) ? ANSI_COLOR_GREEN "1" : ANSI_COLOR_RED "0",
            (bb & (UINT64_C(1) << (rank * 8 + 3))) ? ANSI_COLOR_GREEN "1" : ANSI_COLOR_RED "0",
            (bb & (UINT64_C(1) << (rank * 8 + 4))) ? ANSI_COLOR_GREEN "1" : ANSI_COLOR_RED "0",
            (bb & (UINT64_C(1) << (rank * 8 + 5))) ? ANSI_COLOR_GREEN "1" : ANSI_COLOR_RED "0",
            (bb & (UINT64_C(1) << (rank * 8 + 6))) ? ANSI_COLOR_GREEN "1" : ANSI_COLOR_RED "0",
            (bb & (UINT64_C(1) << (rank * 8 + 7))) ? ANSI_COLOR_GREEN "1" : ANSI_COLOR_RED "0");
}

void cb_print_bitboard(FILE *f, cb_board_t *board)
{
    const char *wheaders[] = { "WHITE", "PAWN", "KNIGHT", "BISHOP", "ROOK", "QUEEN", "KING", "OCC" };
    const char *bheaders[] = { "BLACK", "PAWN", "KNIGHT", "BISHOP", "ROOK", "QUEEN", "KING" };
    int i, j;
    char byte[PRINT_BUF_LEN];

    /* Print white pieces. */
    fprintf(f, "\n");
    for (i = 0; i < 8; i++)
        fprintf(f, "%-17s", wheaders[i]);
    fprintf(f, "\n");
    for (i = 0; i < 8; i++)
        fprintf(f, "===============  ");
    fprintf(f, "\n");
    for (i = 0; i < 8; i++) {
        prep_bb_byte(byte, board->bb.color[1], i);
        fprintf(f, "%s  ", byte);
        for (j = 0; j < 6; j++) {
            prep_bb_byte(byte, board->bb.piece[1][j], i);
            fprintf(f, "%s  ", byte);
        }
        prep_bb_byte(byte, board->bb.occ, i);
        fprintf(f, "%s  ", byte);
        fprintf(f, "\n");
    }

    /* Print black pieces. */
    fprintf(f, "\n");
    for (i = 0; i < 7; i++)
        fprintf(f, "%-17s", bheaders[i]);
    fprintf(f, "\n");
    for (i = 0; i < 7; i++)
        fprintf(f, "===============  ");
    fprintf(f, "\n");
    for (i = 0; i < 8; i++) {
        prep_bb_byte(byte, board->bb.color[0], i);
        fprintf(f, "%s  ", byte);
        for (j = 0; j < 6; j++) {
            prep_bb_byte(byte, board->bb.piece[0][j], i);
            fprintf(f, "%s  ", byte);
        }
        fprintf(f, "\n");
    }
    fprintf(f, "\n");
}

void cb_print_state(FILE *f, cb_state_tables_t *state)
{
    const char *headers[] = { "THREATS", "CHECKS", "CHECK_BLOCKS" };
    int i, j;
    char byte[PRINT_BUF_LEN];

    /* Print pins. */
    fprintf(f, "\nPINS\n");
    for (i = 0; i < 9; i++)
        fprintf(f, "===============  ");
    fprintf(f, "\n");
    for (i = 0; i < 8; i++) {
        for (j = 0; j < 9; j++) {
            prep_bb_byte(byte, state->pins[j], i);
            fprintf(f, "%s  ", byte);
        }
        fprintf(f, "\n");
    }

    /* Print other tables. */
    fprintf(f, "\n");
    for (i = 0; i < 3; i++)
        fprintf(f, "%-17s", headers[i]);
    fprintf(f, "\n");
    for (i = 0; i < 3; i++)
        fprintf(f, "===============  ");
    fprintf(f, "\n");
    for (i = 0; i < 8; i++) {
        prep_bb_byte(byte, state->threats, i);
        fprintf(f, "%s  ", byte);
        prep_bb_byte(byte, state->checks, i);
        fprintf(f, "%s  ", byte);
        prep_bb_byte(byte, state->check_blocks, i);
        fprintf(f, "%s  ", byte);
        fprintf(f, "\n");
    }
    fprintf(f, "\n");
}

void cb_print_moves(FILE *f, cb_mvlst_t *mvlst)
{
    char buf[6];
    int i;

    for (i = 0; i < cb_mvlst_size(mvlst); i++) {
        cb_mv_to_uci_algbr(buf, cb_mvlst_at(mvlst, i));
        fprintf(f, "%s ", buf);
    }
    fprintf(f, "\n");
}
