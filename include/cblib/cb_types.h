
#ifndef CB_TYPES_H
#define CB_TYPES_H

#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>

#define CB_MAX_NUM_MOVES 218
#define CB_ERROR_STRLEN 128

/**
 * @breif Error codes for different operations that can take place.
 */
typedef enum {
    CB_ENOMEM = -2,     /**< Non-recoverable error for malloc failures. */
    CB_EABORT = -1,     /**< Generic non-recoverable error. */
    CB_EOK = 0,         /**< Default return value. */
    CB_EINVAL,          /**< Recoverable invalid formatting error. */
    CB_EILLEGAL,        /**< Recoverable illegal move specification error. */
} cb_errno_t;

/**
 * @breif Defines an error type for errors that will be returned by the library.
 */
typedef struct {
    cb_errno_t num;                 /**< The error code for the error. */
    char desc[CB_ERROR_STRLEN];     /**< A string description of the error. */
} cb_error_t;

/**
 * @breif Enumerates board piece and turn colors.
 */
enum {
    CB_WHITE = 1,
    CB_BLACK = 0
};
typedef uint8_t cb_color_t;

/**
 * @breif Enumerates piece ids.
 *
 * A piece ID contains both the piece type and the color.
 * Primarily for used when converting to a string representation.
 */
enum {
    CB_PID_EMPTY = 0b0000,

    CB_PID_WHITE_PAWN   = 0b0001,
    CB_PID_WHITE_KNIGHT = 0b0010,
    CB_PID_WHITE_BISHOP = 0b0011,
    CB_PID_WHITE_ROOK   = 0b0100,
    CB_PID_WHITE_QUEEN  = 0b0101,
    CB_PID_WHITE_KING   = 0b0110,

    CB_PID_BLACK_PAWN   = 0b1001,
    CB_PID_BLACK_KNIGHT = 0b1010,
    CB_PID_BLACK_BISHOP = 0b1011,
    CB_PID_BLACK_ROOK   = 0b1100,
    CB_PID_BLACK_QUEEN  = 0b1101,
    CB_PID_BLACK_KING   = 0b1110
};
typedef uint8_t cb_pid_t;

/**
 * @breif Enumerates piece types.
 *
 * A piece type contains only information about type, not about color.
 * This type can be used to index the bitboard array and is stored in the mailbox.
 */
enum {
    CB_PTYPE_PAWN   = 0,
    CB_PTYPE_KNIGHT = 1,
    CB_PTYPE_BISHOP = 2,
    CB_PTYPE_ROOK   = 3,
    CB_PTYPE_QUEEN  = 4,
    CB_PTYPE_KING   = 5,
    CB_PTYPE_EMPTY  = 6
};
typedef uint8_t cb_ptype_t;

/**
 * @breif Simple type for a chess move.
 */
typedef uint16_t cb_move_t;

/**
 * @breif Holds a list of moves.
 *
 * This data structure is not dynamically sized. A size of CB_MAX_NUM_MOVES was chosen as it
 * is the supposed maximum number of moves that can be played at any given chess position.
 */
typedef struct {
    cb_move_t moves[CB_MAX_NUM_MOVES];      /**< The list of moves. */
    uint8_t head;                           /**< The index of the top of the stack. */
} cb_mvlst_t;

/**
 * @breif Simple type to hold board state info not captured in the piece organization.
 */
typedef uint16_t cb_history_t;

/**
 * @breif Stack element that holds the history of the board.
 */
typedef struct {
    cb_history_t hist;      /**< The history state at a given position. */
    cb_move_t move;         /**< The last move played at a given position. */
} cb_hist_ele_t;

/**
 * @breif Resizeable stack structure that holds the history of the board.
 */
typedef struct {
    cb_hist_ele_t *data;    /**< The actual stack structure. */
    int count;              /**< The number of full slots in the stack. */
    int size;               /**< The allocated size of the stack. */
} cb_hist_stack_t;

/**
 * @breif State table data structure that is useful in move generation.
 */
typedef struct {
    uint64_t threats;       /**< A bitmask for all pieces that threaten the king. */
    uint64_t checks;        /**< A bitmask for all pieces that check the king. */
    uint64_t check_blocks;  /**< A bitmask for all squares that can break a check. */
    uint64_t pins[10];      /**< A set of bitmasks for all active pin rays. */
} cb_state_tables_t;

/**
 * @breif Bitboard data structure that actually stores peice data.
 *
 * The board squares are counted from the top left of the board (black pieces first) row-wise.
 * E.g. if a black rook is on rank 6 file A, then:
 *
 *      piece[CB_BLACK][CB_PTYPE_ROOK] & (UINT64_T(1) << 16) == 1
 */
typedef struct {
    uint64_t color[2];      /**< A set of bitmasks for colored pieces. */
    uint64_t piece[2][6];   /**< A set of bitmasks for piece types and colors. */
    uint64_t occ;           /**< The union of the above bitmasks. For occupied squares. */
} cb_bitboard_t;

/**
 * @breif Mailbox data structure that duplicates data of bitboard.
 *
 * As stated on the programming wiki, bitboards are great at answering questions like "Where
 * are the white knights?" but struggles to answer questions like "What piece is on this square?"
 *
 * Maintaining this represenation improves efficienty of move generation and application.
 */
typedef struct {
    uint8_t data[64]; /* This is an array of cb_ptype_t but it is stored as a uint8_t. */
} cb_mailbox_t;

/**
 * @breif Data structure that defines the state of the board.
 *
 * This does not include other information about move timing, it only contains information that
 * is useful to move generation and the like.
 */
typedef struct {
    cb_bitboard_t bb;       /**< The bitboard. */
    cb_mailbox_t mb;        /**< The mailbox. */
    cb_hist_stack_t hist;   /**< The history stack. */
    uint8_t turn;           /**< The current turn. */
    uint32_t fullmove_num;  /**< The fullmove number. */
} cb_board_t;

/**
 * @breif Function that creates populates an error struct with error information.
 * @param err The error struct to populate.
 * @param cb_errno The error number.
 * @param format A printf compliant format string.
 * @param ... All remaining arguments passed into the format string.
 */
static cb_errno_t cb_mkerr(cb_error_t *err, cb_errno_t cb_errno, char *format, ...)
{
    va_list args;
    err->num = cb_errno;
    va_start(args, format);
    vsnprintf(err->desc, CB_ERROR_STRLEN, format, args);
    va_end(args);
    return cb_errno;
}


#endif /* CB_TYPES_H */

