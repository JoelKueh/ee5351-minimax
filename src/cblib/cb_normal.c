
#include <string.h>

#include "cb_tables.h"
#include "cb_const.h"

uint64_t pawn_atks[2][64];
uint64_t knight_atks[64];
uint64_t king_atks[64];
uint64_t to_from_table[64][64];

/**
 * Generates a lookup table from a set of offsets on each square.
 */
void gen_table_from_offsets(uint64_t table[64], const int8_t offsets[], uint8_t offset_len)
{
    int i, j, sq;

    /* Zero the table. */
    memset(table, 0, 64 * sizeof(uint64_t));

    /* Loop through all squares. Build the mask from the set of offsets. */
    for (i = 0; i < 64; i++) {
        for (j = 0; j < offset_len; j++) {
            sq = i + offsets[j];
            if (sq < 0 || sq >= 64)
                continue;
            table[i] |= UINT64_C(1) << sq;
        }

        /* Remove all invalid moves.
         * If we are in the right two columns, we cannot jump to the left. */
        if ((UINT64_C(1) << i) & BB_RIGHT_TWO_COLS) {
            table[i] &= ~BB_LEFT_TWO_COLS;
        }

        if ((UINT64_C(1) << i) & BB_LEFT_TWO_COLS) {
            table[i] &= ~BB_RIGHT_TWO_COLS;
        }
    }
}

void gen_pawn_atk_table()
{
    const int8_t OFFSETS[2][8] = {{7, 9}, {-7, -9}};
    gen_table_from_offsets(pawn_atks[0], OFFSETS[0], 2);
    gen_table_from_offsets(pawn_atks[1], OFFSETS[1], 2);
}

void gen_knight_atk_table()
{
    const int8_t OFFSETS[8] = {-17, -15, -10, -6, 6, 10, 15, 17};
    gen_table_from_offsets(knight_atks, OFFSETS, 8);
}

void gen_king_atk_table()
{
    const int8_t OFFSETS[8] = {9, 8, 7, 1, -1, -7, -8, -9};
    gen_table_from_offsets(king_atks, OFFSETS, 8);
}

/**
 * Get the direction of the ray that extends from sq1 to sq2.
 */
uint8_t cb_get_ray_direction(uint8_t sq1, uint8_t sq2)
{
    int8_t sq1_rank = sq1 / 8;
    int8_t sq1_file = sq1 % 8;
    int8_t sq2_rank = sq2 / 8;
    int8_t sq2_file = sq2 % 8;
    uint8_t direction;

    /* This line is a mess, but it does convert from square and file to ray direction. */
    direction = (sq1_rank == sq2_rank) ? (sq1 < sq2 ? CB_DIR_R : CB_DIR_L) :
        (sq1_file == sq2_file) ? (sq1 < sq2 ? CB_DIR_D : CB_DIR_U) :
        (sq1_file + sq1_rank == sq2_file + sq2_rank) ? (sq1 < sq2 ? CB_DIR_DL : CB_DIR_UR) :
        (sq1_file - sq1_rank == sq2_file - sq2_rank) ? (sq1 < sq2 ? CB_DIR_DR : CB_DIR_UL) :
        CB_DIR_INVALID;

    return direction;
}

/**
 * Generate the ray that connects sq1 and sq2.
 */
uint64_t get_connecting_ray(uint64_t sq1, uint64_t sq2)
{
    uint64_t mask = 0;
    uint8_t direction;

    /* Get the ray direction. */
    if ((direction = cb_get_ray_direction(sq1, sq2)) == CB_DIR_INVALID) {
        return mask;
    }

    /* Slide along the ray until we reach the destination. */
    mask = 0;
    while (sq1 != sq2) {
        mask |= UINT64_C(1) << sq1;
        sq1 += dir_offset_mapping[direction];
    }

    return mask;
}

/**
 * Generate the table of rays that extend from one square to another.
 */
void gen_to_from_table()
{
    int i, j;

    /* Loop over all of the squares on the board and generate the respective rays. */
    for (i = 0; i < 64; i++) {
        for (j = 0; j < 64; j++) {
            to_from_table[i][j] = get_connecting_ray(i, j);
        }
    }
}

void cb_init_normal_tables()
{
    gen_pawn_atk_table();
    gen_knight_atk_table();
    gen_king_atk_table();
    gen_to_from_table();
}

uint64_t cb_read_pawn_atk_msk(uint8_t sq, cb_color_t color)
{
    return pawn_atks[color][sq];
}

uint64_t cb_read_knight_atk_msk(uint8_t sq)
{
    return knight_atks[sq];
}

uint64_t cb_read_king_atk_msk(uint8_t sq)
{
    return king_atks[sq];
}

uint64_t cb_read_tf_table(uint8_t sq1, uint8_t sq2)
{
    return to_from_table[sq1][sq2];
}
