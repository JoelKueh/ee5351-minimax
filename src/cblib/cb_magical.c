
#include <stdlib.h>
#include <errno.h>
#include <assert.h>

#include "cb_tables.h"
#include "cb_bitutil.h"

const int8_t dir_offset_mapping[8] = { 1, -7, -8, -9, -1, 7, 8, 9 };

const uint8_t NUM_BISHOP_BITS[64] = {
    6, 5, 5, 5, 5, 5, 5, 6,
    5, 5, 5, 5, 5, 5, 5, 5,
    5, 5, 7, 7, 7, 7, 5, 5,
    5, 5, 7, 9, 9, 7, 5, 5,
    5, 5, 7, 9, 9, 7, 5, 5,
    5, 5, 7, 7, 7, 7, 5, 5,
    5, 5, 5, 5, 5, 5, 5, 5,
    6, 5, 5, 5, 5, 5, 5, 6
};

const uint8_t NUM_ROOK_BITS[64] = {
    12, 11, 11, 11, 11, 11, 11, 12,
    11, 10, 10, 10, 10, 10, 10, 11,
    11, 10, 10, 10, 10, 10, 10, 11,
    11, 10, 10, 10, 10, 10, 10, 11,
    11, 10, 10, 10, 10, 10, 10, 11,
    11, 10, 10, 10, 10, 10, 10, 11,
    11, 10, 10, 10, 10, 10, 10, 11,
    12, 11, 11, 11, 11, 11, 11, 12
};

const uint64_t ROOK_MAGICS[64] = {
    UINT64_C(0xa8002c000108020),
    UINT64_C(0x6c00049b0002001),
    UINT64_C(0x100200010090040),
    UINT64_C(0x2480041000800801),
    UINT64_C(0x280028004000800),
    UINT64_C(0x900410008040022),
    UINT64_C(0x280020001001080),
    UINT64_C(0x2880002041000080),
    UINT64_C(0xa000800080400034),
    UINT64_C(0x4808020004000),
    UINT64_C(0x2290802004801000),
    UINT64_C(0x411000d00100020),
    UINT64_C(0x402800800040080),
    UINT64_C(0xb000401004208),
    UINT64_C(0x2409000100040200),
    UINT64_C(0x1002100004082),
    UINT64_C(0x22878001e24000),
    UINT64_C(0x1090810021004010),
    UINT64_C(0x801030040200012),
    UINT64_C(0x500808008001000),
    UINT64_C(0xa08018014000880),
    UINT64_C(0x8000808004000200),
    UINT64_C(0x201008080010200),
    UINT64_C(0x801020000441091),
    UINT64_C(0x800080204005),
    UINT64_C(0x1040200040100048),
    UINT64_C(0x120200402082),
    UINT64_C(0xd14880480100080),
    UINT64_C(0x12040280080080),
    UINT64_C(0x100040080020080),
    UINT64_C(0x9020010080800200),
    UINT64_C(0x813241200148449),
    UINT64_C(0x491604001800080),
    UINT64_C(0x100401000402001),
    UINT64_C(0x4820010021001040),
    UINT64_C(0x400402202000812),
    UINT64_C(0x209009005000802),
    UINT64_C(0x810800601800400),
    UINT64_C(0x4301083214000150),
    UINT64_C(0x204026458e001401),
    UINT64_C(0x40204000808000),
    UINT64_C(0x8001008040010020),
    UINT64_C(0x8410820820420010),
    UINT64_C(0x1003001000090020),
    UINT64_C(0x804040008008080),
    UINT64_C(0x12000810020004),
    UINT64_C(0x1000100200040208),
    UINT64_C(0x430000a044020001),
    UINT64_C(0x280009023410300),
    UINT64_C(0xe0100040002240),
    UINT64_C(0x200100401700),
    UINT64_C(0x2244100408008080),
    UINT64_C(0x8000400801980),
    UINT64_C(0x2000810040200),
    UINT64_C(0x8010100228810400),
    UINT64_C(0x2000009044210200),
    UINT64_C(0x4080008040102101),
    UINT64_C(0x40002080411d01),
    UINT64_C(0x2005524060000901),
    UINT64_C(0x502001008400422),
    UINT64_C(0x489a000810200402),
    UINT64_C(0x1004400080a13),
    UINT64_C(0x4000011008020084),
    UINT64_C(0x26002114058042)
};

const uint64_t BISHOP_MAGICS[64] = {
    UINT64_C(0x89a1121896040240),
    UINT64_C(0x2004844802002010),
    UINT64_C(0x2068080051921000),
    UINT64_C(0x62880a0220200808),
    UINT64_C(0x4042004000000),
    UINT64_C(0x100822020200011),
    UINT64_C(0xc00444222012000a),
    UINT64_C(0x28808801216001),
    UINT64_C(0x400492088408100),
    UINT64_C(0x201c401040c0084),
    UINT64_C(0x840800910a0010),
    UINT64_C(0x82080240060),
    UINT64_C(0x2000840504006000),
    UINT64_C(0x30010c4108405004),
    UINT64_C(0x1008005410080802),
    UINT64_C(0x8144042209100900),
    UINT64_C(0x208081020014400),
    UINT64_C(0x4800201208ca00),
    UINT64_C(0xf18140408012008),
    UINT64_C(0x1004002802102001),
    UINT64_C(0x841000820080811),
    UINT64_C(0x40200200a42008),
    UINT64_C(0x800054042000),
    UINT64_C(0x88010400410c9000),
    UINT64_C(0x520040470104290),
    UINT64_C(0x1004040051500081),
    UINT64_C(0x2002081833080021),
    UINT64_C(0x400c00c010142),
    UINT64_C(0x941408200c002000),
    UINT64_C(0x658810000806011),
    UINT64_C(0x188071040440a00),
    UINT64_C(0x4800404002011c00),
    UINT64_C(0x104442040404200),
    UINT64_C(0x511080202091021),
    UINT64_C(0x4022401120400),
    UINT64_C(0x80c0040400080120),
    UINT64_C(0x8040010040820802),
    UINT64_C(0x480810700020090),
    UINT64_C(0x102008e00040242),
    UINT64_C(0x809005202050100),
    UINT64_C(0x8002024220104080),
    UINT64_C(0x431008804142000),
    UINT64_C(0x19001802081400),
    UINT64_C(0x200014208040080),
    UINT64_C(0x3308082008200100),
    UINT64_C(0x41010500040c020),
    UINT64_C(0x4012020c04210308),
    UINT64_C(0x208220a202004080),
    UINT64_C(0x111040120082000),
    UINT64_C(0x6803040141280a00),
    UINT64_C(0x2101004202410000),
    UINT64_C(0x8200000041108022),
    UINT64_C(0x21082088000),
    UINT64_C(0x2410204010040),
    UINT64_C(0x40100400809000),
    UINT64_C(0x822088220820214),
    UINT64_C(0x40808090012004),
    UINT64_C(0x910224040218c9),
    UINT64_C(0x402814422015008),
    UINT64_C(0x90014004842410),
    UINT64_C(0x1000042304105),
    UINT64_C(0x10008830412a00),
    UINT64_C(0x2520081090008908),
    UINT64_C(0x40102000a0a60140)
};

const uint8_t MAX_BITS_IN_TABLE = 12;
const uint32_t MAX_TABLE_SIZE = 1 << MAX_BITS_IN_TABLE;

uint64_t bishop_occ_mask[64];
uint64_t rook_occ_mask[64];

uint64_t *bishop_atks[64];
uint64_t *rook_atks[64];

static inline uint16_t get_bishop_key(uint8_t sq, uint64_t occ)
{
    /* Hash the occupancy mask and compute the key. */
    occ &= bishop_occ_mask[sq];
    occ *= BISHOP_MAGICS[sq];
    return occ >> (64 - NUM_BISHOP_BITS[sq]);
}

static inline uint16_t get_rook_key(uint8_t sq, uint64_t occ)
{
    /* Hash the occupancy mask and compute the key. */
    occ &= rook_occ_mask[sq];
    occ *= ROOK_MAGICS[sq];
    return occ >> (64 - NUM_ROOK_BITS[sq]);
}

/**
 * Returns the bishop attack mask given an occupancy set and a square.
 */
uint64_t cb_read_bishop_atk_msk(uint8_t sq, uint64_t occ)
{
    uint16_t key = get_bishop_key(sq, occ);
    return bishop_atks[sq][key];
}

/**
 * Returns the rook attack mask given an occupancy set and a square.
 */
uint64_t cb_read_rook_atk_msk(uint8_t sq, uint64_t occ)
{
    uint16_t key = get_rook_key(sq, occ);
    return rook_atks[sq][key];
}

/**
 * Returns the occupancy mask for a rook on a square. e.g:
 *
 *      . . . . . . . .
 *      . . x . . . . .
 *      . . x . . . . .
 *      . x . x x x x .
 *      . . x . . . . .
 *      . . x . . . . .
 *      . . x . . . . .
 *      . . . . . . . .
 */
uint64_t get_rook_occ_mask(uint8_t sq)
{
    uint64_t result = 0;
    int8_t source_rank = sq / 8;
    int8_t source_file = sq % 8;
    int8_t rank;
    int8_t file;

    /* Down */
    rank = source_rank + 1;
    while (rank <= 6) {
        result |= UINT64_C(1) << (source_file + rank * 8);
        rank++;
    }

    /* Up */
    rank = source_rank - 1;
    while (rank >= 1) {
        result |= UINT64_C(1) << (source_file + rank * 8);
        rank--;
    }

    /* Right */
    file = source_file + 1;
    while (file <= 6) {
        result |= UINT64_C(1) << (file + source_rank * 8);
        file++;
    }

    /* Left */
    file = source_file - 1;
    while (file >= 1) {
        result |= UINT64_C(1) << (file + source_rank * 8);
        file--;
    }

    return result;
}

/**
 * Returns the occupancy mask for a bishop on a square. e.g:
 *
 *      . . . . . . . .
 *      . . . . x . . .
 *      . x . x . . . .
 *      . . . . . . . .
 *      . x . x . . . .
 *      . . . . x . . .
 *      . . . . . x . .
 *      . . . . . . . .
 */
uint64_t get_bishop_occ_mask(uint8_t sq)
{
    uint64_t result = 0;
    int8_t source_rank = sq / 8;
    int8_t source_file = sq % 8;
    int8_t rank;
    int8_t file;

    /* Down-Right */
    rank = source_rank + 1;
    file = source_file + 1;
    while (rank <= 6 && file <= 6) {
        result |= UINT64_C(1) << (file + rank * 8);
        rank++;
        file++;
    }

    /* Down-Left */
    rank = source_rank + 1;
    file = source_file - 1;
    while (rank <= 6 && file >= 1) {
        result |= UINT64_C(1) << (file + rank * 8);
        rank++;
        file--;
    }

    /* Up-Right */
    rank = source_rank - 1;
    file = source_file + 1;
    while (rank >= 1 && file <= 6) {
        result |= UINT64_C(1) << (file + rank * 8);
        rank--;
        file++;
    }

    /* Up-Left */
    rank = source_rank - 1;
    file = source_file - 1;
    while (rank >= 1 && file >= 1) {
        result |= UINT64_C(1) << (file + rank * 8);
        rank--;
        file--;
    }

    return result;
}

/**
 * Returns the attack mask for a rook on a square with a given occupancy set.
 * This mask is much the same as the occupancy mask but you break out of the loop upon hitting
 * a piece in the occupancy mask.
 */
uint64_t get_rook_atk_mask(uint8_t sq, uint64_t occ)
{
    uint64_t result = 0;
    int8_t source_rank = sq / 8;
    int8_t source_file = sq % 8;
    int8_t rank;
    int8_t file;

    /* Down */
    rank = source_rank + 1;
    while (rank <= 7) {
        result |= UINT64_C(1) << (source_file + rank * 8);
        if (occ & (UINT64_C(1) << (source_file + rank * 8)))
            break;
        rank++;
    }

    /* Up */
    rank = source_rank - 1;
    while (rank >= 0) {
        result |= UINT64_C(1) << (source_file + rank * 8);
        if (occ & (UINT64_C(1) << (source_file + rank * 8)))
            break;
        rank--;
    }

    /* Right */
    file = source_file + 1;
    while (file <= 7) {
        result |= UINT64_C(1) << (file + source_rank * 8);
        if (occ & (UINT64_C(1) << (file + source_rank * 8)))
            break;
        file++;
    }

    /* Left */
    file = source_file - 1;
    while (file >= 0) {
        result |= UINT64_C(1) << (file + source_rank * 8);
        if (occ & (UINT64_C(1) << (file + source_rank * 8)))
            break;
        file--;
    }

    return result;
}

/**
 * Returns the attack mask for a rook on a square with a given occupancy set.
 * This mask is much the same as the occupancy mask but you break out of the loop upon hitting
 * a piece in the occupancy mask.
 */
uint64_t get_bishop_atk_mask(uint8_t sq, uint64_t occ)
{
    uint64_t result = 0;
    int8_t source_rank = sq / 8;
    int8_t source_file = sq % 8;
    int8_t rank;
    int8_t file;

    /* Down-Right */
    rank = source_rank + 1;
    file = source_file + 1;
    while (rank <= 7 && file <= 7) {
        result |= UINT64_C(1) << (file + rank * 8);
        if (occ & (UINT64_C(1) << (file + rank * 8)))
            break;
        rank++;
        file++;
    }

    /* Down-Left */
    rank = source_rank + 1;
    file = source_file - 1;
    while (rank <= 7 && file >= 0) {
        result |= UINT64_C(1) << (file + rank * 8);
        if (occ & (UINT64_C(1) << (file + rank * 8)))
            break;
        rank++;
        file--;
    }

    /* Up-Right */
    rank = source_rank - 1;
    file = source_file + 1;
    while (rank >= 0 && file <= 7) {
        result |= UINT64_C(1) << (file + rank * 8);
        if (occ & (UINT64_C(1) << (file + rank * 8)))
            break;
        rank--;
        file++;
    }

    /* Up-Left */
    rank = source_rank - 1;
    file = source_file - 1;
    while (rank >= 0 && file >= 0) {
        result |= UINT64_C(1) << (file + rank * 8);
        if (occ & (UINT64_C(1) << (file + rank * 8)))
            break;
        rank--;
        file--;
    }

    return result;
}

/**
 * Maps an index to the bits of an occupancy mask.
 * Effectively allows you to map an n-bit number to an n-bit mask.
 */
uint64_t map_index_to_occ_mask(uint16_t idx, uint8_t num_bits, uint64_t occ_mask)
{
    uint64_t result = 0;
    uint8_t pos;
    uint8_t i;

    /* Loop over all of the bits in the occupancy mask */
    for (i = 0; i < num_bits; i++) {
        /* Get the position of the bit in the occ mask. */
        pos = pop_rbit(&occ_mask);
        /* If it lines up with the current bit in the index, then place a bit in the result. */
        if (idx & (1 << i)) {
            result |= UINT64_C(1) << pos;
        }
    }

    return result;
}

/**
 * Generates the bishop table.
 */
int gen_bishop_table()
{
    int result;
    int sq, idx;
    uint64_t *table;
    uint64_t occ;

    uint16_t key;
    uint64_t legal_moves;
    uint64_t occupied_squares;

    /* Loop over all squares and compute the corresponding table. */
    for (sq = 0; sq < 64; sq++) {
        /* Allocate space for the table on the heap. */
        if ((table = calloc(1 << NUM_BISHOP_BITS[sq], sizeof(uint64_t))) == 0) {
            result = ENOMEM;
            goto out_free_tables;
        };

        /* Generate the corresponding occupancy mask. */
        occ = get_bishop_occ_mask(sq);
        bishop_occ_mask[sq] = occ;

        /* Loop over all possible occupancies and generate the correct attack sets. */
        for (idx = 0; idx < (1 << NUM_BISHOP_BITS[sq]); idx++) {
            /* Generate the relevant masks and key for the current index. */
            occupied_squares = map_index_to_occ_mask(idx, NUM_BISHOP_BITS[sq], occ);
            legal_moves = get_bishop_atk_mask(sq, occupied_squares);
            key = get_bishop_key(sq, occupied_squares);

            /* DEBUG: Catch any evil hash collisions. */
            assert(table[key] == 0 || table[key] != legal_moves);

            /* Write data into the table. */
            table[key] = legal_moves;
        }

        /* Save the address of the allocated table. */
        bishop_atks[sq] = table;
    }

    /* We have successfully generated tables for all of the squares. Exit out. */
    result = 0;
    goto out_success;

out_free_tables:
    /* Unwind the calloc stack. */
    while (sq > 0) {
        free(bishop_atks[--sq]);
    }
out_success:
    return result;
}

/**
 * Generates the rook table.
 */
int gen_rook_table()
{
    int result;
    int sq, idx;
    uint64_t *table;
    uint64_t occ;

    uint16_t key;
    uint64_t legal_moves;
    uint64_t occupied_squares;

    /* Loop over all squares and compute the corresponding table. */
    for (sq = 0; sq < 64; sq++) {
        /* Allocate space for the table on the heap. */
        if ((table = calloc(1 << NUM_ROOK_BITS[sq], 64)) == NULL) {
            result = ENOMEM;
            goto out_free_tables;
        };

        /* Generate the corresponding occupancy mask. */
        occ = get_rook_occ_mask(sq);
        rook_occ_mask[sq] = occ;

        /* Loop over all possible occupancies and generate the correct attack sets. */
        for (idx = 0; idx < (1 << NUM_ROOK_BITS[sq]); idx++) {
            /* Generate the relevant masks and key for the current index. */
            occupied_squares = map_index_to_occ_mask(idx, NUM_ROOK_BITS[sq], occ);
            legal_moves = get_rook_atk_mask(sq, occupied_squares);
            key = get_rook_key(sq, occupied_squares);

            /* DEBUG: Catch any evil hash collisions. */
            assert(table[key] == 0 || table[key] != legal_moves);

            /* Write data into the table. */
            table[key] = legal_moves;
        }

        /* Save the address of the allocated table. */
        rook_atks[sq] = table;
    }

    /* We have successfully generated tables for all of the squares. Exit out. */
    result = 0;
    goto out_success;

out_free_tables:
    /* Unwind the calloc stack. */
    while (sq > 0) {
        free(rook_atks[--sq]);
    }
out_success:
    return result;
}

void cleanup_bishop_tables()
{
    int sq;
    for (sq = 0; sq < 64; sq++)
        free(bishop_atks[sq]);
}

void cleanup_rook_tables()
{
    int sq;
    for (sq = 0; sq < 64; sq++)
        free(rook_atks[sq]);
}

int cb_init_magic_tables()
{
    int result;

    /* The following functions call calloc so they can err. */
    if ((result = gen_bishop_table()) < 0)
        goto out_no_cleanup;
    if ((result = gen_rook_table()) < 0)
        goto out_cleanup_bishop;
    goto out_no_cleanup;

out_cleanup_bishop:
    cleanup_bishop_tables();
out_no_cleanup:
    return result;
}

void cb_free_magic_tables()
{
    cleanup_bishop_tables();
    cleanup_rook_tables();
}
