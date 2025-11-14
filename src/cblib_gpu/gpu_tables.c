
#include "gpu_tables.h"

/* TODO: Task 1: Implement gpu table read functions.
 *
 * This can likely just be implemented with reads from global memory.
 * You should use the __ldg() function to do the memory load so that
 * the compiler knows that it can cache it.
 *
 * You can assume that the following tables labeled "extern" will be
 * populated with the necessary data before any of these functions are
 * called. The CPU host code will do it.
 *
 * Copying the attack tables to the GPU will be a little bit weird, it is
 * an array of pointers. You will have to copy the data at each pointer
 * to the GPU.
 *
 * You should just be able to malloc space for the array using cudaMalloc
 * on the gpu very similar to this line from src/cblib/cb_magical.c
 *
 *    if ((table = calloc(1 << NUM_BISHOP_BITS[sq], sizeof(uint64_t))) == 0) {
 *        result = ENOMEM;
 *        goto out_free_tables;
 *    };
 *
 * You can then copy the data from the CPU to the GPU using a cudaMemcpy.
 *
 * Most of the information that you need will just be in constant memory.
 * knight_atks, king_atks, etc.... You will probably want to create another
 * version of the bishop_atks[64] and rook_atks[64] tables before you
 * copy them to global memory (you will need to use the device pointers
 * malloced above.
 */

extern uint64_t knight_atks[64];
extern uint64_t king_atks[64];
extern uint64_t to_from_table[64][64];

extern const uint8_t NUM_BISHOP_BITS[64];
extern const uint8_t NUM_ROOK_BITS[64];
extern uint64_t bishop_occ_mask[64];
extern uint64_t rook_occ_mask[64];
extern uint64_t *bishop_atks[64];
extern uint64_t *rook_atks[64];

__device__ void gpu_init_tables()
{
    /* TODO: Impelment me! Lots of cudaMemcpyToSymbol and cudaMemcpy here. */
}

__device__ void gpu_free_tables()
{
    /* TODO: Impelment me! Lots of frees here. */
}

__device__ uint64_t gpu_read_bishop_atk_msk(uint8_t sq, uint64_t occ)
{
    /* TODO: Impelment me! */
    return 0;
}

__device__ uint64_t gpu_read_rook_atk_msk(uint8_t sq, uint64_t occ)
{
    /* TODO: Impelment me! */
    return 0;
}

__device__ uint64_t gpu_read_pawn_atk_msk(uint8_t sq, cb_color_t color)
{
    /* TODO: Impelment me! */
    return 0;
}

__device__ uint64_t gpu_read_knight_atk_msk(uint8_t sq)
{
    /* TODO: Impelment me! */
    return 0;
}

__device__ uint64_t gpu_read_king_atk_msk(uint8_t sq)
{
    /* TODO: Impelment me! */
    return 0;
}

__device__ uint64_t gpu_read_tf_table(uint8_t sq1, uint8_t sq2)
{
    /* TODO: Impelment me! */
    return 0;
}

