
#ifndef GPU_TABLES_H
#define GPU_TABLES_H

/* TODO: Remove me. */
#include <inttypes.h>

#include "gpu_const.cuh"
#include "gpu_tables.cuh"

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

__constant__ uint64_t d_knight_atks[64];
__constant__ uint64_t d_king_atks[64];
__constant__ uint64_t d_bishop_occ_mask[64];
__constant__ uint64_t d_rook_occ_mask[64];
__constant__ uint8_t d_num_bishop_bits[64];
__constant__ uint8_t d_num_rook_bits[64];
__constant__ uint64_t d_bishop_magics[64];
__constant__ uint64_t d_rook_magics[64];
__constant__ uint64_t *d_bishop_atks[64];
__constant__ uint64_t *d_rook_atks[64];
__constant__ uint64_t d_to_from_table[64][64];
__constant__ uint64_t d_bishop_no_occ[64];
__constant__ uint64_t d_rook_no_occ[64];

extern uint64_t knight_atks[64];
extern uint64_t king_atks[64];
extern uint64_t to_from_table[64][64];

extern const uint8_t NUM_BISHOP_BITS[64];
extern const uint8_t NUM_ROOK_BITS[64];
extern uint64_t bishop_occ_mask[64];
extern uint64_t rook_occ_mask[64];
extern uint64_t *bishop_atks[64];
extern uint64_t *rook_atks[64];
extern const uint64_t BISHOP_MAGICS[64];
extern const uint64_t ROOK_MAGICS[64];
extern uint64_t bishop_no_occ[64];
extern uint64_t rook_no_occ[64];

extern uint64_t *gpu_bishop_atk_ptrs_h[64];
extern uint64_t *gpu_rook_atk_ptrs_h[64];

static inline void gpu_init_tables()
{
    cudaMemcpyToSymbol(d_knight_atks, knight_atks, 64 * sizeof(uint64_t));
    cudaMemcpyToSymbol(d_king_atks, king_atks, 64 * sizeof(uint64_t));
    cudaMemcpyToSymbol(d_bishop_occ_mask, bishop_occ_mask, 64 * sizeof(uint64_t));
    cudaMemcpyToSymbol(d_rook_occ_mask, rook_occ_mask, 64 * sizeof(uint64_t));
    cudaMemcpyToSymbol(d_num_bishop_bits, NUM_BISHOP_BITS, 64 * sizeof(uint8_t));
    cudaMemcpyToSymbol(d_num_rook_bits, NUM_ROOK_BITS, 64 * sizeof(uint8_t));
    cudaMemcpyToSymbol(d_bishop_magics, BISHOP_MAGICS, 64 * sizeof(uint64_t));
    cudaMemcpyToSymbol(d_rook_magics, ROOK_MAGICS, 64 * sizeof(uint64_t));
    cudaMemcpyToSymbol(d_to_from_table, to_from_table, 64 * 64 * sizeof(uint64_t));
    cudaMemcpyToSymbol(d_bishop_no_occ, bishop_no_occ, 64 * sizeof(uint64_t));
    cudaMemcpyToSymbol(d_rook_no_occ, rook_no_occ, 64 * sizeof(uint64_t));

    for (int sq = 0; sq < 64; sq++) {
        cudaMalloc((void**)&gpu_bishop_atk_ptrs_h[sq],
                (1 << NUM_BISHOP_BITS[sq]) * sizeof(uint64_t));
        cudaMemcpy(gpu_bishop_atk_ptrs_h[sq], bishop_atks[sq],
                (1 << NUM_BISHOP_BITS[sq]) * sizeof(uint64_t), cudaMemcpyHostToDevice);
    }
    cudaMemcpyToSymbol(d_bishop_atks, gpu_bishop_atk_ptrs_h, 64 * sizeof(uint64_t*));

    for (int sq = 0; sq < 64; sq++) {
        cudaMalloc((void**)&gpu_rook_atk_ptrs_h[sq],
                (1 << NUM_ROOK_BITS[sq]) * sizeof(uint64_t));
        cudaMemcpy(gpu_rook_atk_ptrs_h[sq], rook_atks[sq],
                (1 << NUM_ROOK_BITS[sq]) * sizeof(uint64_t), cudaMemcpyHostToDevice);
    }
    cudaMemcpyToSymbol(d_rook_atks, gpu_rook_atk_ptrs_h, 64 * sizeof(uint64_t*));
}

static inline void gpu_free_tables()
{
    for (int sq = 0; sq < 64; sq++) {
        cudaFree(gpu_bishop_atk_ptrs_h[sq]);
    }
    for (int sq = 0; sq < 64; sq++) {
        cudaFree(gpu_rook_atk_ptrs_h[sq]);
    }
}

__device__ static inline uint64_t gpu_read_bishop_atk_msk(uint8_t sq, uint64_t occ)
{
    occ &= d_bishop_occ_mask[sq];
    occ *= d_bishop_magics[sq];
    uint16_t key = occ >> (64 - d_num_bishop_bits[sq]);
    return d_bishop_atks[sq][key];
}

__device__ static inline uint64_t gpu_read_rook_atk_msk(uint8_t sq, uint64_t occ)
{
    occ &= d_rook_occ_mask[sq];
    occ *= d_rook_magics[sq];
    uint16_t key = occ >> (64 - d_num_rook_bits[sq]);
    return d_rook_atks[sq][key];
}

__device__ static inline uint64_t gpu_read_knight_atk_msk(uint8_t sq)
{
    return d_knight_atks[sq];
}

__device__ static inline uint64_t gpu_read_king_atk_msk(uint8_t sq)
{
    return d_king_atks[sq];
}

__device__ static inline uint64_t gpu_read_tf_table(uint8_t sq1, uint8_t sq2)
{
    return d_to_from_table[sq1][sq2];
}

__device__ static inline uint64_t gpu_read_bishop_no_occ(uint8_t sq)
{
    return d_bishop_no_occ[sq];
}

__device__ static inline uint64_t gpu_read_rook_no_occ(uint8_t sq)
{
    return d_rook_no_occ[sq];
}

#endif /* GPU_TABLES_H */

