
#ifndef GPU_TABLES_H
#define GPU_TABLES_H

#include <stdint.h>
#include <stdbool.h>
#include "cb_types.h"

/**
 * @breif Initializes gpu copies of tables. Copies them to texture memory.
 */
__device__ void gpu_init_magic_tables();

/**
 * @breif Frees gpu copies of tables.
 */
__device__ void gpu_free_magic_tables();

/**
 * @breif Reads the bishop attack masks
 * @param sq The square.
 * @param occ The relevant occupancy mask.
 * @return A bitmask representing the attacked squares.
 */
__device__ uint64_t gpu_read_bishop_atk_msk(uint8_t sq, uint64_t occ);

/**
 * @breif Reads the rook attack masks
 * @param sq The square.
 * @param occ The relevant occupancy mask.
 * @return A bitmask representing the attacked squares.
 */
__device__ uint64_t gpu_read_rook_atk_msk(uint8_t sq, uint64_t occ);

/**
 * @breif Reads the pawn attack masks
 * @param sq The square.
 * @param color The color of the attacking pawn.
 * @return A bitmask representing the attacked squares.
 */
__device__ uint64_t gpu_read_pawn_atk_msk(uint8_t sq, cb_color_t color);

/**
 * @breif Reads the knight attack masks
 * @param sq The square.
 * @return A bitmask representing the attacked squares.
 */
__device__ uint64_t gpu_read_knight_atk_msk(uint8_t sq);

/**
 * @breif Reads the king attack masks
 * @param sq The square.
 * @return A bitmask representing the attacked squares.
 */
__device__ uint64_t gpu_read_king_atk_msk(uint8_t sq);

/**
 * @breif Handles lookups for the ray that connects two squares
 *
 * The ray starts from sq1 and extends up to but not including sq1.
 * If sq1 and sq2 are not on a ray, returns 0.
 *
 * @param sq1 The starting square.
 * @param sq2 The ending square.
 * @return Either the ray connecting the squares or zero.
 */
__device__ uint64_t gpu_read_tf_table(uint8_t sq1, uint8_t sq2);

#endif /* GPU_TABLES_H */

