
#ifndef GPU_RAY_GEN_H
#define GPU_RAY_GEN_H

#include <stdint.h>
#include "gpu_const.cuh"

/**
 * @breif Enum defining directions of rays on the board.
 *
 * Starts with 0 being right and goes counterclockwize.
 *
 *      3 2 1
 *      4 X 0
 *      5 6 7
 *
 * CB_DIF_UNION is the index of the union mask in the PIN set.
 * CB_DIR_INVALID is used for things that are not rays.
 */
typedef enum {
    GPU_DIR_R  = 0,
    GPU_DIR_UR = 1,
    GPU_DIR_U  = 2,
    GPU_DIR_UL = 3,
    GPU_DIR_L  = 4,
    GPU_DIR_DL = 5,
    GPU_DIR_D  = 6,
    GPU_DIR_DR = 7,
    GPU_DIR_UNION = 8,
    GPU_DIR_INVALID = 9
} gpu_dir_t;

/* Get the direction of a ray. */
__device__ static inline uint8_t gpu_get_ray_direction(uint8_t sq1, uint8_t sq2)
{
    int8_t sq1_rank = sq1 / 8;
    int8_t sq1_file = sq1 % 8;
    int8_t sq2_rank = sq2 / 8;
    int8_t sq2_file = sq2 % 8;
    uint8_t direction;

    /* This line is a mess, but it does convert from square and file to ray direction. */
    direction = (sq1_rank == sq2_rank) ? (sq1 < sq2 ? GPU_DIR_R : GPU_DIR_L) :
        (sq1_file == sq2_file) ? (sq1 < sq2 ? GPU_DIR_D : GPU_DIR_U) :
        (sq1_file + sq1_rank == sq2_file + sq2_rank) ? (sq1 < sq2 ? GPU_DIR_DL : GPU_DIR_UR) :
        (sq1_file - sq1_rank == sq2_file - sq2_rank) ? (sq1 < sq2 ? GPU_DIR_DR : GPU_DIR_UL) :
        GPU_DIR_INVALID;

    return direction;
}

/* Empty board ray generation. */
__device__ static inline uint64_t gpu_east_ray(uint64_t gen)
{
    const uint64_t pr0 = ~BB_LEFT_COL;
    const uint64_t pr1 = pr0 & (pr0 << 1);
    const uint64_t pr2 = pr1 & (pr1 << 2);
    gen |= pr0 & (gen << 1);
    gen |= pr1 & (gen << 2);
    gen |= pr2 & (gen << 4);
    return gen;
}

__device__ static inline uint64_t gpu_north_east_ray(uint64_t gen)
{
    const uint64_t pr0 = ~BB_LEFT_COL;
    const uint64_t pr1 = pr0 & (pr0 >>  7);
    const uint64_t pr2 = pr1 & (pr1 >> 14);
    gen |= pr0 & (gen >>  7);
    gen |= pr1 & (gen >> 14);
    gen |= pr2 & (gen >> 28);
    return gen;
}

__device__ static inline uint64_t gpu_north_ray(uint64_t gen)
{
    gen |= (gen >>  8);
    gen |= (gen >> 16);
    gen |= (gen >> 32);
    return gen;
}

__device__ static inline uint64_t gpu_north_west_ray(uint64_t gen)
{
    const uint64_t pr0 = ~BB_RIGHT_COL;
    const uint64_t pr1 = pr0 & (pr0 >>  9);
    const uint64_t pr2 = pr1 & (pr1 >> 18);
    gen |= pr0 & (gen >>  9);
    gen |= pr1 & (gen >> 18);
    gen |= pr2 & (gen >> 36);
    return gen;
}

__device__ static inline uint64_t gpu_west_ray(uint64_t gen)
{
    const uint64_t pr0 = ~BB_RIGHT_COL;
    const uint64_t pr1 = pr0 & (pr0 >> 1);
    const uint64_t pr2 = pr1 & (pr1 >> 2);
    gen |= pr0 & (gen >> 1);
    gen |= pr1 & (gen >> 2);
    gen |= pr2 & (gen >> 4);
    return gen;
}

__device__ static inline uint64_t gpu_south_west_ray(uint64_t gen)
{
    const uint64_t pr0 = ~BB_RIGHT_COL;
    const uint64_t pr1 = pr0 & (pr0 <<  7);
    const uint64_t pr2 = pr1 & (pr1 << 14);
    gen |= pr0 & (gen <<  7);
    gen |= pr1 & (gen << 14);
    gen |= pr2 & (gen << 28);
    return gen;
}

__device__ static inline uint64_t gpu_south_ray(uint64_t gen)
{
    gen |= (gen <<  8);
    gen |= (gen << 16);
    gen |= (gen << 32);
    return gen;
}

__device__ static inline uint64_t gpu_south_east_ray(uint64_t gen)
{
    const uint64_t pr0 = ~BB_LEFT_COL;
    const uint64_t pr1 = pr0 & (pr0 <<  9);
    const uint64_t pr2 = pr1 & (pr1 << 18);
    gen |= pr0 & (gen <<  9);
    gen |= pr1 & (gen << 18);
    gen |= pr2 & (gen << 36);
    return gen;
}

/* Occluded ray generation. */
__device__ static inline uint64_t gpu_east_ray_occ(uint64_t gen, uint64_t pro)
{
    pro &= ~BB_LEFT_COL;
    gen |= pro & (gen << 1);
    pro &=       (pro << 1);
    gen |= pro & (gen << 2);
    pro &=       (pro << 2);
    gen |= pro & (gen << 4);
    return gen;
}

__device__ static inline uint64_t gpu_north_east_ray_occ(uint64_t gen, uint64_t pro)
{
    pro &= ~BB_LEFT_COL;
    gen |= pro & (gen >>  7);
    pro &=       (pro >>  7);
    gen |= pro & (gen >> 14);
    pro &=       (pro >> 14);
    gen |= pro & (gen >> 28);
    return gen;
}

__device__ static inline uint64_t gpu_north_ray_occ(uint64_t gen, uint64_t pro)
{
    gen |= pro & (gen >>  8);
    pro &=       (pro >>  8);
    gen |= pro & (gen >> 16);
    pro &=       (pro >> 16);
    gen |= pro & (gen >> 32);
    return gen;
}

__device__ static inline uint64_t gpu_north_west_ray_occ(uint64_t gen, uint64_t pro)
{
    pro &= ~BB_RIGHT_COL;
    gen |= pro & (gen >>  9);
    pro &=       (pro >>  9);
    gen |= pro & (gen >> 18);
    pro &=       (pro >> 18);
    gen |= pro & (gen >> 36);
    return gen;
}

__device__ static inline uint64_t gpu_west_ray_occ(uint64_t gen, uint64_t pro)
{
    pro &= ~BB_RIGHT_COL;
    gen |= pro & (gen >> 1);
    pro &=       (pro >> 1);
    gen |= pro & (gen >> 2);
    pro &=       (pro >> 2);
    gen |= pro & (gen >> 4);
    return gen;
}

__device__ static inline uint64_t gpu_south_west_ray_occ(uint64_t gen, uint64_t pro)
{
    pro &= ~BB_RIGHT_COL;
    gen |= pro & (gen <<  7);
    pro &=       (pro <<  7);
    gen |= pro & (gen << 14);
    pro &=       (pro << 14);
    gen |= pro & (gen << 28);
    return gen;
}

__device__ static inline uint64_t gpu_south_ray_occ(uint64_t gen, uint64_t pro)
{
    gen |= pro & (gen <<  8);
    pro &=       (pro <<  8);
    gen |= pro & (gen << 16);
    pro &=       (pro << 16);
    gen |= pro & (gen << 32);
    return gen;
}

__device__ static inline uint64_t gpu_south_east_ray_occ(uint64_t gen, uint64_t pro)
{
    pro &= ~BB_LEFT_COL;
    gen |= pro & (gen <<  9);
    pro &=       (pro <<  9);
    gen |= pro & (gen << 18);
    pro &=       (pro << 18);
    gen |= pro & (gen << 36);
    return gen;
}

__device__ static inline uint64_t gpu_ray_through_sq2(uint8_t sq1, uint8_t sq2)
{
    uint8_t dir = gpu_get_ray_direction(sq1, sq2);
    switch (dir) {
        case GPU_DIR_R:
            return gpu_east_ray(UINT64_C(1) << sq1);
        case GPU_DIR_UR:
            return gpu_north_east_ray(UINT64_C(1) << sq1);
        case GPU_DIR_U:
            return gpu_north_ray(UINT64_C(1) << sq1);
        case GPU_DIR_UL:
            return gpu_north_west_ray(UINT64_C(1) << sq1);
        case GPU_DIR_L:
            return gpu_west_ray(UINT64_C(1) << sq1);
        case GPU_DIR_DL:
            return gpu_south_west_ray(UINT64_C(1) << sq1);
        case GPU_DIR_D:
            return gpu_south_ray(UINT64_C(1) << sq1);
        case GPU_DIR_DR:
            return gpu_south_east_ray(UINT64_C(1) << sq1);
        default:
            return 0;
    }
}

__device__ static inline uint64_t gpu_east_atk(uint64_t rooks, uint64_t occ)
{
    return (gpu_east_ray_occ(rooks, ~occ) << 1) & ~BB_LEFT_COL;
}

__device__ static inline uint64_t gpu_north_atk(uint64_t rooks, uint64_t occ)
{
    return gpu_north_ray_occ(rooks, ~occ) >> 8;
}

__device__ static inline uint64_t gpu_west_atk(uint64_t rooks, uint64_t occ)
{
    return (gpu_west_ray_occ(rooks, ~occ) >> 1) & ~BB_RIGHT_COL;
}

__device__ static inline uint64_t gpu_south_atk(uint64_t rooks, uint64_t occ)
{
    return gpu_south_ray_occ(rooks, ~occ) << 8;
}

__device__ static inline uint64_t gpu_north_east_atk(uint64_t bishops, uint64_t occ)
{
    return (gpu_north_east_ray_occ(bishops, ~occ) >> 7) & ~BB_LEFT_COL;
}

__device__ static inline uint64_t gpu_north_west_atk(uint64_t bishops, uint64_t occ)
{
    return (gpu_north_west_ray_occ(bishops, ~occ) >> 9) & ~BB_RIGHT_COL;
}

__device__ static inline uint64_t gpu_south_west_atk(uint64_t bishops, uint64_t occ)
{
    return (gpu_south_west_ray_occ(bishops, ~occ) << 7) & ~BB_RIGHT_COL;
}

__device__ static inline uint64_t gpu_south_east_atk(uint64_t bishops, uint64_t occ)
{
    return (gpu_south_east_ray_occ(bishops, ~occ) << 9) & ~BB_LEFT_COL;
}

#endif /* GPU_RAY_GEN_H */

