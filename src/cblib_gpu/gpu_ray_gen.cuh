
#ifndef GPU_RAY_GEN_H
#define GPU_RAY_GEN_H

#include <stdint.h>
#include "gpu_const.cuh"

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
    const uint64_t pr2 = pr1 & (pr1 >> 14);
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

#endif /* GPU_RAY_GEN_H */

