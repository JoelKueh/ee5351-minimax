
#ifndef CB_BITUTIL_H
#define CB_BITUTIL_H

#include <stdint.h>

/* Non-hardware implementations taken from https://www.chessprogramming.org/BitScan. */

static inline uint8_t peek_rbit(uint64_t bb)
{
    return __builtin_ctzl(bb);
}

static inline uint8_t pop_rbit(uint64_t *bb)
{
    uint8_t idx = peek_rbit(*bb);
    *bb ^= UINT64_C(1) << idx;
    return idx;
}

static inline uint8_t popcnt(uint64_t bb)
{
    return __builtin_popcountl(bb);
}

#endif /* CB_BITUTIL_H */
