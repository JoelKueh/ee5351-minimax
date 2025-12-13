
#ifndef CB_BITUTIL_H
#define CB_BITUTIL_H

#include <stdint.h>

/* Non-hardware implementations taken from https://www.chessprogramming.org/BitScan. */

static inline uint8_t peek_rbit(uint64_t bb)
{
#if __has_builtin (__builtin_ctzl)
    return __builtin_ctzl(bb);
#else
    const int index64[64] = {
        0,  1, 48,  2, 57, 49, 28,  3,
       61, 58, 50, 42, 38, 29, 17,  4,
       62, 55, 59, 36, 53, 51, 43, 22,
       45, 39, 33, 30, 24, 18, 12,  5,
       63, 47, 56, 27, 60, 41, 37, 16,
       54, 35, 52, 21, 44, 32, 23, 11,
       46, 26, 40, 15, 34, 20, 31, 10,
       25, 14, 19,  9, 13,  8,  7,  6
    };

    const U64 debruijn64 = C64(0x03f79d71b4cb0a89);
    assert (bb != 0);
    return index64[((bb & -bb) * debruijn64) >> 58];
#endif
}

static inline uint8_t pop_rbit(uint64_t *bb)
{
    uint8_t idx = peek_rbit(*bb);
    *bb ^= UINT64_C(1) << idx;
    return idx;
}

static inline uint8_t popcnt(uint64_t bb)
{
#if __has_builtin (__builtin_popcountl)
    return __builtin_popcountl(bb);
#else
    int count = 0;
    while (x) {
        count++;
        x &= x - 1; // reset LS1B
    }
    return count;
#endif
}

#endif /* CB_BITUTIL_H */
