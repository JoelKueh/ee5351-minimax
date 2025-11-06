
#ifndef CB_BITUTIL_H
#define CB_BITUTIL_H

#include <stdint.h>

static inline uint8_t peek_rbit(uint64_t bb)
{
#if __has_builtin (__builtin_ctzl)
    return __builtin_ctzl(bb);
#else
#   error __builtin_ctzl not supported!
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
#   error __builtin_popcountl not supported!
#endif
}

#endif /* CB_BITUTIL_H */
