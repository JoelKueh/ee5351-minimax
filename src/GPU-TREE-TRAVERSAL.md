
# GPU Tree Traversal

I think that it makes sense to have data parallelism be across boards. Each
thread should perform traversal from a single board (keeping track of the board
state and moves in its registers. This heavily limits our representation to
things that can be stored in a very small space (if we want full utilization
then we can only have 32 registers per thread).

A full bitboard is structured as follows

```c
typedef struct {
    uint64_t color[2];      /**< A set of bitmasks for colored pieces. */
    uint64_t piece[2][6];   /**< A set of bitmasks for piece types and colors. */
    uint64_t occ;           /**< The union of the above bitmasks. For occupied squares. */
} cb_bitboard_t;
```

This has a total memory requirement of (2 + 2 * 6 + 1) * 2 = 30 registers. With
just this

We will attempt to have each thread in the board work on one position. We can
then store 
