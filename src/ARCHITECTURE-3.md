
# Detailed Architecture for Depth First Search on the GPU

## Board Representation

My vote is on this board representation. Minimal packing is always good.
The occupancy bitmask is, however, too useful to pass up.

```c
typedef struct {
    uint64_t color;         /**< Bitmasks for colored pieces. */
    uint64_t piece[5];      /**< A set of bitmasks for piece types. */
    uint64_t occ;           /**< Bitmask for occupancy the chessboard. */
} gpu_bitboard_t;
```

## Search Strategy / Move List Storage / Handling Divergence / Memory Bandwidth

The key to this implementation will be reducing global memory bandwidth for
move storage and the like and reducing divergence between threads during the
tree search. Divergence is a price that we will unfortunately have to pay.
The tree of possible chess positions is necessarily unbalanced. I am just
hoping that divergence averages out at about 50%. I can take a 50% performance
hit, maybe.

- I think that implementing unmake will cause so much divergence that it won't
be worth doing.
    - We should just re-read the board whenever we unmake a move.

## Handling Divergence

Depth first search on the GPU will always be at risk of having bad divergence,
this is the cost of doing the search in the order that we are doing it. Depth
first search has the drawback of requiring lots and lots of 

## Handling Global Memory Bandwidth Limitations

