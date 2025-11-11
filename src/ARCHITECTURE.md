
# Introduction

GPU-based chess engines have exhibited somewhat mediocre performance when
compared to modern CPU-based implementation. This is primarily due to the
effectiveness of the heuristic-based and necessarily sequential alpha-beta
search. Engines like [Stockfish](https://github.com/official-stockfish/Stockfish)
are able to search through game trees to impressive depths because they are
very good at pruning the game tree, removing paths that they can very quickly
determine will never be taken assuming perfect play by both sides. For this
reason, CPU-based have dominated the chess engine scene and will likely
continue to do so.

GPUs, however, still have an edge over CPUs in one specific area of the game.
GPUs are useful in the area of chess theory and investigation because of their
ability to quickly, exhaustively search the game tree and discover lines that
CPU-based engines simply don't have the time and power to find. GPU-based chess
engines have been used to explore positions to incredible depths. With a fast,
sensible, GPU implementation of a move generator, we should be able to generate
move counts at higher depths for some of the 'low-hanging fruit' positions on
[Perft Results](https://www.chessprogramming.org/Perft_Results). Additionally,
by implementing a simple position evaluation function and a minimax search,
we should be able to create a mediocre chess engine that suggests optimal
moves by exhaustively searching the game tree to a reasonable depth. Our
performance metric will therefore be the standard "Leaf Nodes Per Second"
used in Perft move generation in the page discussed above. If we can get
performance faster than 40 million nodes per second, we will have beaten my
personal CPU implementation, and I will consider that a success.

# Implementation

## Board Representation

The standard CPU algorithms for Chess move generation make use of bitboards
for their efficient use in move generation (see
[Bitboards](https://www.chessprogramming.org/Bitboards)). A standard CPU
bitboard-based board representation looks something like this.

```c
/**
 * @breif Bitboard data structure that actually stores peice data.
 *
 * The board squares are counted from the top left of the board (black pieces first) row-wise.
 * E.g. if a black rook is on rank 6 file A, then:
 *
 *      piece[CB_BLACK][CB_PTYPE_ROOK] & (UINT64_T(1) << 16) == 1
 */
typedef struct {
    uint64_t color[2];      /**< A set of bitmasks for colored pieces. */
    uint64_t piece[2][6];   /**< A set of bitmasks for piece types and colors. */
    uint64_t occ;           /**< The union of the above bitmasks. For occupied squares. */
} cb_bitboard_t;
```

In CPU implementations, we make use of large memory capabilities to avoid doing
redundant work wherever possible. Sometimes even including duplicate data
structures to reduce number of operations during move generation at the cost
of additional memory and a slight cost during incremental updates.

```c
/**
 * @breif Mailbox data structure that duplicates data of bitboard.
 *
 * As stated on the programming wiki, bitboards are great at answering questions like "Where
 * are the white knights?" but struggles to answer questions like "What piece is on this square?"
 *
 * Maintaining this represenation improves efficienty of move generation and application.
 */
typedef struct {
    uint8_t data[64]; /* This is an array of cb_ptype_t but it is stored as a uint8_t. */
} cb_mailbox_t;
```

GPU-based implementations do not have this luxury. The more data you have in
your board representation, the more data you have to read from global memory.
To reduce memory usage, I suggest that we drop the "mailbox" representation
entirely, and cut redundant information from the bitboard representation
wherever possible. As suggested in the chess programming wiki, we can save
space by no longer including piece bitboards for both sides. We can also remove
one of the color bitboards because all white pieces are not black. We can also
remove the occupancy bitboard because it is the union of all piece bitboards.
Finally, we can remove the queen bitboard by declaring that queens are rooks
and bishops that share the same square.

```c
typedef struct {
    uint64_t color;         /**< Bitmasks for colored pieces. */
    uint64_t piece[5];      /**< A set of bitmasks for piece types. */
} gpu_bitboard_t;
```

## Search Strategy

Some preliminary research suggests that a breath-first search should minimize
divergence and maximize coalescence. I tend to agree with this notion. We can
search the tree depth first on the CPU, pushing board positions to a large
buffer. Aiming for 1 to 10 Million nodes in this buffer (depth of 5) seems like
it should be an effective goal. We can then take chunks off of this array in
groups of ~1000 and expand their subtree on the GPU. This will require a
recursive search like we did with the recursive reduction.

On the GPU, we will generate a vector of ~1000 * ~20 moves that we can explore
further. We take that vector and generate a vector of ~20000 board positions.
We take that vector and generate a vector of ~20000 * ~20 moves. This continues
until we reach our desired depth (we hope that we don't run out of GPU memory
before we reach that point). With our board encoding, we should be able to
store 8 GiB of board positions (or 166 million positions). This will be at
about a depth of 3 from the vector of 1000 moves that we started with. We
should be able to squeeze out one or two more levels of move counts from here
by only skipping the phase where we write move counts to memory. This should
give us a total depth of about 9 half moves relatively quickly. We could
increase this to 10 or 11 by adjusting the GPU launch depth, searches to this
depth might take an incredibly long time, though they would be possible.

## Move Generation

I believe that lookups from global memory aided by the read-only texture cache
will result in the best performance for move generation (see
[Magic Bitboards](https://www.chessprogramming.org/Magic_Bitboards)).
This might have a high memory bottleneck, the sliding Kogge-Stone fill could
result in better performance (see
[Kogge-Stone](https://www.chessprogramming.org/Kogge-Stone_Algorithm)).

We can use the standard move encoding to fit moves into two byte integers.
We need to store the target and source squares (6-bit each). That leaves 4
bits for flags.

## Move Making

This should be fairly easy, we can copy the CPU implementation here, we just
need to do some different work when we build the board position. When writing
back positions to global memory, we can try for good coalescence by writing
the positions in a good ordering (write the struct field by field for each
thread).

## Move Counting

To allocate the right amount of memory for level 2 starting at level 1, we will
need to count the required number of moves ahead of time. We will have to
implement separate move counting and making functions. There will be a penalty
associated with this, but it should not be all that significant as we do not
have to do not have to store moves at the last layer (we only have to count
them).

## Move Legalization

We may have to slightly modify the CPU paradigm for detecting pinned pieces
and guaranteeing we only make legal moves.

## Enpassant

Enpassant is slow, but rare. We do not need to optimize for it and can
implement it with slow methods. A similar thing can be said for castleing.

