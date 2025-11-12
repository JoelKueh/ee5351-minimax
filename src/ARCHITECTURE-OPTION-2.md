
# Differences from Architecture 1

Instead of performing a single breadth first search on the GPU, we could
simplify our implementation and reduce memory requirements by performing many
depth first searches on a vector of positions on the GPU. Once again, the CPU
would provide a vector of positions, and that position would be the root for
the tree search for a particular thread.

This results in a greatly simplified implementation (only requires a search
that returns a vector of integers and a reduction). Moreover, many of the
internal board representation and implementation details remain the same.
We are no longer limited by the size of our board representation in global
memory, but we are limited by size of the board representation in each threads
registers. Each thread has 32 general purpose 32-bit registers that they can
use for whatever they want. As each bitboard is 64-bits, each will take up
2 general purpose registers. Using the same representation as before, we use
6 bitboards (12 registers) for the board representation and then have 20
general purpose registers to do whatever we want with before we start running
into problems keeping each SM fully occupied.

The only difference from the first architecture is how we store moves. It is
widely accepted (but not proven, see
[Chess Positions](https://www.chessprogramming.org/Chess_Position)) that the
maximum number of moves reachable from any reachable chess position is 218.
We can fit the information for a single position and information for each board
in 256 bytes with room to spare. In total, if we start with a vector of 100000
positions for maximum occupancy we would require 100000 positions * 256 moves
per position * 2 bytes per move = 512MB (not all that much in the grand scheme
of things).
