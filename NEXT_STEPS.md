
# Next Steps

Below are a list of tasks that need to be completed to speed up the now working
move generation code.

## Multi-Threaded Depth First Search Kernel

This is definitely the most important part to get done. See the function
`perft_gpu_slow_kernel()` for reference. The function should take a vector of
board positions and a pointer to GPU heap memory filled with the proper number
of `gpu_search_struct_node_t`s and output a vector of `uint64_t` counts of all
LEAF NODES (internal nodes should not be counted, just add one to the count
every time you reach the maximum depth as I do in my kernel.

## Interval-Move Expand Kernel

This is a kernel that we will need in the future. It can be developed totally
independently of the rest of the code and requires absolutely no knowledge
of chess. As a reference, you can find an implementation of the function at
[Perft GPU](https://github.com/ankan-ban/perft_gpu/blob/6d58bf23ae627805eb2a3de4a436e7df80ea250e/moderngpu-master/include/kernels/intervalmove.cuh#L48).
Otherwise, I believe this is what the kernel does.

To quote myself on discord: "It takes an array of values [0, 1, 2, 3, 4] and an
array of numbers [3, 2, 1, 3, 3] and creates duplicates of the numbers in values
based on the numbers in numbers (e.g. [0, 0, 0, 1, 1, 2, 3, 3, 3, 4, 4, 4]).

## Other Move Generation Related Kernels

To turn our depth first search into a breadth first search, we will need the
following kernels to be implemented.

1. `apply_moves(gpu_board_t *new_boards, gpu_board_t *boards, gpu_move_t *moves, uint32_t *indices)`
  - Takes a vector of moves and applies them on a vector of boards.
  - Each move in moves has an index in indices which tells you which board in
  boards the move should apply to. For reference, this vector of moves will be
  produced by the interval-move expand kernel.
  - Once again, `perft_gpu_slow_kernel()` in `perft_gpu.cu` for reference.
2. `count_moves(uint8_t *counts, gpu_board_t *boards)`
  - Returns the number of moves that can be made from any of the positions in boards.
  - Index in counts should line up with the index in boards.
  - This is guaranteed to be no greater than 218, so a `uint8_t` is enough.
3. `generate_moves(gpu_move_t *moves, gpu_board_t *boards, uint32_t *indices)`
  - Generates a vector of moves given a vector of boards.
  - Each thread needs to know where it has to write its output moves. This is
  determined by the `indices` vector with is the scanned results from the
  `count_moves` kernel above. I think Sartori mentioned this sort of
  application in his slides on uses for scan, so that's intersting I guess.

Apart from these, we will need an implementation of a scan and the
Interval-Move Expand kernel mentioned above.
