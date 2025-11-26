
List of notes while I try to fix move generation.

- First fix initial move generation of all other pieces.
  - Fact that you get 20 in the initial position is really a coincidence.
  - This was partly caused by me copying the bishop atks into the rook positions. Pawn moves are screwed up.
- Looks like a pawn is turning into a knight after the first move, why is this?
  - Had to do with the way that I was handling promotions
