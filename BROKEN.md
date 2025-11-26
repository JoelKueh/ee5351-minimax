
List of notes while I try to fix move generation.

- First fix initial move generation of all other pieces.
  - Fact that you get 20 in the initial position is really a coincidence.
  - This was partly caused by me copying the bishop atks into the rook positions. Pawn moves are screwed up.
- Looks like a pawn is turning into a knight after the first move, why is this?
  - Had to do with the way that I was handling promotions
- Looks like things are being detected as pinned when they shouldn't be
- Move generation for pinned pawns might not work as it should. Specifically black pawns.
- Theat generation is slightly off.
  - Problem was kings moves not being added to threat mask.
  - Threat generation is still cooked.
    - Wait no I think its good
- **Pinned piece generation is very off.**
  - May have fixed it. Might still be broken though.
- Move generation
- Captures are broken.
  - Looks like the problem is with unmake.
- Kings are being duplicated, this might be a sign that kings are being taken.
