
- move count kernel verified to work for root position with 1 thread at least.
- Looks like castle rights decayed for no reason
- pbfs works for the first move
  - I'm likely not clearing board vector after every search.
  - This does raise the question of why I start capping out.
- TODO: Fix the scan, you screwed it up pretty bad
