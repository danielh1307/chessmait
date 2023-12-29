# Model vs Model

Now it's time for the models to compete against each other. Here's their knowledge base:

* They recognize valid chess positions.
* For any given position, they can identify all potential subsequent moves, understanding piece movements.
* They can identify checkmate positions and will always choose these without further evaluation.
* Beyond this, their knowledge is limited to what they've learned during training.

The models calculate one full move ahead, considering not just their next move, but also the opponent's possible
responses. They decide their move based on this comprehensive evaluation.
This implies that all the games are entirely deterministic; the moves remain consistent as the model's evaluations are
also deterministic.

## Results

| Position                   | White                     | Black                     | White +10 | Black +10 | White wins | Black Wins | Draw |
|----------------------------|---------------------------|---------------------------|-----------|-----------|------------|------------|------|
| Initial Position           | **graceful-glitter-166**  | smart-valley-6            |           |           | 1          |            |      |
| King's Gamit Accepted      | **graceful-glitter-166**  | smart-valley-6            |           |           | 1          |            |      |
| Evan's Gambit              | **graceful-glitter-166**  | smart-valley-6            |           |           | 1          |            |      |
| King and Pawn vs King      | graceful-glitter-166      | smart-valley-6            | 1         |           |            |            | 1    |
| Blackburne Shilling Gambit | **graceful-glitter-166**  | smart-valley-6            |           |           | 1          |            |      |
| Traxler Counterattack      | **graceful-glitter-166**  | smart-valley-6            |           |           | 1          |            |      |
|                            |                           |                           |           |           |            |            |      |
| Initial Position           | graceful-glitter-166      | **apricot-armadillo-167** |           |           |            | 1          |      |
| King's Gamit Accepted      | graceful-glitter-166      | **apricot-armadillo-167** |           |           |            | 1          |      |
| Evan's Gambit              | graceful-glitter-166      | apricot-armadillo-167     | 1         |           |            |            | 1    |
| King and Pawn vs King      | graceful-glitter-166      | apricot-armadillo-167     | 1         |           |            |            | 1    |
| Blackburne Shilling Gambit | graceful-glitter-166      | **apricot-armadillo-167** |           |           |            | 1          |      |
| Traxler Counterattack      | graceful-glitter-166      | **apricot-armadillo-167** |           |           |            | 1          |      |
|                            |                           |                           |           |           |            |            |      |
| Initial Position           | apricot-armadillo-167     | **graceful-glitter-166**  |           |           |            | 1          |      |
| King's Gamit Accepted      | apricot-armadillo-167     | **graceful-glitter-166**  |           |           |            | 1          |      |
| Evan's Gambit              | apricot-armadillo-167     | **graceful-glitter-166**  |           |           |            | 1          |      |
| King and Pawn vs King      | **apricot-armadillo-167** | graceful-glitter-166      |           |           | 1          |            |      |
| Blackburne Shilling Gambit | apricot-armadillo-167     | **graceful-glitter-166**  |           |           |            | 1          |      |
| Traxler Counterattack      | apricot-armadillo-167     | **graceful-glitter-166**  |           |           |            | 1          |      |
|                            |                           |                           |           |           |            |            |      |
| Initial Position           | apricot-armadillo-167     | effortless-vortex-142     |           | 1         |            |            | 1    |
| King's Gamit Accepted      | apricot-armadillo-167     | effortless-vortex-142     |           | 1         |            |            | 1    |
| Evan's Gambit              | **apricot-armadillo-167** | effortless-vortex-142     |           |           | 1          |            |      |
| King and Pawn vs King      | apricot-armadillo-167     | effortless-vortex-142     | 1         |           |            |            | 1    |
| Blackburne Shilling Gambit | **apricot-armadillo-167** | effortless-vortex-142     |           |           | 1          |            |      |
| Traxler Counterattack      | apricot-armadillo-167     | effortless-vortex-142     |           |           |            |            | 1    |
|                            |                           |                           |           |           |            |            |      |
| Initial Position           | **fresh_blaze_174**       | effortless-vortex-142     |           |           | 1          |            |      |
| King's Gamit Accepted      | **fresh_blaze_174**       | effortless-vortex-142     |           |           | 1          |            |      |
| Evan's Gambit              | fresh_blaze_174           | **effortless-vortex-142** |           |           |            | 1          |      |
| King and Pawn vs King      | fresh_blaze_174           | effortless-vortex-142     | 1         |           |            |            | 1    |
| Blackburne Shilling Gambit | fresh_blaze_174           | effortless-vortex-142     |           |           |            |            | 1    |
| Traxler Counterattack      | fresh_blaze_174           | effortless-vortex-142     |           | 1         |            |            | 1    |
|                            |                           |                           |           |           |            |            |      |
| Initial Position           | graceful-glitter-166      | **fresh_blaze_174**       |           |           |            | 1          |      |
| King's Gamit Accepted      | graceful-glitter-166      | fresh_blaze_174           | 1         |           |            |            | 1    |
| Evan's Gambit              | graceful-glitter-166      | **fresh_blaze_174**       |           |           |            | 1          |      |
| King and Pawn vs King      | graceful-glitter-166      | fresh_blaze_174           | 1         |           |            |            | 1    |
| Blackburne Shilling Gambit | graceful-glitter-166      | **fresh_blaze_174**       |           |           |            | 1          |      |
| Traxler Counterattack      | graceful-glitter-1664     | **fresh_blaze_174**       |           |           |            | 1          |      |
|                            |                           |                           |           |           |            |            |      |
| Initial Position           | fresh_blaze_174           | apricot-armadillo-167     |           | 1         |            |            | 1    |
| King's Gamit Accepted      | fresh_blaze_174           | **apricot-armadillo-167** |           |           |            | 1          |      |
| Evan's Gambit              | **fresh_blaze_174**       | apricot-armadillo-167     |           |           | 1          |            |      |
| King and Pawn vs King      | fresh_blaze_174           | apricot-armadillo-167     | 1         |           |            |            | 1    |
| Blackburne Shilling Gambit | **fresh_blaze_174**       | apricot-armadillo-167     |           |           | 1          |            |      |
| Traxler Counterattack      | fresh_blaze_174           | **apricot-armadillo-167** |           |           |            | 1          |      |
|                            |                           |                           |           |           |            |            |      |
| Initial Position           | apricot-armadillo-167     | **fresh_blaze_174**       |           |           |            | 1          |      |
| King's Gamit Accepted      | **apricot-armadillo-167** | fresh_blaze_174           |           |           | 1          |            |      |
| Evan's Gambit              | **apricot-armadillo-167** | fresh_blaze_174           |           |           | 1          |            |      |
| King and Pawn vs King      | apricot-armadillo-167     | fresh_blaze_174           | 1         |           |            |            | 1    |
| Blackburne Shilling Gambit | **apricot-armadillo-167** | fresh_blaze_174           |           |           | 1          |            |      |
| Traxler Counterattack      | **apricot-armadillo-167** | fresh_blaze_174           |           |           | 1          |            |      |

