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

Therefore, the matches start at different positions:

| Position                   | FEN                                                                                                                                                                                    | Evaluation |
|----------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------|
| Initial Position           | [rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1](https://lichess.org/analysis)                                                                                               | ~+0.2      |
| King's Gambit Accepted     | [rnbqkbnr/pppp1ppp/8/8/4Pp2/8/PPPP2PP/RNBQKBNR w KQkq - 0 3](https://lichess.org/analysis/standard/rnbqkbnr/pppp1ppp/8/8/4Pp2/8/PPPP2PP/RNBQKBNR_w_KQkq_-_0_3)                         | ~-0.4      |
| Evan's Gambit              | [r1bqk1nr/pppp1ppp/2n5/2b1p3/1PB1P3/5N2/P1PP1PPP/RNBQK2R b KQkq b3 0 4](https://lichess.org/analysis/standard/r1bqk1nr/pppp1ppp/2n5/2b1p3/1PB1P3/5N2/P1PP1PPP/RNBQK2R_b_KQkq_b3_0_4)   | ~-0.3      |
| King and Pawn vs King      | [3k4/8/3K4/3P4/8/8/8/8 w - - 0 1](https://lichess.org/analysis/fromPosition/3k4/8/3K4/3P4/8/8/8/8_w_-_-_0_1)                                                                           | #11        |
| Blackburne Shilling Gambit | [r1bqkbnr/pppp1ppp/8/4p3/2BnP3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4](https://lichess.org/analysis/fromPosition/r1bqkbnr/pppp1ppp/8/4p3/2BnP3/5N2/PPPP1PPP/RNBQK2R_w_KQkq_-_4_4)           | ~+1.6      |
| Traxler Counterattack      | [r1bqk2r/pppp1ppp/2n2n2/2b1p1N1/2B1P3/8/PPPP1PPP/RNBQK2R w KQkq - 6 5](https://lichess.org/analysis/fromPosition/r1bqk2r/pppp1ppp/2n2n2/2b1p1N1/2B1P3/8/PPPP1PPP/RNBQK2R_w_KQkq_-_6_5) | ~+1.4      |

## Results

If the result is a draw but one color reached an evaluation of +10 (white) or -10 (black), which is a clear win, it is
marked in the table.

| Position                   | White                     | Black                     | White +10 | Black +10 | White wins | Black Wins | Draw |
|----------------------------|---------------------------|---------------------------|-----------|-----------|------------|------------|------|
| Initial Position           | **graceful-glitter-166**  | smart-valley-6            |           |           | 1          |            |      |
| King's Gambit Accepted     | **graceful-glitter-166**  | smart-valley-6            |           |           | 1          |            |      |
| Evan's Gambit              | **graceful-glitter-166**  | smart-valley-6            |           |           | 1          |            |      |
| King and Pawn vs King      | graceful-glitter-166      | smart-valley-6            | 1         |           |            |            | 1    |
| Blackburne Shilling Gambit | **graceful-glitter-166**  | smart-valley-6            |           |           | 1          |            |      |
| Traxler Counterattack      | **graceful-glitter-166**  | smart-valley-6            |           |           | 1          |            |      |
|                            |                           |                           |           |           |            |            |      |
| Position                   | White                     | Black                     | White +10 | Black +10 | White wins | Black Wins | Draw |
| Initial Position           | graceful-glitter-166      | **apricot-armadillo-167** |           |           |            | 1          |      |
| King's Gambit Accepted     | graceful-glitter-166      | **apricot-armadillo-167** |           |           |            | 1          |      |
| Evan's Gambit              | graceful-glitter-166      | apricot-armadillo-167     | 1         |           |            |            | 1    |
| King and Pawn vs King      | graceful-glitter-166      | apricot-armadillo-167     | 1         |           |            |            | 1    |
| Blackburne Shilling Gambit | graceful-glitter-166      | **apricot-armadillo-167** |           |           |            | 1          |      |
| Traxler Counterattack      | graceful-glitter-166      | **apricot-armadillo-167** |           |           |            | 1          |      |
|                            |                           |                           |           |           |            |            |      |
| Initial Position           | apricot-armadillo-167     | **graceful-glitter-166**  |           |           |            | 1          |      |
| King's Gambit Accepted     | apricot-armadillo-167     | **graceful-glitter-166**  |           |           |            | 1          |      |
| Evan's Gambit              | apricot-armadillo-167     | **graceful-glitter-166**  |           |           |            | 1          |      |
| King and Pawn vs King      | **apricot-armadillo-167** | graceful-glitter-166      |           |           | 1          |            |      |
| Blackburne Shilling Gambit | apricot-armadillo-167     | **graceful-glitter-166**  |           |           |            | 1          |      |
| Traxler Counterattack      | apricot-armadillo-167     | **graceful-glitter-166**  |           |           |            | 1          |      |
|                            |                           |                           |           |           |            |            |      |
| Position                   | White                     | Black                     | White +10 | Black +10 | White wins | Black Wins | Draw |
| Initial Position           | apricot-armadillo-167     | effortless-vortex-142     |           | 1         |            |            | 1    |
| King's Gambit Accepted     | apricot-armadillo-167     | effortless-vortex-142     |           | 1         |            |            | 1    |
| Evan's Gambit              | **apricot-armadillo-167** | effortless-vortex-142     |           |           | 1          |            |      |
| King and Pawn vs King      | apricot-armadillo-167     | effortless-vortex-142     | 1         |           |            |            | 1    |
| Blackburne Shilling Gambit | **apricot-armadillo-167** | effortless-vortex-142     |           |           | 1          |            |      |
| Traxler Counterattack      | apricot-armadillo-167     | effortless-vortex-142     |           |           |            |            | 1    |
|                            |                           |                           |           |           |            |            |      |
| Initial Position           | **effortless-vortex-142** | apricot-armadillo-167     |           |           | 1          |            |      |
| King's Gambit Accepted     | effortless-vortex-142     | **apricot-armadillo-167** |           |           |            | 1          |      |
| Evan's Gambit              | effortless-vortex-142     | **apricot-armadillo-167** |           |           |            | 1          |      |
| King and Pawn vs King      | effortless-vortex-142     | apricot-armadillo-167     | 1         |           |            |            | 1    |
| Blackburne Shilling Gambit | effortless-vortex-142     | apricot-armadillo-167     | 1         |           |            |            | 1    |
| Traxler Counterattack      | effortless-vortex-142     | **apricot-armadillo-167** |           |           |            | 1          |      |
|                            |                           |                           |           |           |            |            |      |
| Position                   | White                     | Black                     | White +10 | Black +10 | White wins | Black Wins | Draw |
| Initial Position           | **fresh_blaze_174**       | effortless-vortex-142     |           |           | 1          |            |      |
| King's Gambit Accepted     | **fresh_blaze_174**       | effortless-vortex-142     |           |           | 1          |            |      |
| Evan's Gambit              | fresh_blaze_174           | **effortless-vortex-142** |           |           |            | 1          |      |
| King and Pawn vs King      | fresh_blaze_174           | effortless-vortex-142     | 1         |           |            |            | 1    |
| Blackburne Shilling Gambit | fresh_blaze_174           | effortless-vortex-142     |           |           |            |            | 1    |
| Traxler Counterattack      | fresh_blaze_174           | effortless-vortex-142     |           | 1         |            |            | 1    |
|                            |                           |                           |           |           |            |            |      |
| Initial Position           | **effortless-vortex-142** | fresh_blaze_174           |           |           | 1          |            |      |
| King's Gambit Accepted     | effortless-vortex-142     | **fresh_blaze_174**       |           |           |            | 1          |      |
| Evan's Gambit              | effortless-vortex-142     | **fresh_blaze_174**       |           |           |            | 1          |      |
| King and Pawn vs King      | effortless-vortex-142     | fresh_blaze_174           | 1         |           |            |            | 1    |
| Blackburne Shilling Gambit | **effortless-vortex-142** | fresh_blaze_174           |           |           | 1          |            |      |
| Traxler Counterattack      | effortless-vortex-142     | **fresh_blaze_174**       |           |           |            | 1          |      |
|                            |                           |                           |           |           |            |            |      |
| Position                   | White                     | Black                     | White +10 | Black +10 | White wins | Black Wins | Draw |
| Initial Position           | graceful-glitter-166      | **fresh_blaze_174**       |           |           |            | 1          |      |
| King's Gambit Accepted     | graceful-glitter-166      | fresh_blaze_174           | 1         |           |            |            | 1    |
| Evan's Gambit              | graceful-glitter-166      | **fresh_blaze_174**       |           |           |            | 1          |      |
| King and Pawn vs King      | graceful-glitter-166      | fresh_blaze_174           | 1         |           |            |            | 1    |
| Blackburne Shilling Gambit | graceful-glitter-166      | **fresh_blaze_174**       |           |           |            | 1          |      |
| Traxler Counterattack      | graceful-glitter-166      | **fresh_blaze_174**       |           |           |            | 1          |      |
|                            |                           |                           |           |           |            |            |      |
| Initial Position           | **fresh_blaze_174**       | graceful-glitter-166      |           |           | 1          |            |      |
| King's Gambit Accepted     | fresh_blaze_174           | **graceful-glitter-166**  |           |           |            | 1          |      |
| Evan's Gambit              | fresh_blaze_174           | graceful-glitter-166      |           |           |            |            | 1    |
| King and Pawn vs King      | fresh_blaze_174           | graceful-glitter-166      | 1         |           |            |            | 1    |
| Blackburne Shilling Gambit | fresh_blaze_174           | **graceful-glitter-166**  |           |           |            | 1          |      |
| Traxler Counterattack      | **fresh_blaze_174**       | graceful-glitter-166      |           |           | 1          |            |      |
|                            |                           |                           |           |           |            |            |      |
| Position                   | White                     | Black                     | White +10 | Black +10 | White wins | Black Wins | Draw |
| Initial Position           | fresh_blaze_174           | apricot-armadillo-167     |           | 1         |            |            | 1    |
| King's Gambit Accepted     | fresh_blaze_174           | **apricot-armadillo-167** |           |           |            | 1          |      |
| Evan's Gambit              | **fresh_blaze_174**       | apricot-armadillo-167     |           |           | 1          |            |      |
| King and Pawn vs King      | fresh_blaze_174           | apricot-armadillo-167     | 1         |           |            |            | 1    |
| Blackburne Shilling Gambit | **fresh_blaze_174**       | apricot-armadillo-167     |           |           | 1          |            |      |
| Traxler Counterattack      | fresh_blaze_174           | **apricot-armadillo-167** |           |           |            | 1          |      |
|                            |                           |                           |           |           |            |            |      |
| Initial Position           | apricot-armadillo-167     | **fresh_blaze_174**       |           |           |            | 1          |      |
| King's Gambit Accepted     | **apricot-armadillo-167** | fresh_blaze_174           |           |           | 1          |            |      |
| Evan's Gambit              | **apricot-armadillo-167** | fresh_blaze_174           |           |           | 1          |            |      |
| King and Pawn vs King      | apricot-armadillo-167     | fresh_blaze_174           | 1         |           |            |            | 1    |
| Blackburne Shilling Gambit | **apricot-armadillo-167** | fresh_blaze_174           |           |           | 1          |            |      |
| Traxler Counterattack      | **apricot-armadillo-167** | fresh_blaze_174           |           |           | 1          |            |      |
|                            |                           |                           |           |           |            |            |      |
| Position                   | White                     | Black                     | White +10 | Black +10 | White wins | Black Wins | Draw |
| Initial Position           | **effortless-vortex-142** | graceful-glitter-166      |           |           | 1          |            |      |
| King's Gambit Accepted     | effortless-vortex-142     | **graceful-glitter-166**  |           |           |            | 1          |      |
| Evan's Gambit              | effortless-vortex-142     | graceful-glitter-166      | 1         |           |            |            | 1    |
| King and Pawn vs King      | effortless-vortex-142     | graceful-glitter-166      | 1         |           |            |            | 1    |
| Blackburne Shilling Gambit | effortless-vortex-142     | graceful-glitter-166      |           |           |            |            | 1    |
| Traxler Counterattack      | effortless-vortex-142     | **graceful-glitter-166**  |           |           |            | 1          |      |
|                            |                           |                           |           |           |            |            |      |
| Initial Position           | **graceful-glitter-166**  | effortless-vortex-142     |           |           | 1          |            |      |
| King's Gambit Accepted     | graceful-glitter-166      | **effortless-vortex-142** |           |           |            | 1          |      |
| Evan's Gambit              | **graceful-glitter-166**  | effortless-vortex-142     |           |           | 1          |            |      |
| King and Pawn vs King      | graceful-glitter-166      | effortless-vortex-142     | 1         |           |            |            | 1    |
| Blackburne Shilling Gambit | **graceful-glitter-166**  | effortless-vortex-142     |           |           | 1          |            |      |
| Traxler Counterattack      | graceful-glitter-166      | **effortless-vortex-142** |           |           |            | 1          |      |

| x                     | effortless-vortex-142 | graceful-glitter-166 | apricot-armadillo-167 | fresh_blaze_174 | Ergebnis |
|-----------------------|-----------------------|----------------------|-----------------------|-----------------|----------|
| effortless-vortex-142 | x                     | 5:7                  | 4:8                   | 5:7             | 0        |
| graceful-glitter-166  | 7:5                   | x                    | 6:6                   | 4:8             | +1.5     |
| apricot-armadillo-167 | 8:4                   | 6:6                  | x                     | 7.5:4.5         | +2.5     |
| fresh_blaze_174       | 7:5                   | 8:4                  | 4.5:7.5               | x               | +2       |

