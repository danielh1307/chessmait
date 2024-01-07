# Results

When analyzing a chess model's performance, various factors are considered. These include traditional machine learning
metrics such as training and validation loss.

## Loss Function

| Model                                                                       | Training Data          | Number of epochs | Neural Network | Loss function | Training loss | Validation loss | Comment                  |
|-----------------------------------------------------------------------------|------------------------|------------------|----------------|---------------|---------------|-----------------|--------------------------|
| [smart-valley-6](https://wandb.ai/chessmait/chessmait/runs/nsr3fgu3)        | 1.6 million positions  | 15               | MLP            | MSE           | 0.63          | 0.22            |                          |
| [effortless-vortex-142](https://wandb.ai/chessmait/chessmait/runs/ncf2q0rz) | 5.2 million positions  | 30               | CNN            | Huber Loss    | 51.9          | 95.6            |                          |
| [graceful-glitter-166](https://wandb.ai/chessmait/chessmait/runs/jle1wzp7)  | 15.8 million positions | 15               | MLP            | Huber Loss    | 124.8         | 127.0           |                          |
| [apricot-armadillo-167](https://wandb.ai/chessmait/chessmait/runs/z5ras9pj) | 17.3 million positions | 15               | MLP            | Huber Loss    | 163.0         | 162.3           | Contained mate positions |
| [fresh-blaze-174](https://wandb.ai/chessmait/chessmait/runs/m70b0o0m)       | 7.7 million positions  | 50               | CNN            | Huber Loss    | 50.6          | 122.2           | Contained mate positions |

In addition to Mean Squared Error and Huber Loss, we experimented with a custom loss function. However, this did not
yield better results than Pytorch's built-in losses
(see [chessmait5-mse-adam](https://wandb.ai/chessmait/chessmait/runs/7wkgtgig)
vs. [chessmait5-customloss-adam](https://wandb.ai/chessmait/chessmait/runs/l3ddtxyp)).
Consequently, we reverted to using Huber Loss, a type of MAE (mean absolute error), for its clarity and
interpretability. For instance, a MAE of 124 indicates the model's prediction is off by approximately 1.24 (slightly
more than a pawn).

Our Convolutional Neural Network exhibited signs of overfitting, as evidenced by the significant disparity between
training and validation losses. Although the Multilayer Perceptron (MLP) displayed higher loss values, their proximity
suggests a more reliable performance. In scenarios involving mate positions, the loss is higher (as expected, since the
evaluation range (2000 to -2000) is broader compared to non-mate positions (1000 to -1000)).

## Kaufman Test

In addition to standard performance metrics, we evaluated how effectively our models play chess. Prior to testing them
against other engines or each other, we conducted the [Kaufman test](https://www.chessprogramming.org/Kaufman_Test),
which involves 25 challenging positions where the model predicts the next best move.

The results are shown in the links below, indicating the ranking of the model's suggested move by Stockfish among all
possible moves. A lower number signifies a better move by the model, with the ideal being 1.

[Results smart-valley-6](../../documentation/kaufman-results/smart-valley-6.txt)  
[Results effortless-vortex-142](../../documentation/kaufman-results/effortless-vortex-142.txt)  
[Results graceful-glitter-166](../../documentation/kaufman-results/graceful-glitter-166.txt)  
[Results apricot-armadillo-167](../../documentation/kaufman-results/apricot-armadillo-167.txt)  
[Results fresh-blaze-174](../../documentation/kaufman-results/fresh-blaze-174.txt)

All in all, the following results were achieved:  

| Model                  | Number of times best move was found |
|------------------------|-------------------------------------|
| smart-valley-6         | 0                                   |
| effortless-vortext-142 | 2                                   |
| graceful-glitter-166   | 4                                   |
| apricot-armadillo-167  | 7                                   |
| fresh-blaze-174        | 7                                   |

## Games vs. Computer Engine

Below are some sample matches of our models vs different engines.

| White                         | Black                       | Result  | Link                                          | Model Accurary | Model centipawn loss |
|-------------------------------|-----------------------------|---------|-----------------------------------------------|----------------|----------------------|
| graceful-glitter-166          | Lichess Level 3             | 1/2-1/2 | https://lichess.org/aCEivRm7                  | 77%            | 85                   |
| apricot-armadillo-167         | **Lichess Level 3**         | 0-1     | https://lichess.org/lgM48gJu                  | 60%            | 70                   |
| **Lichess Level 3**           | apricot-armadillo-167       | 1-0     | https://lichess.org/NTLBsvC9                  | 59%            | 120                  |
| **Lichess Level 3**           | graceful-glitter-166        | 1-0     | https://lichess.org/D6orh1tF                  | 56%            | 144                  |
| **apricot-armadillo-167**     | Karim-Bot (850) (chess.com) | 1-0     | https://www.chess.com/game/computer/104024995 | 77%            | n/a                  |
| **graceful-glitter-166**      | Lichess Level 3             | 1-0     | https://lichess.org/NuG12cYA                  | 96%            | 19                   |
| **apricot-armadillo-167**     | Lichess Level 3             | 1-0     | https://lichess.org/1j6YIfuF                  | 98%            | 10                   |
| smart-valley-6                | **Lichess Level 3**         | 0-1     | https://lichess.org/o0MDSHHy                  | 44%            | 234                  |
| smart-valley-6                | **Lichess Level 3**         | 0-1     | https://lichess.org/KDp1UW47                  | 74%            | 99                   |
| Matteo-Bot (1400) (chess.com) | apricot-armadillo-167       | 1/2-1/2 | https://www.chess.com/game/computer/104190707 | 74%            | n/a                  |

The game at https://lichess.org/NuG12cYA, won by the model, is a typical example. After 19 moves, the model had a mate
in 3 but failed to execute it. This is common for graceful-glitter-166, as all moves are capped at +1000 or -1000 to
eliminate outliers, excluding mate positions. Consequently, the model makes arbitrary moves within the +1000 range,
hindering its progress.