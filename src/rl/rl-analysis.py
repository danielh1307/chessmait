import matplotlib.colors
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def draw_dataframe(name, title, best_loss_scaler, chess):

    df = pd.read_csv(name + ".csv", delimiter="\t", header=0)
    df.drop(df.tail(1).index, inplace=True)  # drop last row
    df['#of-trainings'] = df['#of-trainings'].astype(np.float64)
    df['p1-training-won-scaled'] = df['p1-training-won'] / df['#of-trainings'] * 1000
    df['p2-training-won-scaled'] = df['p2-training-won'] / df['#of-trainings'] * 1000
    df['draw-training-scaled'] = df['draw-training'] / df['#of-trainings'] * 1000

    fig, axes = plt.subplots(1, 1, figsize=(20, 15))

    player_1 = "white" if chess else "player 1"
    player_2 = "black" if chess else "player 2"

    colors = matplotlib.colors.CSS4_COLORS
    sns.lineplot(data=df, y='p1-test-won', x='#of-trainings', ax=axes, color=colors['mediumblue'], lw=4, label=player_1 + " (test)")
    sns.lineplot(data=df, y='p1-training-won-scaled', x='#of-trainings', ax=axes, color=colors['mediumblue'], lw=2, label=player_1 + " (training scaled)")
    sns.lineplot(data=df, y='p2-test-won', x='#of-trainings', ax=axes, color=colors['forestgreen'], lw=4, label=player_2 + " (test)")
    sns.lineplot(data=df, y='p2-training-won-scaled', x='#of-trainings', ax=axes, color=colors['forestgreen'], lw=2, label=player_2 + " (training scaled)")
    sns.lineplot(data=df, y='draw-test', x='#of-trainings', ax=axes, color=colors['orangered'], lw=4, label="draw (test)")
    sns.lineplot(data=df, y='draw-training-scaled', x='#of-trainings', ax=axes, color=colors['orangered'], lw=2, label="draw (training scaled)")

    axes2 = axes.twinx()

    if best_loss_scaler is not None:
        if chess:
            axes2.set_ylim([-0.0002, 0.0022])
        else:
            axes2.set_ylim([-1, 11])

        val = df['best-loss'].iloc[-1]
        plt.text(100000, val, "{:0.5f}".format(val), fontsize=14)

        sns.lineplot(y=df['best-loss']*best_loss_scaler, x=df['#of-trainings'], ax=axes, color=colors['black'], linestyle='dashed', lw=5, label="best-loss")
        axes2.tick_params(axis='y', labelcolor=colors['black'])
        axes2.set_ylabel("best-loss", fontsize=20, color=colors['black'])
    else:
        axes2.set_ylim([-0.1, 1.1])
        axes2.set_ylabel("percent", fontsize=20, color=colors['black'])

    axes.set_ylim([-100, 1100])
    axes.set_ylabel("games won", fontsize=20)
    axes.set(xticks=np.arange(0.0, 110000.0, 10000.0), yticks=np.arange(0, 1100, 100))
    axes.grid()
    axes.set_xlabel("number of games", fontsize=20)
    plt.suptitle(title, fontsize=30, fontweight='bold')
    plt.title("Comparison between " + player_1 + ", " + player_2 + " and draws", fontsize=20)
    axes.legend(borderpad=1, labelspacing=1, prop={'size': 12}, loc='center right')

    plt.tight_layout()
    #plt.show()

    plt.savefig(title + '.png', bbox_inches='tight')


def draw_dataset(name, axes, color):

    df = pd.read_csv("chess-statistics-q-net-" + name + ".csv", delimiter="\t", header=0)
    df.drop(df.tail(1).index, inplace=True)  # drop last row
    df['#of-trainings'] = df['#of-trainings'].astype(np.float64)
    sns.lineplot(data=df, y='best-loss', x='#of-trainings', ax=axes, color=color, lw=3, label=name)


def draw_optimizer_loss_function_statistics():

    fig, axes = plt.subplots(1, 1, figsize=(20, 20))

    colors = matplotlib.colors.CSS4_COLORS
    draw_dataset("ada-huber", axes, colors['mediumblue'])
    draw_dataset("ada-L1", axes, colors['cornflowerblue'])
    draw_dataset("ada-mse", axes, colors['lightsteelblue'])
    draw_dataset("adam-huber", axes, colors['crimson'])
    draw_dataset("adam-L1", axes, colors['hotpink'])
    draw_dataset("adam-mse", axes, colors['pink'])
    draw_dataset("adamw-huber", axes, colors['green'])
    draw_dataset("adamw-L1", axes, colors['limegreen'])
    draw_dataset("adamw-mse", axes, colors['aquamarine'])
    draw_dataset("sgd-huber", axes, colors['darkorange'])
    draw_dataset("sgd-L1", axes, colors['bisque'])
    draw_dataset("sgd-mse", axes, colors['gold'])

    axes.set(xticks=np.arange(100.0, 1100.0, 100.0), yticks=np.arange(0, 0.015, 0.001))
    axes.grid()
    axes.legend(borderpad=1, labelspacing=1, prop={'size': 12}, loc='upper right')

    plt.suptitle("Losses", fontsize=30, fontweight='bold')
    plt.title("Comparison of losses for different optimizer and loss-functions", fontsize=20)

    plt.tight_layout()
    #plt.show()

    plt.savefig('optimizer-vs-loss-function.png', bbox_inches='tight')


draw_dataframe("tic-tac-toe-statistics-q-table", "Q-Table-Tic-Tac-Toe", None, False)
draw_dataframe("tic-tac-toe-statistics-q-net", "Q-Net-Tic-Tac-Toe", 100, False)
draw_dataframe("chess-statistics-q-net-3-layer", "Q-Net-Chess", 500000, True)
draw_optimizer_loss_function_statistics()
