import matplotlib.colors
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

df_q_table_tic_tac_toe = pd.read_csv("tic-tac-toe-statistics-q-table.csv", delimiter="\t", header=0)
df_q_net_tic_tac_toe = pd.read_csv("tic-tac-toe-statistics-q-net.csv", delimiter="\t", header=0)
df_q_net_chess = pd.read_csv("chess-statistics-q-net.csv", delimiter="\t", header=0)


def draw_dataframe(df, title, best_loss, chess):

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

    if best_loss is not None:
        if chess:
            axes2.set_ylim([-0.000001, 0.000011])
        else:
            axes2.set_ylim([-1, 11.0])

        sns.lineplot(y=best_loss, x=df['#of-trainings'], ax=axes, color=colors['black'], linestyle='dashed', lw=5, label="best-loss")
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
    plt.show()

    plt.savefig(title + '.png', bbox_inches='tight')


draw_dataframe(df_q_table_tic_tac_toe, "Q-Table-Tic-Tac-Toe", None, False)
draw_dataframe(df_q_net_tic_tac_toe, "Q-Net-Tic-Tac-Toe", df_q_net_tic_tac_toe['best-loss']*100, False)
#draw_dataframe(df_q_net_chess, "Q-Net-Chess", df_q_net_chess['best-loss']*1000000, True)
