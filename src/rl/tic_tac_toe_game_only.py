from pettingzoo.classic import tictactoe_v3
from src.rl.tic_tac_toe_utilities import *


def run(_env):

    observation, reward, termination, truncation, info = _env.last()

    cache = np.zeros((9, 9))

    rounds = 0

    indexes = [0, 0, 0]
    winner_name = ""

    while not termination and rounds < 10:

        agent = _env.agents[rounds % 2]

        mask = observation["action_mask"]
        # this is where you would insert your policy
        action = _env.action_space(agent).sample(mask)

        _env.step(action)
        observation, reward, termination, truncation, info = _env.last()

        cache_board(cache, rounds, _env.board)
        if termination:
            winner, indexes = check_winner(cache[rounds], _env.board.winning_combinations)
            if winner:
                winner_name = agent
                print(f"{COLOR_WIN}won by: {agent}{COLOR_DEFAULT}")
            else:
                winner_name = "draw"
                print("draw")

        rounds += 1

    print_cache(cache, rounds, indexes)
    return winner_name


if __name__ == "__main__":
    print("RL start ***")
    env = tictactoe_v3.raw_env()
    env.reset(seed=42)
    games = {}
    for i in range(10):
        winner = run(env)
        if winner not in games.keys():
            games[winner] = 0
        games[winner] += 1
        env.reset()
    env.close()
    print_summary(games)
    print("RL end ***")
