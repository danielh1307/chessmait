from lib import utilities


def test_is_checkmate():
    assert utilities.is_checkmate("2rqkbnr/p2np1pp/1pp1bp2/3p4/3P1P2/NPP4N/P2QP1PP/1RB1KB1R b Kk - 7 8") == False
    assert utilities.is_checkmate("3rkbnr/2nqpQ1R/p1P3P1/1p1p1b2/2P5/NP6/P3P1P1/1RB1KBN1 b k - 2 19") == True

# fastest stalemate known in chess
# https://www.chess.com/forum/view/game-showcase/fastest-stalemate-known-in-chess
def test_is_stalemate():
    assert utilities.is_stalemate("5bnr/4p1pq/4Qpkr/7p/7P/4P3/PPPP1PP1/RNB1KBNR b KQ - 2 10")
