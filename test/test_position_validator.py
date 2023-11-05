from src.position_validator import get_valid_positions

def test_valid_positions():
    legal_moves = get_valid_positions("2rqkbnr/p2np1pp/1pp1bp2/3p4/3P1P2/NPP5/P2QP1PP/1RB1KBNR w Kk - 6 8")
    assert 24 == len(legal_moves)
    assert legal_moves.index("2rqkbnr/p2np1pp/1pp1bp2/3p4/3P1P2/NPP1P3/P2Q2PP/1RB1KBNR b Kk - 0 8") >= 0
    assert legal_moves.index("2rqkbnr/p2np1pp/1pp1bp2/3p4/3PPP2/NPP5/P2Q2PP/1RB1KBNR b Kk - 0 8") >= 0
    assert legal_moves.index("2rqkbnr/p2np1pp/1pp1bp2/3p1P2/3P4/NPP5/P2QP1PP/1RB1KBNR b Kk - 0 8") >= 0
    assert legal_moves.index("2rqkbnr/p2np1pp/1pp1bp2/3p4/3P1P2/NPP3P1/P2QP2P/1RB1KBNR b Kk - 0 8") >= 0
    assert legal_moves.index("2rqkbnr/p2np1pp/1pp1bp2/3p4/3P1PP1/NPP5/P2QP2P/1RB1KBNR b Kk - 0 8") >= 0
    assert legal_moves.index("2rqkbnr/p2np1pp/1pp1bp2/3p4/3P1P2/NPP4P/P2QP1P1/1RB1KBNR b Kk - 0 8") >= 0
    assert legal_moves.index("2rqkbnr/p2np1pp/1pp1bp2/3p4/3P1P1P/NPP5/P2QP1P1/1RB1KBNR b Kk - 0 8") >= 0
    assert legal_moves.index("2rqkbnr/p2np1pp/1pp1bp2/3p4/2PP1P2/NP6/P2QP1PP/1RB1KBNR b Kk - 0 8") >= 0
    assert legal_moves.index("2rqkbnr/p2np1pp/1pp1bp2/3p4/1P1P1P2/N1P5/P2QP1PP/1RB1KBNR b Kk - 0 8") >= 0
    assert legal_moves.index("2rqkbnr/p2np1pp/1pp1bp2/1N1p4/3P1P2/1PP5/P2QP1PP/1RB1KBNR b Kk - 7 8") >= 0
    assert legal_moves.index("2rqkbnr/p2np1pp/1pp1bp2/3p4/2NP1P2/1PP5/P2QP1PP/1RB1KBNR b Kk - 7 8") >= 0
    assert legal_moves.index("2rqkbnr/p2np1pp/1pp1bp2/3p4/3P1P2/1PP5/P1NQP1PP/1RB1KBNR b Kk - 7 8") >= 0
    assert legal_moves.index("2rqkbnr/p2np1pp/1pp1bp2/3p4/3P1P2/NPPQ4/P3P1PP/1RB1KBNR b Kk - 7 8") >= 0
    assert legal_moves.index("2rqkbnr/p2np1pp/1pp1bp2/3p4/3P1P2/NPP1Q3/P3P1PP/1RB1KBNR b Kk - 7 8") >= 0
    assert legal_moves.index("2rqkbnr/p2np1pp/1pp1bp2/3p4/3P1P2/NPP5/P1Q1P1PP/1RB1KBNR b Kk - 7 8") >= 0
    assert legal_moves.index("2rqkbnr/p2np1pp/1pp1bp2/3p4/3P1P2/NPP5/PQ2P1PP/1RB1KBNR b Kk - 7 8") >= 0
    assert legal_moves.index("2rqkbnr/p2np1pp/1pp1bp2/3p4/3P1P2/NPP5/P3P1PP/1RBQKBNR b Kk - 7 8") >= 0
    assert legal_moves.index("2rqkbnr/p2np1pp/1pp1bp2/3p4/3P1P2/NPP5/PB1QP1PP/1R2KBNR b Kk - 7 8") >= 0
    assert legal_moves.index("2rqkbnr/p2np1pp/1pp1bp2/3p4/3P1P2/NPP5/PR1QP1PP/2B1KBNR b Kk - 7 8") >= 0
    assert legal_moves.index("2rqkbnr/p2np1pp/1pp1bp2/3p4/3P1P2/NPP5/P2QP1PP/R1B1KBNR b Kk - 7 8") >= 0
    assert legal_moves.index("2rqkbnr/p2np1pp/1pp1bp2/3p4/3P1P2/NPP5/P2QP1PP/1RBK1BNR b k - 7 8") >= 0
    assert legal_moves.index("2rqkbnr/p2np1pp/1pp1bp2/3p4/3P1P2/NPP5/P2QPKPP/1RB2BNR b k - 7 8") >= 0
    assert legal_moves.index("2rqkbnr/p2np1pp/1pp1bp2/3p4/3P1P2/NPP2N2/P2QP1PP/1RB1KB1R b Kk - 7 8") >= 0
    assert legal_moves.index("2rqkbnr/p2np1pp/1pp1bp2/3p4/3P1P2/NPP4N/P2QP1PP/1RB1KB1R b Kk - 7 8") >= 0

    legal_moves = get_valid_positions("xxx")
    assert 0 == len(legal_moves)
