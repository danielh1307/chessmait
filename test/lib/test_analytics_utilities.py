import pandas as pd

from src.lib.analytics_utilities import remove_mates, evaluation_to_class


def test_remove_mates():
    # arrange
    data = {
        "FEN": ["Fen1", "Fen2", "Fen3", "Fen4"],
        "Evaluation": ["1", "2", "3", "#4"],
    }
    df = pd.DataFrame(data)

    # act
    cleaned_df = remove_mates(df, "Evaluation")

    # assert
    assert len(cleaned_df) == 3  # Expected length after removal
    assert "Fen4" not in cleaned_df["FEN"].tolist()  # "#" row should be removed
    assert "Fen1" in cleaned_df["FEN"].tolist()  # should still be present
    assert cleaned_df["Evaluation"].dtype == int  # "Evaluation" column type should be int


def test_evaluation_to_classes():
    # arrange
    CLASSES = {
        ">4": {
            "min": 400
        },
        "4>p>2": {
            "min": 200,
            "max": 400
        },
        "2>p>1": {
            "min": 100,
            "max": 200
        },
        "1>p>.5": {
            "min": 50,
            "max": 100
        },
        ".5>p>0": {
            "min": 0,
            "max": 50
        },
        "0>p>-0.5": {
            "min": -50,
            "max": 0
        },
        "-0.5>p>-1": {
            "min": -100,
            "max": -50
        },
        "-1>p>-2": {
            "min": -200,
            "max": -100
        },
        "-2>p>-4": {
            "min": -400,
            "max": -200
        },
        "<-4": {
            "max": -400,
        }
    }

    # act and assert
    assert "<-4" == evaluation_to_class(CLASSES, -401)
    assert "-2>p>-4" == evaluation_to_class(CLASSES, -400)
    assert "-0.5>p>-1" == evaluation_to_class(CLASSES, -100)
    assert "4>p>2" == evaluation_to_class(CLASSES, 400)
    assert ">4" == evaluation_to_class(CLASSES, 401)
    # TODO: this only works because this is the first entry with 0
    # it would be better not to include both max and min but only one of them
    assert ".5>p>0" == evaluation_to_class(CLASSES, 0)
    assert "1>p>.5" == evaluation_to_class(CLASSES, 75)