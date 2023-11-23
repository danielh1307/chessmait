import pandas as pd

from src.lib.analytics_utilities import remove_mates


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
    assert "Fen1" in cleaned_df["FEN"].tolist() # should still be present
    assert cleaned_df["Evaluation"].dtype == int  # "Evaluation" column type should be int
