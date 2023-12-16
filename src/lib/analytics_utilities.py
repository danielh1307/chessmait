import pandas as pd


def evaluation_to_class(CLASSES, evaluation):
    """
    This method creates class labels from regression values.

    Parameters
    ----------
    CLASSES : dict
        Dictionary with class names associated to min and max values.
    evaluation
        The regression value which is transformed to a class label.

    Returns
    -------
    The class label which matches the regression value.

    """
    for key, range_dict in CLASSES.items():
        if "min" in range_dict and "max" in range_dict:
            # if we have both min and max, our value is in between
            min_value = range_dict["min"]
            max_value = range_dict["max"]
            if min_value <= evaluation <= max_value:
                return key
        elif "min" in range_dict:
            # if we have just a min, our value must be greater (no upper-bound)
            min_value = range_dict["min"]
            if evaluation > min_value:
                return key
        elif "max" in range_dict:
            # if we have just a max, our value must be smaller (no lower-bound)
            max_value = range_dict["max"]
            if evaluation < max_value:
                return key
    raise Exception(f"No class found for {evaluation}")


def remove_mates(df: pd.DataFrame, column):
    if df[column].dtype == 'object':
        df = df[~df[column].str.startswith('#')]
        df.loc[:, column] = df[column].astype(int)
    return df


def count_pieces(fen):
    return sum(1 for char in fen.split()[0] if char.isalpha())
