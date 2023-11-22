def evaluation_to_class(CLASSES, evaluation):
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
