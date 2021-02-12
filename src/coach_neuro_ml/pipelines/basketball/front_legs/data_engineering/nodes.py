from ......coach_neuro_ml.utilities import process_raw_data_generic


def process_raw_data(raw_data):

    # w = wide stance
    # g = good stance
    # n = narrow stance

    class_mappings = {
        "g": 0,
        "w": 1,
        "n": 2
    }

    return process_raw_data_generic(raw_data, class_mappings)