from ....utilities import process_raw_data_generic, gather_data_generic


def gather_data():
    return gather_data_generic()


def process_raw_data(raw_data):

    # i = elbow in
    # m = elbow a little out
    # o = elbow out

    class_mappings = {
        "i": 0,
        "m": 1,
        "o": 2
    }

    return process_raw_data_generic(raw_data, class_mappings)
