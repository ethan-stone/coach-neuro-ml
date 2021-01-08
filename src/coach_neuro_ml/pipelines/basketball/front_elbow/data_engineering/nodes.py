import pandas as pd


def process_raw_data(raw_data):
    class_mappings = {
        "i": 1,
        "o": 0
    }

    new_samples = []

    for i in range(len(raw_data.index)):
        sample = raw_data.iloc[i, 0]
        xs = sample["xs"]
        ys = sample["ys"]

        new_sample = {}

        for key, value in xs.items():
            new_sample[key] = value

        for key, value in ys.items():
            new_sample["Class Name"] = class_mappings[value]

        new_samples.append(new_sample)

    df = pd.DataFrame(new_samples)

    return df
