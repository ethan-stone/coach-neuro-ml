import pandas as pd


def process_raw_elbow_data(elbow_raw: pd.DataFrame) -> pd.DataFrame:
    class_mapping = {
        "i": 1,
        "o": 0
    }

    new_samples = []

    for i in range(len(elbow_raw.index)):
        sample = elbow_raw.iloc[i, 0]
        xs = sample["xs"]
        ys = sample["ys"]

        new_sample = {}

        for key, value in xs.items():
            new_sample[key] = value

        for key, value in ys.items():
            new_sample["Class Name"] = class_mapping[value]

        new_samples.append(new_sample)

    df = pd.DataFrame(new_samples)

    return df
