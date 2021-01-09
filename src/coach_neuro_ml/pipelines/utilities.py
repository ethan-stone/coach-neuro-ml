import logging
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.regularizers import l2


def process_raw_data_generic(raw_data, class_mappings):
    new_samples = []

    for i in range(len(raw_data.index)):
        sample = raw_data.iloc[i, 0]
        xs = sample["xs"]
        ys = sample["ys"]

        new_sample = {}

        for key, value in xs.items():
            new_sample[key] = value

        for key, value in ys.items():
            new_sample["Class"] = class_mappings[value]

        new_samples.append(new_sample)

    df = pd.DataFrame(new_samples)

    return df


def split_data_generic(data, parameters):
    X = data.iloc[:, :-1]
    for column in X.columns:
        X[column] = (X[column] - X[column].min()) / (X[column].max() - X[column].min())
    y = np.array(data["Class"])
    y_one_hot = to_categorical(y)
    print(y_one_hot)

    X_train, X_test, y_train, y_test = train_test_split(X, y_one_hot, test_size=parameters["test_size"],
                                                        random_state=parameters["random_state"])

    return [X_train, X_test, y_train, y_test]


def train_model_generic(X_train, y_train):

    callback = EarlyStopping(monitor='loss', patience=3)

    model = Sequential([
        Dense(64, input_shape=(34,), activation='relu', kernel_regularizer=l2(0.001)),
        Dropout(0.2),
        Dense(16, activation='relu', kernel_regularizer=l2(0.01)),
        Dropout(0.1),
        Dense(3, activation='softmax')
    ])

    model.compile(optimizer="Adam", loss="mse")

    model.fit(X_train, y_train, epochs=100, batch_size=16, callbacks=[callback])

    return model


def evaluate_model_generic(model, X_test, y_test, model_name):

    predictions = model.predict(X_test)
    predictions = np.around(predictions)

    accuracy = accuracy_score(y_test, predictions)

    logger = logging.getLogger(__name__)
    logger.info(f"{model_name} has an accuracy of {accuracy}")

    return None
