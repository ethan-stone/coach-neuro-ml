import logging
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np


def split_data(data, parameters):
    X = data.iloc[:, :-1]
    for column in X.columns:
        X[column] = (X[column] - X[column].min()) / (X[column].max() - X[column].min())
    y = data["Class Name"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=parameters["test_size"],
                                                        random_state=parameters["random_state"])

    return [X_train, X_test, y_train, y_test]


def train_model(X_train, y_train):

    model = Sequential([
        Dense(64, input_shape=(34,), activation='relu'),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='Adam', loss="mse")

    model.fit(X_train, y_train, epochs=50, batch_size=16)

    return model


def evaluate_model(model, X_test, y_test):

    predictions = model.predict(X_test)
    predictions = np.around(predictions)

    accuracy = accuracy_score(y_test, predictions)

    logger = logging.getLogger(__name__)
    logger.info(f"BasketballFrontElbowModel has an accuracy of {accuracy}")

    return None
