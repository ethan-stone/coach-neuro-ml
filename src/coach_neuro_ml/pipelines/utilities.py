import logging
from typing import Dict
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.regularizers import l2

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


def gather_data_generic(class_mappings: Dict[str, int], data_gather_params: Dict[str, int]):

    poses = []

    cap = cv2.cv2.VideoCapture(0)
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            for class_name, class_val in class_mappings.items():
                print(f"Starting class {class_name} in 5 seconds")
                time.sleep(data_gather_params["sleep_time"])
                start = time.time()
                while time.time() - start < data_gather_params["record_time"]:
                    success, image = cap.read()
                    if not success:
                        print("Ignoring empty camera frame.")
                        continue

                    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
                    image.flags.writeable = False
                    results = pose.process(image)
                    image.flags.writeable = True
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                    frame_pose = {}
                    if results.pose_landmarks is not None:
                        for lm_id, lm in enumerate(results.pose_landmarks.landmark):
                            h, w, _ = image.shape
                            frame_pose[str(2 * lm_id)] = lm.x * w
                            frame_pose[str(2 * lm_id + 1)] = lm.y * h
                        frame_pose["Class"] = class_val

                        poses.append(frame_pose)

                    cv2.imshow("MediaPipe Pose", image)
                    if cv2.waitKey(5) & 0xFF == ord("q"):
                        break
            break

        cap.release()

    columns = [str(i) for i in list(range(0, 66))]
    columns.append("Class")

    return pd.DataFrame(poses, columns=columns)


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

    for i in range(len(X.index)):
        X.iloc[i, :] = X.iloc[i, :] / np.linalg.norm(X.iloc[i, :])

    y = np.array(data["Class"])
    y_one_hot = to_categorical(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y_one_hot, test_size=parameters["test_size"],
                                                        random_state=parameters["random_state"])

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=parameters["val_size"],
                                                      random_state=parameters["random_state"])

    return [X_train, X_test, X_val, y_val, y_train, y_test]


def train_model_generic(X_train, y_train, X_val, y_val, parameters):

    input_dim = parameters["input_dim"]
    output_dim = parameters["output_dim"]

    early_stopping = EarlyStopping(monitor='loss', patience=5)

    model = Sequential([
        Input(input_dim),
        Dense(64, activation="relu", kernel_regularizer=l2(0.01)),
        Dropout(0.1),
        Dense(128, activation="relu"),
        Dense(64, activation="relu"),
        Dense(output_dim, activation="softmax")
    ])

    model.compile(optimizer="Adam", loss="mse")

    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=1000, batch_size=16, callbacks=[early_stopping])

    return model


def evaluate_model_generic(model, X_test, y_test, model_name):
    predictions = model.predict(X_test)
    predictions = np.around(predictions)

    accuracy = accuracy_score(y_test, predictions)

    logger = logging.getLogger(__name__)
    logger.info("%s has an accuracy of %.5f" % (model_name, accuracy))
