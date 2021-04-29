import logging
from typing import Dict
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from ..extras.utilities import Net, LossEarlyStopping
import torch
from torch.utils.data import TensorDataset, DataLoader
import time

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")


def to_categorical(y, num_classes=None, dtype='float32'):
    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape):
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=dtype)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical


def gather_data_generic(class_mappings: Dict[str, int]):

    poses = []

    cap = cv2.cv2.VideoCapture(0)
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            for class_name, class_val in class_mappings.items():
                print(f"Starting class {class_name} in 5 seconds")
                time.sleep(5)
                start = time.time()
                while time.time() - start < 10:
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
    X_train = torch.from_numpy(X_train.to_numpy().astype(np.float32))
    y_train = torch.from_numpy(y_train.astype(np.float32))

    X_val = torch.from_numpy(X_val.to_numpy().astype(np.float32))
    y_val = torch.from_numpy(y_val.astype(np.float32))

    train_set = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_set, batch_size=64)

    val_set = TensorDataset(X_val, y_val)
    val_loader = DataLoader(val_set, batch_size=64)

    input_dim = parameters["input_dim"]
    output_dim = parameters["output_dim"]

    model = Net(input_dim, output_dim)
    model.to(device)

    criterion = torch.nn.MSELoss(reduction='sum')
    optimizer = torch.optim.Adam(model.parameters())
    early_stopping = LossEarlyStopping(10)

    epochs = 1000

    train_losses = []
    valid_losses = []
    avg_train_losses = []
    avg_valid_losses = []

    for epoch in range(1, epochs + 1):
        print("\n==============================\n")
        print("Epoch = " + str(epoch))

        model.train()
        for i, batch in enumerate(train_loader, 0):
            inputs, labels = batch
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            y_pred = model(inputs)
            loss = criterion(y_pred, labels)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        model.eval()
        for i, batch in enumerate(val_loader):
            inputs, labels = batch
            inputs = inputs.to(device)
            labels = labels.to(device)

            y_pred = model(inputs)

            loss = criterion(y_pred, labels)
            valid_losses.append(loss.item())

        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)

        epoch_len = len(str(epochs))

        print_msg = f"[{epoch:>{epoch_len}}/{epochs:>{epoch_len}}] " \
                    f"train_loss: {train_loss:.5f} " \
                    f"valid_loss: {valid_loss:.5f}"

        print(print_msg)

        train_losses = []
        valid_losses = []

        if early_stopping.stop_early(valid_loss):
            print("Early Stopping")
            break

    print("Finished Training")
    return model


def evaluate_model_generic(model, X_test, y_test, model_name):
    X_test = torch.from_numpy(X_test.to_numpy().astype(np.float32)).to(device)
    y_test = y_test.astype(np.float32)

    model = model.to(device)

    predictions = model(X_test)
    predictions = np.around(predictions.cpu().detach().numpy())

    accuracy = accuracy_score(y_test, predictions)

    logger = logging.getLogger(__name__)
    logger.info("%s has an accuracy of %.5f" % (model_name, accuracy))
