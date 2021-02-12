import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")


class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.h1 = torch.nn.Linear(34, 64)
        self.h2 = torch.nn.Linear(64, 16)
        self.o = torch.nn.Linear(16, 3)

        self.dropout = torch.nn.Dropout(0.5)

    def forward(self, x):
        x = F.relu(self.h1(x))
        x = self.dropout(x)
        x = F.relu(self.h2(x))
        x = F.softmax(self.o(x), dim=1)
        return x


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

    return [X_train, X_test, y_train, y_test]


def train_model_generic(X_train, y_train):
    X_train = torch.from_numpy(X_train.to_numpy().astype(np.float32))
    y_train = torch.from_numpy(y_train.astype(np.float32))

    dataset = TensorDataset(X_train, y_train)

    data_loader = DataLoader(dataset, batch_size=16)

    model = Net()
    model.to(device)

    criterion = torch.nn.MSELoss(reduction='sum')
    optimizer = torch.optim.Adam(model.parameters())

    epochs = 100

    for epoch in range(epochs):
        print("\n==============================\n")
        print("Epoch = " + str(epoch))
        running_loss = 0.0
        for i, batch in enumerate(data_loader, 0):
            inputs, labels = batch
            inputs = inputs.to(device)
            labels = labels.to(device)

            y_pred = model(inputs)
            loss = criterion(y_pred, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if i % 100 == 99:
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
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
    logger.info(f"{model_name} has an accuracy of {accuracy}")

    return None
