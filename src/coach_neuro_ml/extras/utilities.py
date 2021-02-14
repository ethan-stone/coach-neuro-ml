import torch
import torch.nn.functional as F

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


class EarlyStopping:
    def __init__(self, patience):
        self.patience = 0

    def stop_early(self, last_metric): pass


class LossEarlyStopping(EarlyStopping):
    def __init__(self, patience):
        super().__init__(patience)
        self.count = 0
        self.losses = [float("inf")]

    def stop_early(self, current_loss):
        if current_loss >= self.losses[-1]:
            self.count += 1
            self.losses.append(current_loss)
            if self.count == self.patience:
                return True
            else:
                return False
        else:
            self.losses.append(current_loss)
            self.count = 0
            return False
