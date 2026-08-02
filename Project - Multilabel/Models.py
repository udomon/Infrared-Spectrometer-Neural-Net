import torch.nn as nn
import torch

class FocalLoss(nn.Module):
    def __init__(self, alpha = 1.0, gamma = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        bce_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        p_t = torch.exp(-bce_loss)
        focal_term = (1 - p_t) ** self.gamma
        loss = self.alpha * focal_term * bce_loss
        return loss.mean()

class SpectrumNet(nn.Module):
    def __init__(self, input_size=1000, num_labels = 5):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, num_labels)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class SpectrumNetBigger(nn.Module):
    def __init__(self, input_size=1000, num_labels = 5):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(256, 64)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(64, num_labels)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x


class SpectrumCNN(nn.Module):
    def __init__(self, input_size=1000, num_labels = 5):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=15, padding=7)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=4)

        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=9, padding=4)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=4)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * 62, 64)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(64, num_labels)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        return x