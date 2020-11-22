# LeNet model

from collections import OrderedDict
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

class LeNet(nn.Module):
    def __init__(self, activation=nn.ReLU(), batch_size=64, num_epochs=30, learning_rate=1e-2, learning_rate_decay=0.0):
        super(LeNet, self).__init__()
        self.input_height = 28
        self.input_width = 28
        self.model = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(
                in_channels=1, 
                out_channels=6, 
                kernel_size=5, 
                padding=2, 
                padding_mode='zeros',
            )),
            ('activation1', activation),
            ('pool1', nn.AvgPool2d(
                kernel_size=2, 
                stride=2,
            )),
            # No custom connections in this layer, unlike in the original paper
            ('conv2', nn.Conv2d(
                in_channels=6, 
                out_channels=16, 
                kernel_size=5, 
            )),
            ('activation2', activation),
            ('pool2', nn.AvgPool2d(
                kernel_size=2, 
                stride=2,
            )),
            ('conv3', nn.Conv2d(
                in_channels=16, 
                out_channels=120, 
                kernel_size=5, 
            )),
            ('activation3', activation),
            ('flatten', nn.Flatten()),
            ('linear1', nn.Linear(
                in_features=120,
                out_features=84,
            )),
            ('activation4', activation),
            ('linear2', nn.Linear(
                in_features=84,
                out_features=10,
            )),
        ]))
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.learning_rate_decay = learning_rate_decay

    def preprocess_X(self, raw_X):
        flat_X = torch.tensor(raw_X, dtype=torch.float32)
        return torch.reshape(flat_X, (-1, 1, self.input_height, self.input_width)) / 255

    def preprocess_train_inputs(self, raw_X, raw_y, val_prop=0.2):
        val_size = int(0.2 * raw_X.shape[0])
        indices = np.arange(raw_X.shape[0])
        np.random.shuffle(indices)
        train_X, val_X = raw_X[indices[val_size:]], raw_X[indices[: val_size]]
        train_y, val_y = raw_y[indices[val_size:]], raw_y[indices[: val_size]]

        train_dataset = TensorDataset(self.preprocess_X(train_X), 
            torch.tensor(train_y, dtype=torch.long))
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        val_dataset = TensorDataset(self.preprocess_X(val_X), 
            torch.tensor(val_y, dtype=torch.long))
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=True)
        return train_loader, val_loader

    def forward(self, X):
        return self.model(X)

    def predict(self, X):
        return torch.argmax(self.forward(X), dim=1)

    def train_on_dataset(self, raw_X, raw_y, save_path=None):
        train_loader, val_loader = self.preprocess_train_inputs(raw_X, raw_y)
        phase_data = [('train', train_loader), ('val', val_loader)]
        
        loss_fxn = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(
            self.model.parameters(), lr=self.learning_rate)
        for epoch in range(self.num_epochs):
            print(f'Epoch {epoch + 1}:')
            for phase, data_loader in phase_data:
                if phase == 'train':
                    self.model.train()
                else:
                    self.model.eval()

                total_loss = 0.0
                total_correct = 0.0
                total_size = 0.0

                for X, y in data_loader:
                    optimizer.zero_grad()
                    with torch.set_grad_enabled(phase == 'train'):
                        y_pred = self.forward(X)
                        loss = loss_fxn(y_pred, y)
                        total_loss += loss.item() * X.size(0)

                        class_pred = self.predict(X)
                        total_correct += (class_pred == y).sum().item()
                        total_size += X.size(0)
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                print(f'{phase.upper()} AVERAGE LOSS: {total_loss / total_size}')
                print(f'{phase.upper()} ACCURACY: {total_correct / total_size}')
        if save_path:
            torch.save(self.model.state_dict(), save_path)

    def predict_on_dataset(self, raw_X):
        X = self.preprocess_X(raw_X)
        return self.predict(X).numpy()

    def load_saved_state(self, path):
        self.model.load_state_dict(torch.load(path))
        self.model.eval()


