# LeNet model

from collections import OrderedDict

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

class LeNet(nn.Module):
    def __init__(self, activation=nn.Tanh(), batch_size=64, num_epochs=20, learning_rate=3e-1, learning_rate_decay=0.05):
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

    def preprocess_inputs(self, raw_X, raw_y):
        flat_X = torch.tensor(raw_X, dtype=torch.float32)
        dataset = TensorDataset(torch.reshape(flat_X, (-1, 1, self.input_height, self.input_width)) / 255, 
            torch.tensor(raw_y, dtype=torch.long))
        return DataLoader(dataset, batch_size=64, shuffle=True)

    def forward(self, X):
        return self.model(X)

    def predict(self, X):
        return torch.argmax(self.forward(X), dim=1)

    def train(self, train_X, train_y, val_X=None, val_y=None):
        print(f'Training Set Size: {train_X.shape[0]}')
        phase_data = [('train', self.preprocess_inputs(train_X, train_y))]
        if val_X and val_y:
            phase_data.append(('val', self.preprocess_inputs(val_X, val_y)))
        
        loss_fxn = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(
            self.model.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=self.learning_rate_decay)
        for epoch in range(self.num_epochs):
            print(f'Epoch {epoch + 1}:')
            for phase, data_loader in phase_data:
                if phase == 'train':
                    self.model.train()
                else:
                    self.model.eval()

                total_loss = 0.0
                total_correct = 0.0

                for X, y in data_loader:
                    optimizer.zero_grad()
                    with torch.set_grad_enabled(phase == 'train'):
                        y_pred = self.forward(X)
                        loss = loss_fxn(y_pred, y)
                        total_loss += loss.item() * X.size(0)

                        class_pred = self.predict(X)
                        total_correct += (class_pred == y).sum().item()
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                print(f'{phase.upper()} LOSS: {total_loss}')
                print(f'{phase.upper()} ACCURACY: {total_correct / train_X.shape[0]}')

            # TODO: evaluate and print
            # TODO: print accuracy
            # TODO: check the schedule
            scheduler.step()

            

        # save model




# TODO: make sure to make input (n, 1, h, w) rather than (n, h, w)
