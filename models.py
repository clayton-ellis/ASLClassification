import torch.nn as nn
import torch.optim.optimizer
from torchvision.transforms import RandomRotation
import torch
from utils import *

class ASL_Classifier(nn.Module):
    def __init__(self, dropout_rate = 0.2, device = 'cpu'):
        super().__init__()
        self.device = device
        self.dropout_rate = dropout_rate

        kernel_size = 3

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=kernel_size, padding_mode='zeros', padding = 2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=kernel_size, padding_mode='zeros', padding = 2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=kernel_size, padding_mode='zeros', padding = 2)

        self.dense1 = nn.Linear(10368, 128)
        self.dense2 = nn.Linear(128, 26)

        self.train_loss_history = []
        self.val_loss_history = []
        self.to(device)

    def forward(self, x: torch.tensor):
        x = self.conv1(x)
        x = nn.MaxPool2d((2,2))(x)
        x = nn.ReLU()(x)
        x = nn.Dropout2d(self.dropout_rate)(x)
        x = self.conv2(x)
        x = nn.MaxPool2d((2,2))(x)
        x = nn.ReLU()(x)
        x = nn.Dropout2d(self.dropout_rate)(x)
        x = self.conv3(x)
        x = nn.MaxPool2d((2,2))(x)
        x = nn.ReLU()(x)
        x = nn.Dropout2d(self.dropout_rate)(x)
        x = nn.Flatten()(x)
        x = self.dense1(x)
        x = nn.ReLU()(x)
        x = nn.Dropout(self.dropout_rate)(x)
        x = self.dense2(x)
        x = nn.Softmax()(x)
        return x
    
    def fit(self, loader: DataLoader, val_loader: DataLoader = None, epochs: int = 50, lr: float = 1e-3, patience: int = 10, verbose: bool = True):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        loss = nn.CrossEntropyLoss(reduction='sum').to(self.device)

        for epoch in range(epochs):
            
            running_loss = 0
            total_train = 0
            self.train()
            for data in tqdm(loader):
                optimizer.zero_grad()
                inputs, truths = data
                inputs = inputs.to(self.device)
                truths = truths.to(self.device)
                outputs = self.forward(inputs)
                curr_loss = loss(outputs, truths.squeeze(1))
                curr_loss.backward() # Calculate Derivatices

                optimizer.step()
                running_loss += curr_loss.item()
                total_train += inputs.size(0)

            epoch_loss = running_loss / total_train
            self.train_loss_history.append(epoch_loss)
            
            if val_loader:
                total_val = 0
                self.eval()
                val_loss = 0
                for data in tqdm(val_loader):
                    inputs, truths = data
                    inputs=inputs.to(self.device)
                    truths=truths.to(self.device)
                    outputs = self.forward(inputs)
                    curr_loss = loss(outputs, truths.squeeze(1))
                    val_loss += curr_loss.item()
                    total_val += inputs.size(0)


            if verbose:
                print(f'Epoch {epoch+1}: Training Loss = {epoch_loss}')
                if val_loader:
                    print(f'Epoch {epoch+1}: Validation Loss = {val_loss/total_val}')
                    self.val_loss_history.append(val_loss/total_val)


    def predict(self, test_data):
        preds = self.forward(test_data)
        preds = preds.argmax(dim = 1)
        return preds



