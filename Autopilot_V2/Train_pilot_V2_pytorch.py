import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import pickle

def loadFromPickle():
    with open("features", "rb") as f:
        features = pickle.load(f)
    with open("labels", "rb") as f:
        labels = pickle.load(f)
    return features, labels

class Normalize(nn.Module):
    def forward(self, x):
        print(x.shape)
        return x / 127.5 - 1.0

class MyModel(nn.Module):
    def __init__(self, image_x=100, image_y=100):
        super(MyModel, self).__init__()
        # This structure must match your desired design
        self.net = nn.Sequential(
            # nn.Conv2d(1, 32, kernel_size=5, padding=0), nn.ReLU(),
            # nn.MaxPool2d(2),
            # nn.Conv2d(32, 32, kernel_size=5, padding=0), nn.ReLU(),
            # nn.MaxPool2d(2),
            # nn.Conv2d(32, 48, kernel_size=5, padding=0), nn.ReLU(),
            # nn.MaxPool2d(2),
            # nn.Conv2d(48, 64, kernel_size=3, padding=0), nn.ReLU(),
            # nn.MaxPool2d(2),
            # nn.Conv2d(64, 64, kernel_size=3, padding=0), nn.ReLU(),
            # nn.MaxPool2d(2),
            # nn.Conv2d(64, 64, kernel_size=3, padding=0), nn.ReLU(),
            # nn.MaxPool2d(2),
            # nn.Flatten(),
            # nn.Dropout(0.5),
            # nn.Linear(128 * (image_x // 64) * (image_y // 64), 1024),
            # nn.Linear(1024, 256),
            # nn.Linear(256, 64),
            # nn.Linear(64, 1)
            nn.Conv2d(1, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(128 * (image_x // 64) * (image_y // 64), 1024),
            nn.Linear(1024, 256),
            nn.Linear(256, 64),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)
    
if __name__ == "__main__":
    # Load, shuffle, split
    features, labels = loadFromPickle()
    train_x, test_x, train_y, test_y = train_test_split(
        features, labels, random_state=0, test_size=0.3, shuffle=True
    )

    # Reshape: (N, C, H, W) for PyTorch
    train_x = torch.from_numpy(train_x.reshape(-1, 1, 100, 100)).float()
    test_x  = torch.from_numpy(test_x.reshape(-1, 1, 100, 100)).float()
    train_y = torch.from_numpy(train_y).float()
    test_y  = torch.from_numpy(test_y).float()

    # Datasets and loaders
    train_data = TensorDataset(train_x, train_y)
    test_data  = TensorDataset(test_x, test_y)
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    test_loader  = DataLoader(test_data, batch_size=32)

    # Model setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MyModel(100, 100).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Training loop
    for epoch in range(20):
        model.train()
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(x_batch).view(-1)
            loss = criterion(outputs, y_batch)
            print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_batch, y_batch in test_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                outputs = model(x_batch).view(-1)
                val_loss += criterion(outputs, y_batch).item()
        val_loss /= len(test_loader)
        print(f"Epoch {epoch+1}, Validation Loss: {val_loss:.4f}")

    # Print model summary (simple approach)
    print(model)

    # Save model weights
    torch.save(model.state_dict(), "/workspace/models/Autopilot_10.pth")