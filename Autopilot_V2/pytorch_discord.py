
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

class LambdaLayer(nn.Module):
    def __init__(self, func):
        super(LambdaLayer, self).__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)

class MyModel(nn.Module):
    def __init__(self, image_x=100, image_y=100):
        super(MyModel, self).__init__()
        # This structure must match your desired design
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=5, stride=2),
            nn.BatchNorm2d(24),
            nn.ReLU(),
            nn.Conv2d(24, 36, kernel_size=5, stride=2),
            nn.BatchNorm2d(36),
            nn.ReLU(),
            nn.Conv2d(36, 48, kernel_size=5, stride=2),
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.Conv2d(48, 64, kernel_size=3, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        
        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1600, 1164),  # Adjust based on feature map size after conv layers
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(1164, 100),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(50, 10),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(10, 1)  # Output: Steering angle
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

def transpose_data(data):
    return np.moveaxis(data, -1, 0)

if __name__ == "__main__":
    # Load, shuffle, split
    print("Loading data...")
    features, labels = loadFromPickle()
    print("Shuffling data...")
    features, labels = shuffle(features, labels)
    print("Splitting data...")
    train_x_ori, test_x, train_y_ori, test_y = train_test_split(
        features, labels, random_state=0, test_size=0.1
    )
    train_x, val_x, train_y, val_y = train_test_split(
        train_x_ori, train_y_ori, random_state=0, test_size=0.2
    )

    train_x = torch.from_numpy(np.array([transpose_data(x) for x in train_x])).float()
    val_x   = torch.from_numpy(np.array([transpose_data(x) for x in val_x])).float()
    test_x  = torch.from_numpy(np.array([transpose_data(x) for x in test_x])).float()
    train_y = torch.from_numpy(train_y).float()
    val_y   = torch.from_numpy(val_y).float()
    test_y  = torch.from_numpy(test_y).float()

    # # Reshape: (N, C, H, W) for PyTorch
    # train_x = torch.from_numpy(train_x.reshape(-1, 100, 100, 3)).float()
    # test_x  = torch.from_numpy(test_x.reshape(-1, 100, 100, 3)).float()
    # train_y = torch.from_numpy(train_y).float()
    # test_y  = torch.from_numpy(test_y).float()

    print(train_x.shape, train_x[0])

    batch_size = 3000
    # Datasets and loaders
    train_data = TensorDataset(train_x, train_y)
    test_data  = TensorDataset(test_x, test_y)
    val_data   = TensorDataset(val_x, val_y)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_data, batch_size=batch_size)
    val_loader   = DataLoader(val_data, batch_size=batch_size)

    # Model setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MyModel(100, 100)
    model = torch.nn.DataParallel(model)
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    i = 0
    # Training loop
    for epoch in range(30):
        model.train()
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(x_batch).view(-1)
            loss = criterion(outputs, y_batch)
            if i % 10 == 0:
                print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
            loss.backward()
            optimizer.step()
            i += 1
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                outputs = model(x_batch).view(-1)
                val_loss += criterion(outputs, y_batch).item()
        val_loss /= len(val_loader)
        print(f"Epoch {epoch+1}, Validation Loss: {val_loss:.4f}")

    test_loss = 0.0
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            outputs = model(x_batch).view(-1)
            test_loss += criterion(outputs, y_batch).item()
        test_loss /= len(test_loader)
        print(f"Test Loss: {test_loss:.4f}")
    # Print model summary (simple approach)
    print(model)

    # Save model weights
    torch.save(model.state_dict(), "/workspace/models/Autopilot_10.pth")