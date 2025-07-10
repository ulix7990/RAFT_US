import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os

from convgru_classifier import ConvGRUClassifier

# Dummy Dataset for demonstration
class OpticalFlowDataset(Dataset):
    def __init__(self, data_dir, num_classes=5, sequence_length=10, img_size=(368, 496)):
        self.data_dir = data_dir
        self.num_classes = num_classes
        self.sequence_length = sequence_length
        self.img_size = img_size
        self.flow_channels = 2 # Optical flow has 2 channels (dx, dy)

        # In a real scenario, you would load actual file paths and labels
        # For this example, we'll simulate data
        self.samples = []
        for i in range(100): # Simulate 100 samples
            # Each sample is a sequence of optical flow data and a random label
            dummy_flow_sequence = np.random.rand(sequence_length, self.flow_channels, img_size[0], img_size[1]).astype(np.float32)
            dummy_label = np.random.randint(0, num_classes)
            self.samples.append((dummy_flow_sequence, dummy_label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        flow_sequence, label = self.samples[idx]
        return torch.from_numpy(flow_sequence), torch.tensor(label, dtype=torch.long)


def train_classifier(args):
    # Hyperparameters
    input_dim = 2  # Optical flow has 2 channels (dx, dy)
    hidden_dims = [64, 128] # Example hidden dimensions for ConvGRU layers
    kernel_size = 3
    n_layers = 2
    num_classes = args.num_classes
    learning_rate = 0.001
    num_epochs = args.epochs
    batch_size = args.batch_size

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset and DataLoader
    dataset = OpticalFlowDataset(data_dir=args.data_dir, num_classes=num_classes, sequence_length=args.sequence_length, img_size=args.img_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Model, Loss, and Optimizer
    model = ConvGRUClassifier(input_dim, hidden_dims, kernel_size, n_layers, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    print("Starting training...")
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(dataloader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(dataloader):.4f}")

    print("Training complete.")

    # Save the trained model
    torch.save(model.state_dict(), args.model_save_path)
    print(f"Model saved to {args.model_save_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a ConvGRU classifier for optical flow sequences.")
    parser.add_argument('--data_dir', type=str, default='./saved_optical_flow', help='Directory containing optical flow .npy files')
    parser.add_argument('--num_classes', type=int, default=5, help='Number of classes for classification')
    parser.add_argument('--sequence_length', type=int, default=10, help='Length of optical flow sequences')
    parser.add_argument('--img_size', type=int, nargs=2, default=[368, 496], help='Height and width of optical flow frames')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for training')
    parser.add_argument('--model_save_path', type=str, default='./convgru_classifier.pth', help='Path to save the trained model')

    args = parser.parse_args()
    train_classifier(args)
