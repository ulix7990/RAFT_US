import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import glob
import argparse

from core.convgru_classifier import ConvGRUClassifier

class OpticalFlowDataset(Dataset):
    def __init__(self, data_dir, sequence_length=10):
        self.sequence_length = sequence_length
        self.samples = []
        
        # Find all sequence directories and parse their class and path
        class_dirs = glob.glob(os.path.join(data_dir, 'class*'))
        for class_dir in class_dirs:
            class_label = int(class_dir.split('class')[-1])
            seq_dirs = glob.glob(os.path.join(class_dir, 'seq*'))
            for seq_dir in seq_dirs:
                self.samples.append((seq_dir, class_label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        seq_dir, class_label = self.samples[idx]
        
        flow_files = sorted(glob.glob(os.path.join(seq_dir, 'flow_*.npy')))
        
        # Load flow files into a list of numpy arrays
        flow_arrays = [np.load(f) for f in flow_files]
        
        # Stack arrays and permute to (T, C, H, W)
        # Original .npy is (H, W, C), so permute to (C, H, W) first
        flow_sequence = torch.from_numpy(np.array(flow_arrays)).permute(0, 3, 1, 2)

        # Pad or truncate the sequence to a fixed length
        current_len = flow_sequence.shape[0]
        if current_len > self.sequence_length:
            # Truncate
            flow_sequence = flow_sequence[:self.sequence_length]
        elif current_len < self.sequence_length:
            # Pad with zeros
            padding_len = self.sequence_length - current_len
            # Get C, H, W from the first frame
            _, C, H, W = flow_sequence.shape
            padding = torch.zeros((padding_len, C, H, W), dtype=flow_sequence.dtype)
            flow_sequence = torch.cat([flow_sequence, padding], dim=0)
            
        # Convert label to tensor
        label = torch.tensor(class_label, dtype=torch.long)
        
        return flow_sequence, label


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
    # Note: num_classes and img_size are no longer needed for dataset creation
    dataset = OpticalFlowDataset(data_dir=args.data_dir, sequence_length=args.sequence_length)
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
    parser.add_argument('--data_dir', type=str, default='./saved_optical_flow', help='Directory containing the class/seq structured optical flow .npy files')
    parser.add_argument('--num_classes', type=int, default=5, help='Number of classes for classification')
    parser.add_argument('--sequence_length', type=int, default=10, help='Fixed length of optical flow sequences for training')
    # img_size is no longer needed as it's inferred from the data
    # parser.add_argument('--img_size', type=int, nargs=2, default=[368, 496], help='Height and width of optical flow frames')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for training')
    parser.add_argument('--model_save_path', type=str, default='./convgru_classifier.pth', help='Path to save the trained model')

    args = parser.parse_args()
    train_classifier(args)
