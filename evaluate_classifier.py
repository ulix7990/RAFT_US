import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import argparse
import os

from convgru_classifier import ConvGRUClassifier

# Dummy Dataset for demonstration (same as in train_classifier.py)
class OpticalFlowDataset(Dataset):
    def __init__(self, data_dir, num_classes=5, sequence_length=10, img_size=(368, 496)):
        self.data_dir = data_dir
        self.num_classes = num_classes
        self.sequence_length = sequence_length
        self.img_size = img_size
        self.flow_channels = 2 # Optical flow has 2 channels (dx, dy)

        self.samples = []
        for i in range(20): # Simulate 20 test samples
            dummy_flow_sequence = np.random.rand(sequence_length, self.flow_channels, img_size[0], img_size[1]).astype(np.float32)
            dummy_label = np.random.randint(0, num_classes)
            self.samples.append((dummy_flow_sequence, dummy_label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        flow_sequence, label = self.samples[idx]
        return torch.from_numpy(flow_sequence), torch.tensor(label, dtype=torch.long)


def evaluate_classifier(args):
    # Hyperparameters (must match training)
    input_dim = 2
    hidden_dims = [64, 128]
    kernel_size = 3
    n_layers = 2
    num_classes = args.num_classes
    batch_size = args.batch_size

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset and DataLoader
    dataset = OpticalFlowDataset(data_dir=args.data_dir, num_classes=num_classes, sequence_length=args.sequence_length, img_size=args.img_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Model
    model = ConvGRUClassifier(input_dim, hidden_dims, kernel_size, n_layers, num_classes).to(device)
    
    # Load trained model weights
    if not os.path.exists(args.model_path):
        print(f"Error: Model checkpoint not found at {args.model_path}")
        print("Please train the model first using train_classifier.py")
        return
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    correct = 0
    total = 0
    print("Starting evaluation...")
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Accuracy on test data: {accuracy:.2f}%")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate a ConvGRU classifier for optical flow sequences.")
    parser.add_argument('--data_dir', type=str, default='./saved_optical_flow', help='Directory containing optical flow .npy files')
    parser.add_argument('--num_classes', type=int, default=5, help='Number of classes for classification')
    parser.add_argument('--sequence_length', type=int, default=10, help='Length of optical flow sequences')
    parser.add_argument('--img_size', type=int, nargs=2, default=[368, 496], help='Height and width of optical flow frames')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for evaluation')
    parser.add_argument('--model_path', type=str, default='./convgru_classifier.pth', help='Path to the trained model checkpoint')

    args = parser.parse_args()
    evaluate_classifier(args)
