import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import glob
import argparse

from core.multimodal_convgru_classifier import MultimodalConvGRUClassifier

class OpticalFlowDataset(Dataset):
    def __init__(self, data_dir, sequence_length=10):
        self.sequence_length = sequence_length
        self.samples = []
        
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
        
        flow_arrays = [np.load(f) for f in flow_files]
        
        flow_sequence = torch.from_numpy(np.array(flow_arrays)).permute(0, 3, 1, 2)

        current_len = flow_sequence.shape[0]
        if current_len > self.sequence_length:
            flow_sequence = flow_sequence[:self.sequence_length]
        elif current_len < self.sequence_length:
            padding_len = self.sequence_length - current_len
            _, C, H, W = flow_sequence.shape
            padding = torch.zeros((padding_len, C, H, W), dtype=flow_sequence.dtype)
            flow_sequence = torch.cat([flow_sequence, padding], dim=0)
            
        label = torch.tensor(class_label, dtype=torch.long)
        
        return flow_sequence, label

def train_classifier(args):
    # Hyperparameters
    input_dim = 2
    hidden_dims = [64, 128]
    kernel_size = 3
    n_layers = 2
    num_classes = args.num_classes
    embedding_dim = args.embedding_dim
    learning_rate = 0.001
    num_epochs = args.epochs
    batch_size = args.batch_size

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = OpticalFlowDataset(data_dir=args.data_dir, sequence_length=args.sequence_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = MultimodalConvGRUClassifier(input_dim, hidden_dims, kernel_size, n_layers, num_classes, embedding_dim).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print("Starting training...")
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for i, (flow_sequences, labels) in enumerate(dataloader):
            flow_sequences, labels = flow_sequences.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(flow_sequences, labels)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(dataloader):.4f}")

    print("Training complete.")

    torch.save(model.state_dict(), args.model_save_path)
    print(f"Model saved to {args.model_save_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a Multimodal ConvGRU classifier.")
    parser.add_argument('--data_dir', type=str, default='./saved_optical_flow', help='Directory of the dataset')
    parser.add_argument('--num_classes', type=int, default=5, help='Number of classes')
    parser.add_argument('--embedding_dim', type=int, default=16, help='Dimension of the class embedding')
    parser.add_argument('--sequence_length', type=int, default=10, help='Fixed length of optical flow sequences')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for training')
    parser.add_argument('--model_save_path', type=str, default='./multimodal_convgru_classifier.pth', help='Path to save the trained model')

    args = parser.parse_args()
    train_classifier(args)
