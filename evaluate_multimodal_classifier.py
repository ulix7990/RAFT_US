import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import argparse
import os
import glob

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

def evaluate_classifier(args):
    # Hyperparameters
    input_dim = 2
    hidden_dims = [64, 128]
    kernel_size = 3
    n_layers = 2
    num_classes = args.num_classes
    embedding_dim = args.embedding_dim
    batch_size = args.batch_size

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = OpticalFlowDataset(data_dir=args.data_dir, sequence_length=args.sequence_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model = MultimodalConvGRUClassifier(input_dim, hidden_dims, kernel_size, n_layers, num_classes, embedding_dim).to(device)
    
    if not os.path.exists(args.model_path):
        print(f"Error: Model checkpoint not found at {args.model_path}")
        return
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    correct = 0
    total = 0
    all_labels = []
    all_predictions = []

    print("Starting evaluation...")
    with torch.no_grad():
        for flow_sequences, labels in dataloader:
            flow_sequences, labels = flow_sequences.to(device), labels.to(device)
            outputs = model(flow_sequences, labels)
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    accuracy = 100 * correct / total
    print(f"Accuracy on test data: {accuracy:.2f}%")

    try:
        from sklearn.metrics import confusion_matrix, classification_report
        print("\nClassification Report:")
        print(classification_report(all_labels, all_predictions))
        print("Confusion Matrix:")
        print(confusion_matrix(all_labels, all_predictions))
    except ImportError:
        print("\nTo see a detailed report, please install scikit-learn: pip install -U scikit-learn")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate a Multimodal ConvGRU classifier.")
    parser.add_argument('--data_dir', type=str, default='./saved_optical_flow', help='Directory of the evaluation dataset')
    parser.add_argument('--num_classes', type=int, default=5, help='Number of classes')
    parser.add_argument('--embedding_dim', type=int, default=16, help='Dimension of the class embedding')
    parser.add_argument('--sequence_length', type=int, default=10, help='Fixed length of optical flow sequences')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for evaluation')
    parser.add_argument('--model_path', type=str, default='./multimodal_convgru_classifier.pth', help='Path to the trained model checkpoint')

    args = parser.parse_args()
    evaluate_classifier(args)
