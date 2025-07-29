# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import Dataset, DataLoader
# import numpy as np
# import os
# import glob
# import argparse
# import wandb  # Import wandb

# from core.convgru_classifier import ConvGRUClassifier

# class OpticalFlowDataset(Dataset):
#     def __init__(self, data_dir, sequence_length=10):
#         self.sequence_length = sequence_length
#         self.samples = []
        
#         # Find all sequence directories and parse their class and path
#         class_dirs = glob.glob(os.path.join(data_dir, 'class*'))
#         for class_dir in class_dirs:
#             class_label = int(class_dir.split('class')[-1])
#             seq_dirs = glob.glob(os.path.join(class_dir, 'seq*'))
#             for seq_dir in seq_dirs:
#                 self.samples.append((seq_dir, class_label))

#     def __len__(self):
#         return len(self.samples)

#     def __getitem__(self, idx):
#         seq_dir, class_label = self.samples[idx]
        
#         flow_files = sorted(glob.glob(os.path.join(seq_dir, 'flow_*.npy')))
        
#         # Load flow files into a list of numpy arrays
#         flow_arrays = [np.load(f) for f in flow_files]
        
#         # Stack arrays and permute to (T, C, H, W)
#         # Original .npy is (H, W, C), so permute to (C, H, W) first
#         flow_sequence = torch.from_numpy(np.array(flow_arrays)).permute(0, 3, 1, 2)

#         # Pad or truncate the sequence to a fixed length
#         current_len = flow_sequence.shape[0]
#         if current_len > self.sequence_length:
#             # Truncate
#             flow_sequence = flow_sequence[:self.sequence_length]
#         elif current_len < self.sequence_length:
#             # Pad with zeros
#             padding_len = self.sequence_length - current_len
#             # Get C, H, W from the first frame
#             _, C, H, W = flow_sequence.shape
#             padding = torch.zeros((padding_len, C, H, W), dtype=flow_sequence.dtype)
#             flow_sequence = torch.cat([flow_sequence, padding], dim=0)
            
#         # Convert label to tensor
#         label = torch.tensor(class_label, dtype=torch.long)
        
#         return flow_sequence, label


# def train_classifier(args):
#     # Initialize wandb
#     wandb.init(project="optical-flow-classification", config=args)

#     # Hyperparameters
#     input_dim = 2  # Optical flow has 2 channels (dx, dy)
#     hidden_dims = [64, 128] # Example hidden dimensions for ConvGRU layers
#     kernel_size = 3
#     n_layers = 2
#     num_classes = args.num_classes
#     learning_rate = args.learning_rate  # Use learning_rate from args
#     num_epochs = args.epochs
#     batch_size = args.batch_size

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     # Dataset and DataLoader
#     dataset = OpticalFlowDataset(data_dir=args.data_dir, sequence_length=args.sequence_length)
#     dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

#     # Model, Loss, and Optimizer
#     model = ConvGRUClassifier(input_dim, hidden_dims, kernel_size, n_layers, num_classes).to(device)
#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.Adam(model.parameters(), lr=learning_rate)

#     # Log model architecture to wandb
#     wandb.watch(model, log="all")

#     # Training loop
#     print("Starting training...")
#     for epoch in range(num_epochs):
#         model.train()
#         running_loss = 0.0
#         for i, (inputs, labels) in enumerate(dataloader):
#             inputs, labels = inputs.to(device), labels.to(device)

#             optimizer.zero_grad()
#             outputs = model(inputs)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()

#             running_loss += loss.item()
#             # Log loss to wandb
#             wandb.log({"batch_loss": loss.item()})

#         epoch_loss = running_loss / len(dataloader)
#         print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")
#         # Log epoch loss to wandb
#         wandb.log({"epoch_loss": epoch_loss, "epoch": epoch + 1})

#     print("Training complete.")

#     # Save the trained model
#     torch.save(model.state_dict(), args.model_save_path)
#     print(f"Model saved to {args.model_save_path}")
    
#     # Finish wandb run
#     wandb.finish()


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description="Train a ConvGRU classifier for optical flow sequences.")
#     parser.add_argument('--data_dir', type=str, default='./saved_optical_flow', help='Directory containing the class/seq structured optical flow .npy files')
#     parser.add_argument('--num_classes', type=int, default=5, help='Number of classes for classification')
#     parser.add_argument('--sequence_length', type=int, default=10, help='Fixed length of optical flow sequences for training')
#     # img_size is no longer needed as it's inferred from the data
#     parser.add_argument('--img_size', type=int, nargs=2, default=[368, 496], help='Height and width of optical flow frames')
#     parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
#     parser.add_argument('--batch_size', type=int, default=4, help='Batch size for training')
#     parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for optimizer')
#     parser.add_argument('--model_save_path', type=str, default='./convgru_classifier.pth', help='Path to save the trained model')

#     args = parser.parse_args()
#     train_classifier(args)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import glob
import argparse
import gc

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
        flow_sequence = torch.from_numpy(np.array(flow_arrays)).permute(0, 3, 1, 2)

        # Pad or truncate the sequence to a fixed length
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
    # GPU 장치 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ✅ GPU 사용 여부 출력
    print(f"[INFO] GPU 사용 가능 여부: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"[INFO] 사용 중인 GPU: {torch.cuda.get_device_name(0)}")
        torch.cuda.empty_cache()  # ✅ 메모리 캐시 정리
        gc.collect()
        torch.cuda.empty_cache()
        print("[INFO] GPU 메모리를 정리했습니다.")
        print("[INFO] 현재 메모리 상태 요약:\n", torch.cuda.memory_summary(device=None, abbreviated=True))

    # Hyperparameters
    input_dim = 2
    # hidden_dims = [64, 128]
    hidden_dims = [32, 64]
    kernel_size = 3
    n_layers = 2
    num_classes = args.num_classes
    learning_rate = args.learning_rate
    num_epochs = args.epochs
    batch_size = args.batch_size

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset and DataLoader
    dataset = OpticalFlowDataset(data_dir=args.data_dir, sequence_length=args.sequence_length)
    
    # 디버깅용: 데이터 개수 출력
    print(f"불러온 데이터 샘플 수: {len(dataset)}")
    if len(dataset) == 0:
        print("⚠️ 데이터셋이 비어 있습니다. 데이터 경로를 확인하세요.")
        return
    
    # 첫 샘플 해상도 출력
    seq0, _ = dataset[0]             # (T, C, H, W)
    T, C, H, W = seq0.shape
    print(f"[INFO] 첫 샘플 해상도: H={H}, W={W} (참고: T={T}, C={C})")

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Model, Loss, Optimizer
    model = ConvGRUClassifier(input_dim, hidden_dims, kernel_size, n_layers, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # === 사전 배치/메모리/입력 크기 체크 ===
    try:
        first_inputs, first_labels = next(iter(dataloader))  # 첫 배치 가져오기
    except StopIteration:
        print("⚠️ 데이터셋에서 배치를 가져올 수 없습니다.")
        return

    # 첫 배치 크기 출력
    b, t, c, h, w = first_inputs.shape
    print(f"[INFO] 모델 입력(첫 배치): B={b}, T={t}, C={c}, H={h}, W={w}")

    # no_grad로 1회 forward → 메모리/shape 오류 조기 검출
    with torch.no_grad():
        try:
            _ = model(first_inputs.to(device))  # 필요하면 .float() 추가
            print("[DEBUG] 모델 입력 처리 성공")
        except RuntimeError as e:
            print(f"[ERROR] 입력 또는 메모리 오류 발생: {e}")
            return

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

        epoch_loss = running_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")

    print("Training complete.")

    # Save the model
    torch.save(model.state_dict(), args.model_save_path)
    print(f"Model saved to {args.model_save_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a ConvGRU classifier for optical flow sequences.")
    parser.add_argument('--data_dir', type=str, default='./saved_optical_flow', help='Directory containing the class/seq structured optical flow .npy files')
    parser.add_argument('--num_classes', type=int, default=5, help='Number of classes for classification')
    parser.add_argument('--sequence_length', type=int, default=10, help='Fixed length of optical flow sequences for training')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for optimizer')
    parser.add_argument('--model_save_path', type=str, default='./convgru_classifier.pth', help='Path to save the trained model')

    args = parser.parse_args()
    train_classifier(args)
