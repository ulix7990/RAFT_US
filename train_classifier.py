import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
import os
import glob
import argparse
import gc
import wandb
import cv2
from tqdm import tqdm
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from collections import Counter
from torch.utils.data import WeightedRandomSampler
import random

from core.convgru_classifier import ConvGRUClassifier

class OpticalFlowDataset(Dataset):
    def __init__(self, data_dir, sequence_length=10, resize_to=None, preprocessing_mode='resize', crop_location='center'):
        self.sequence_length = sequence_length
        self.samples = []
        self.resize_to = resize_to
        self.preprocessing_mode = preprocessing_mode
        self.crop_location = crop_location
        self.crop_positions = ['top-left', 'top-center', 'top-right', 'middle-left', 'center', 'middle-right', 'bottom-left', 'bottom-center', 'bottom-right']

        self.data_dir = os.path.abspath(data_dir)
        print(f"[DEBUG] Dataset initialized with data_dir: {self.data_dir}")

        class_dirs = sorted(glob.glob(os.path.join(self.data_dir, 'class*')))
        if not class_dirs:
            print(f"⚠️ Warning: No 'class*' directories found in {self.data_dir}.")
            self.label_mapping = {}
            self.num_actual_classes = 0
            return

        self.label_mapping = {}
        unique_raw_labels = set()
        for class_dir in class_dirs:
            try:
                raw_class_label = int(class_dir.split('class')[-1])
                unique_raw_labels.add(raw_class_label)
            except ValueError:
                continue

        for i, raw_label in enumerate(sorted(unique_raw_labels)):
            self.label_mapping[raw_label] = i

        self.num_actual_classes = len(self.label_mapping)

        for class_dir in class_dirs:
            try:
                raw_class_label = int(class_dir.split('class')[-1])
                mapped_label = self.label_mapping[raw_class_label]
            except:
                continue

            npz_files = sorted(glob.glob(os.path.join(class_dir, "seq*", "*.npz")))
            for npz_path in npz_files:
                self.samples.append((npz_path, mapped_label))

        print(f"[DEBUG] Total loaded .npz samples: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def _get_crop_coordinates(self, H_old, W_old, H_new, W_new, location):
        if location == 'random':
            location = random.choice(self.crop_positions)

        if 'top' in location:
            top = 0
        elif 'middle' in location or 'center' in location:
            top = (H_old - H_new) // 2
        elif 'bottom' in location:
            top = H_old - H_new
        
        if 'left' in location:
            left = 0
        elif 'center' in location:
            left = (W_old - W_new) // 2
        elif 'right' in location:
            left = W_old - W_new
            
        return top, left

    def __getitem__(self, idx):
        npz_path, label = self.samples[idx]

        try:
            data = np.load(npz_path)
            flow_sequence = data['flow_sequence']
            data.close()
        except Exception as e:
            print(f"⚠️ Failed to load {npz_path}: {e}")
            dummy = torch.zeros((self.sequence_length, 2, 64, 64), dtype=torch.float32)
            return dummy, torch.tensor(0, dtype=torch.long)

        if self.resize_to is not None:
            processed_flows = []
            H_new, W_new = self.resize_to
            T, H_old, W_old, _ = flow_sequence.shape

            if self.preprocessing_mode == 'resize':
                for t in range(T):
                    frame = flow_sequence[t]
                    channels = []
                    for c in range(frame.shape[2]):
                        resized_c = cv2.resize(frame[:, :, c], (W_new, H_new), interpolation=cv2.INTER_LINEAR)
                        channels.append(resized_c)
                    resized_frame = np.stack(channels, axis=2)
                    processed_flows.append(resized_frame)

            elif self.preprocessing_mode == 'crop':
                if H_new > H_old or W_new > W_old:
                    raise ValueError("Crop size must be smaller than original frame size.")
                
                top, left = self._get_crop_coordinates(H_old, W_old, H_new, W_new, self.crop_location)

                for t in range(T):
                    frame = flow_sequence[t]
                    cropped_frame = frame[top:top+H_new, left:left+W_new, :]
                    processed_flows.append(cropped_frame)
            
            flow_sequence = np.array(processed_flows)

        flow_tensor = torch.from_numpy(flow_sequence).permute(0, 3, 1, 2).float()

        T = flow_tensor.shape[0]
        if T > self.sequence_length:
            flow_tensor = flow_tensor[:self.sequence_length]
        elif T < self.sequence_length:
            pad_len = self.sequence_length - T
            _, C, H, W = flow_tensor.shape
            pad_tensor = torch.zeros((pad_len, C, H, W), dtype=torch.float32)
            flow_tensor = torch.cat([flow_tensor, pad_tensor], dim=0)

        return flow_tensor, torch.tensor(label, dtype=torch.long)

def validate(model, dataloader, criterion, device):
    model.eval()
    val_loss = 0.0
    correct_predictions = 0
    total_predictions = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device).float(), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total_predictions += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            del inputs, labels, outputs, predicted, loss
            torch.cuda.empty_cache()
            gc.collect()

    avg_val_loss = val_loss / len(dataloader)
    accuracy = 100 * correct_predictions / total_predictions

    f1 = f1_score(all_labels, all_preds, average='macro') * 100

    return avg_val_loss, accuracy, f1

def test_classifier(model, dataloader, criterion, device):
    model.eval()
    test_loss = 0.0
    correct_predictions = 0
    total_predictions = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device).float(), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total_predictions += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            del inputs, labels, outputs, predicted, loss
            torch.cuda.empty_cache()
            gc.collect()

    avg_test_loss = test_loss / len(dataloader)
    accuracy = 100 * correct_predictions / total_predictions

    f1 = f1_score(all_labels, all_preds, average='macro') * 100

    return avg_test_loss, accuracy, f1

def train_classifier(args):
    wandb.init(project="optical-flow-classification", config=args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[INFO] GPU 사용 가능 여부: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"[INFO] 사용 중인 GPU: {torch.cuda.get_device_name(0)}")
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.empty_cache()
        print("[INFO] GPU 메모리를 정리했습니다.")
        print("[INFO] 현재 메모리 상태 요약:\n", torch.cuda.memory_summary(device=None, abbreviated=True))

    input_dim = 2
    hidden_dims = [32]
    kernel_size = 3
    n_layers = 1

    learning_rate = args.learning_rate
    num_epochs = args.epochs
    batch_size = args.batch_size
    weight_decay = args.weight_decay
    patience = args.patience
    dropout_rate = args.dropout_rate

    resize_tuple = (args.resize_h, args.resize_w) if args.resize_h and args.resize_w else None

    full_dataset = OpticalFlowDataset(
        data_dir=args.data_dir,
        sequence_length=args.sequence_length,
        resize_to=resize_tuple,
        preprocessing_mode=args.preprocessing_mode,
        crop_location=args.crop_location
    )

    print(f"불러온 전체 데이터 샘플 수: {len(full_dataset)}")
    if len(full_dataset) == 0:
        print("⚠️ 데이터셋이 비어 있습니다. 데이터 경로를 확인하세요.")
        wandb.finish()
        return

    try:
        seq0, label0 = full_dataset[0]
        T, C, H, W = seq0.shape
        print(f"[INFO] 첫 샘플 해상도: H={H}, W={W} (참고: T={T}, C={C})")
        print(f"[INFO] 첫 샘플의 매핑된 클래스 레이블: {label0.item()}")
    except Exception as e:
        print(f"[ERROR] 데이터셋에서 첫 샘플을 가져오는 데 실패했습니다: {e}")
        print("데이터셋 초기화 또는 파일 로드에 문제가 있을 수 있습니다.")
        wandb.finish()
        return

    actual_num_classes = full_dataset.num_actual_classes
    print(f"[INFO] 데이터셋 내 실제 고유 클래스 개수: {actual_num_classes}")
    print(f"[INFO] `--num_classes` 인자 값 (모델 정의용): {args.num_classes}")

    if actual_num_classes != args.num_classes:
        print(f"❗❗❗ Warning: `--num_classes` argument ({args.num_classes}) does not match the actual number of unique classes in the dataset ({actual_num_classes}).")
        print(f"Using `actual_num_classes` ({actual_num_classes}) for model definition.")
        num_classes_for_model = actual_num_classes
    else:
        num_classes_for_model = args.num_classes

    train_ratio = 0.8
    val_ratio = 0.1
    test_ratio = 0.1

    assert train_ratio + val_ratio + test_ratio == 1.0, "학습, 검증, 테스트 비율의 합은 1.0이어야 합니다."

    indices = np.arange(len(full_dataset))
    labels = [sample[1] for sample in full_dataset.samples]

    train_indices, temp_indices, _, temp_labels = train_test_split(
        indices, labels,
        test_size=(val_ratio + test_ratio),
        random_state=42,
        stratify=labels
    )

    val_indices, test_indices, _, _ = train_test_split(
        temp_indices, temp_labels,
        test_size=(test_ratio / (val_ratio + test_ratio)),
        random_state=42,
        stratify=temp_labels
    )

    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)
    test_dataset = Subset(full_dataset, test_indices)

    print(f"[INFO] 데이터셋 분할: 학습 {len(train_dataset)}개, 검증 {len(val_dataset)}개, 테스트 {len(test_dataset)}개")

    train_labels = [full_dataset.samples[i][1] for i in train_dataset.indices]
    val_labels = [full_dataset.samples[i][1] for i in val_dataset.indices]
    test_labels = [full_dataset.samples[i][1] for i in test_dataset.indices]

    train_distribution = Counter(train_labels)
    val_distribution = Counter(val_labels)
    test_distribution = Counter(test_labels)

    print(f"[INFO] Train dataset class distribution: {sorted(train_distribution.items())}")
    print(f"[INFO] Validation dataset class distribution: {sorted(val_distribution.items())}")
    print(f"[INFO] Test dataset class distribution: {sorted(test_distribution.items())}")

    label_counts = Counter(label for _, label in full_dataset.samples)
    sample_weights = [1.0 / label_counts[label] for _, label in full_dataset.samples]
    train_indices = train_dataset.indices
    train_weights = [sample_weights[i] for i in train_indices]
    sampler = WeightedRandomSampler(train_weights, num_samples=len(train_indices), replacement=True)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = ConvGRUClassifier(input_dim, hidden_dims, kernel_size, n_layers, num_classes_for_model, dropout_rate=dropout_rate).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    wandb.watch(model, log="all")

    try:
        first_inputs, first_labels = next(iter(train_dataloader))
    except StopIteration:
        print("⚠️ 학습 데이터셋에서 배치를 가져올 수 없습니다. 데이터가 부족하거나 DataLoader 설정 문제일 수 있습니다.")
        wandb.finish()
        return

    b, t, c, h, w = first_inputs.shape
    print(f"[INFO] 모델 입력(첫 학습 배치): B={b}, T={t}, C={c}, H={h}, W={w}")
    print(f"[INFO] 첫 학습 배치의 매핑된 레이블: {first_labels.tolist()}")

    with torch.no_grad():
        try:
            _ = model(first_inputs.to(device).float())
            print("[DEBUG] 모델 포워드 패스 성공")
        except RuntimeError as e:
            print(f"[ERROR] 모델 포워드 패스 중 런타임 오류 발생: {e}")
            print("입력 텐서의 형태, 모델 정의 또는 GPU 메모리 부족을 확인하세요.")
            wandb.finish()
            return

    print("Starting training...")
    best_val_accuracy = -1.0
    epochs_no_improve = 0
    global_step = 0

    for epoch in range(num_epochs):
        model.train()
        train_running_loss = 0.0
        train_correct_predictions = 0
        train_total_predictions = 0

        for i, (inputs, labels) in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")):
            inputs, labels = inputs.to(device).float(), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_running_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            train_total_predictions += labels.size(0)
            train_correct_predictions += (predicted == labels).sum().item()

            wandb.log({"batch/train_loss": loss.item()}, step=global_step)
            global_step += 1

            del inputs, labels, outputs, predicted, loss
            torch.cuda.empty_cache()
            gc.collect()

        epoch_train_loss = train_running_loss / len(train_dataloader)
        epoch_train_accuracy = 100 * train_correct_predictions / train_total_predictions
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_accuracy:.2f}%")
        wandb.log({"epoch/train_loss": epoch_train_loss, "epoch/train_accuracy": epoch_train_accuracy}, step=global_step - 1)

        val_loss, val_accuracy, val_f1 = validate(model, val_dataloader, criterion, device)
        print(f"Epoch {epoch+1}/{num_epochs} - Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%, Val F1: {val_f1:.2f}%")
        wandb.log({"epoch/val_loss": val_loss, "epoch/val_accuracy": val_accuracy, "epoch/val_f1": val_f1}, step=global_step - 1)

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            epochs_no_improve = 0
            torch.save(model.state_dict(), args.model_save_path)
            print(f"👍 Best validation accuracy improved. Model saved to {args.model_save_path}")
            wandb.run.summary["best_val_accuracy"] = best_val_accuracy
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"[INFO] Early stopping triggered after {patience} epochs with no improvement.")
            break

    print("Training complete.")

    print("Starting final testing...")
    model.load_state_dict(torch.load(args.model_save_path))
    test_loss, test_accuracy, test_f1 = test_classifier(model, test_dataloader, criterion, device)
    print(f"Final Test Loss: {test_loss:.4f}, Final Test Acc: {test_accuracy:.2f}%, Final Test F1: {test_f1:.2f}%")
    wandb.log({"final_test_loss": test_loss, "final_test_accuracy": test_accuracy, "final_test_f1": test_f1})

    wandb.finish()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a ConvGRU classifier for optical flow sequences.")
    parser.add_argument('--data_dir', type=str, default='./saved_optical_flow', help='Directory containing the class/seq structured optical flow .npy files')
    parser.add_argument('--num_classes', type=int, default=3, help='Number of classes for classification. Will be automatically adjusted if mismatch with dataset.')
    parser.add_argument('--sequence_length', type=int, default=10, help='Fixed length of optical flow sequences for training')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for optimizer')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay (L2 regularization) for optimizer')
    parser.add_argument('--patience', type=int, default=10, help='Patience for early stopping')
    parser.add_argument('--dropout_rate', type=float, default=0.5, help='Dropout rate for the classifier')
    parser.add_argument('--model_save_path', type=str, default='./convgru_classifier.pth', help='Path to save the trained model')
    parser.add_argument('--resize_h', type=int, default=None, help='Resize/Crop height (H) for input frames')
    parser.add_argument('--resize_w', type=int, default=None, help='Resize/Crop width (W) for input frames')
    parser.add_argument('--preprocessing_mode', type=str, default='resize', choices=['resize', 'crop'], help='Preprocessing mode: resize or crop')
    parser.add_argument('--crop_location', type=str, default='center', 
                        choices=['top-left', 'top-center', 'top-right', 'middle-left', 'center', 'middle-right', 'bottom-left', 'bottom-center', 'bottom-right', 'random'], 
                        help='Crop location if preprocessing_mode is crop.')

    args = parser.parse_args()
    train_classifier(args)