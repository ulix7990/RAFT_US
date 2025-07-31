import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import os
import glob
import argparse
import gc
import wandb

from core.convgru_classifier import ConvGRUClassifier

class OpticalFlowDataset(Dataset):
    def __init__(self, data_dir, sequence_length=10):
        self.sequence_length = sequence_length
        self.samples = []

        self.data_dir = os.path.abspath(data_dir)
        print(f"[DEBUG] Dataset initialized with data_dir: {self.data_dir}")

        class_dirs = sorted(glob.glob(os.path.join(self.data_dir, 'class*')))

        print(f"[DEBUG] Found class directories: {class_dirs}")
        if not class_dirs:
            print(f"⚠️ Warning: No 'class*' directories found in {self.data_dir}. Check your --data_dir path and directory naming.")
            self.label_mapping = {}
            self.num_actual_classes = 0
            return

        self.label_mapping = {}
        unique_raw_labels = set()

        for class_dir in class_dirs:
            try:
                class_label_str = class_dir.split('class')[-1]
                raw_class_label = int(class_label_str)
                unique_raw_labels.add(raw_class_label)
            except ValueError:
                print(f"⚠️ Warning: Could not parse valid class label number from '{class_dir}'. Skipping this directory.")
                continue

        sorted_raw_labels = sorted(list(unique_raw_labels))
        for i, raw_label in enumerate(sorted_raw_labels):
            self.label_mapping[raw_label] = i

        print(f"[INFO] Original class label mapping: {self.label_mapping}")
        self.num_actual_classes = len(self.label_mapping)
        print(f"[INFO] Actual number of unique classes in dataset: {self.num_actual_classes}")

        sample_count_in_init = 0
        for class_dir in class_dirs:
            try:
                raw_class_label = int(class_dir.split('class')[-1])
                mapped_class_label = self.label_mapping[raw_class_label]
            except (ValueError, KeyError):
                continue

            seq_dirs = sorted(glob.glob(os.path.join(class_dir, 'seq*')))
            if not seq_dirs:
                print(f"⚠️ Warning: No 'seq*' directories found in {class_dir}. Skipping this class directory.")
                continue

            for seq_dir in seq_dirs:
                flow_files = sorted(glob.glob(os.path.join(seq_dir, 'flow_*.npy')))
                if not flow_files:
                    print(f"⚠️ Warning: No 'flow_*.npy' files found in {seq_dir}. Skipping this sequence.")
                    continue

                self.samples.append((seq_dir, mapped_class_label))
                sample_count_in_init += 1

        print(f"[DEBUG] Total samples populated in __init__: {sample_count_in_init}")
        if not self.samples:
             print("❗❗❗ Critical Error: self.samples is empty after dataset initialization. No data will be loaded.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        seq_dir, class_label = self.samples[idx]

        flow_files = sorted(glob.glob(os.path.join(seq_dir, 'flow_*.npy')))

        flow_arrays = []
        for f in flow_files:
            try:
                flow_arrays.append(np.load(f))
            except Exception as e:
                print(f"⚠️ Warning: Failed to load file {f}. Error: {e}. Skipping this file.")
                continue

        if not flow_arrays:
            print(f"⚠️ Warning: No valid flow files loaded for sequence {seq_dir}. Returning dummy data.")
            _dummy_C, _dummy_H, _dummy_W = 2, 64, 64
            if hasattr(self, '_cached_h_w'):
                _dummy_H, _dummy_W = self._cached_h_w

            dummy_flow_sequence = torch.zeros((self.sequence_length, _dummy_C, _dummy_H, _dummy_W), dtype=torch.float32)
            dummy_label = torch.tensor(0, dtype=torch.long)
            if self.label_mapping:
                dummy_label = torch.tensor(min(self.label_mapping.values()), dtype=torch.long)
            return dummy_flow_sequence, dummy_label

        flow_sequence = torch.from_numpy(np.array(flow_arrays)).permute(0, 3, 1, 2)

        if not hasattr(self, '_cached_h_w'):
            self._cached_h_w = (flow_sequence.shape[2], flow_sequence.shape[3])

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

def validate(model, dataloader, criterion, device):
    model.eval()
    val_loss = 0.0
    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device).float(), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total_predictions += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

            del inputs, labels, outputs, predicted, loss
            torch.cuda.empty_cache()
            gc.collect()

    avg_val_loss = val_loss / len(dataloader)
    accuracy = 100 * correct_predictions / total_predictions
    return avg_val_loss, accuracy

def test_classifier(model, dataloader, criterion, device):
    model.eval()
    test_loss = 0.0
    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device).float(), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total_predictions += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

            del inputs, labels, outputs, predicted, loss
            torch.cuda.empty_cache()
            gc.collect()

    avg_test_loss = test_loss / len(dataloader)
    accuracy = 100 * correct_predictions / total_predictions
    return avg_test_loss, accuracy


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
    # hidden_dims = [16, 32]
    hidden_dims = [64]
    kernel_size = 3
    n_layers = 1

    learning_rate = args.learning_rate
    num_epochs = args.epochs
    batch_size = args.batch_size
    weight_decay = args.weight_decay # weight_decay 인자 추가

    full_dataset = OpticalFlowDataset(data_dir=args.data_dir, sequence_length=args.sequence_length)

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

    total_size = len(full_dataset)
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    test_size = total_size - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(42)
    )

    print(f"[INFO] 데이터셋 분할: 학습 {len(train_dataset)}개, 검증 {len(val_dataset)}개, 테스트 {len(test_dataset)}개")

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = ConvGRUClassifier(input_dim, hidden_dims, kernel_size, n_layers, num_classes_for_model).to(device)
    criterion = nn.CrossEntropyLoss()
    # weight_decay 인자를 optimizer에 추가했습니다.
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
    global_step = 0 # 전역 스텝 변수 추가

    for epoch in range(num_epochs):
        model.train()
        train_running_loss = 0.0
        train_correct_predictions = 0
        train_total_predictions = 0

        for i, (inputs, labels) in enumerate(train_dataloader):
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

            # WandB에 배치 손실 로깅 (전역 스텝 사용)
            wandb.log({"batch/train_loss": loss.item()}, step=global_step)
            global_step += 1 # 스텝 증가

            del inputs, labels, outputs, predicted, loss
            torch.cuda.empty_cache()
            gc.collect()

        epoch_train_loss = train_running_loss / len(train_dataloader)
        epoch_train_accuracy = 100 * train_correct_predictions / train_total_predictions
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_accuracy:.2f}%")
        # WandB에 에포크 결과 로깅 (마지막 전역 스텝 사용)
        wandb.log({"epoch/train_loss": epoch_train_loss, "epoch/train_accuracy": epoch_train_accuracy}, step=global_step - 1)


        val_loss, val_accuracy = validate(model, val_dataloader, criterion, device)
        print(f"Epoch {epoch+1}/{num_epochs} - Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")
        # WandB에 에포크 결과 로깅 (마지막 전역 스텝 사용)
        wandb.log({"epoch/val_loss": val_loss, "epoch/val_accuracy": val_accuracy}, step=global_step - 1)

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), args.model_save_path)
            print(f"👍 Best validation accuracy improved. Model saved to {args.model_save_path}")
            wandb.run.summary["best_val_accuracy"] = best_val_accuracy

    print("Training complete.")

    print("Starting final testing...")
    # model.load_state_dict(torch.load(args.model_save_path)) # Best 모델로 테스트하려면 이 줄의 주석을 해제
    test_loss, test_accuracy = test_classifier(model, test_dataloader, criterion, device)
    print(f"Final Test Loss: {test_loss:.4f}, Final Test Acc: {test_accuracy:.2f}%")
    wandb.log({"final_test_loss": test_loss, "final_test_accuracy": test_accuracy})

    wandb.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a ConvGRU classifier for optical flow sequences.")
    parser.add_argument('--data_dir', type=str, default='./saved_optical_flow', help='Directory containing the class/seq structured optical flow .npy files')
    parser.add_argument('--num_classes', type=int, default=3, help='Number of classes for classification. Will be automatically adjusted if mismatch with dataset.')
    parser.add_argument('--sequence_length', type=int, default=10, help='Fixed length of optical flow sequences for training')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs') # 에포크 수 100으로 변경
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for optimizer')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay (L2 regularization) for optimizer') # weight_decay 인자 추가
    parser.add_argument('--model_save_path', type=str, default='./convgru_classifier.pth', help='Path to save the trained model')

    args = parser.parse_args()
    train_classifier(args)