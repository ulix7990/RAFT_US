import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split # random_split 추가
import numpy as np
import os
import glob
import argparse
import gc
import wandb # wandb 다시 import

# ConvGRUClassifier는 별도의 파일(core/convgru_classifier.py)에 있다고 가정합니다.
from core.convgru_classifier import ConvGRUClassifier

# OpticalFlowDataset 클래스는 이전과 동일하게 유지됩니다.
# (데이터 로딩, 레이블 매핑, 시퀀스 패딩/자르기, __len__, __getitem__ 포함)
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

# 검증 함수
def validate(model, dataloader, criterion, device):
    model.eval() # 모델을 평가 모드로 설정
    val_loss = 0.0
    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad(): # 검증 단계에서는 그래디언트 계산 비활성화
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

# 테스트 함수
def test_classifier(model, dataloader, criterion, device):
    model.eval() # 모델을 평가 모드로 설정
    test_loss = 0.0
    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad(): # 테스트 단계에서는 그래디언트 계산 비활성화
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
    # WandB 초기화
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
    hidden_dims = [16, 32] # ConvGRU Classifier의 hidden_dims
    kernel_size = 3
    n_layers = 2
    
    learning_rate = args.learning_rate
    num_epochs = args.epochs
    batch_size = args.batch_size
    
    # 데이터셋 로드
    full_dataset = OpticalFlowDataset(data_dir=args.data_dir, sequence_length=args.sequence_length)
    
    print(f"불러온 전체 데이터 샘플 수: {len(full_dataset)}")
    if len(full_dataset) == 0:
        print("⚠️ 데이터셋이 비어 있습니다. 데이터 경로를 확인하세요.")
        wandb.finish() # WandB 실행 종료
        return

    # 첫 샘플 정보 출력 (데이터 유효성 검사)
    try:
        seq0, label0 = full_dataset[0]
        T, C, H, W = seq0.shape
        print(f"[INFO] 첫 샘플 해상도: H={H}, W={W} (참고: T={T}, C={C})")
        print(f"[INFO] 첫 샘플의 매핑된 클래스 레이블: {label0.item()}")
    except Exception as e:
        print(f"[ERROR] 데이터셋에서 첫 샘플을 가져오는 데 실패했습니다: {e}")
        print("데이터셋 초기화 또는 파일 로드에 문제가 있을 수 있습니다.")
        wandb.finish() # WandB 실행 종료
        return

    # 실제 데이터셋의 클래스 개수 확인 및 모델 클래스 수 조정
    actual_num_classes = full_dataset.num_actual_classes
    print(f"[INFO] 데이터셋 내 실제 고유 클래스 개수: {actual_num_classes}")
    print(f"[INFO] `--num_classes` 인자 값 (모델 정의용): {args.num_classes}")

    if actual_num_classes != args.num_classes:
        print(f"❗❗❗ Warning: `--num_classes` argument ({args.num_classes}) does not match the actual number of unique classes in the dataset ({actual_num_classes}).")
        print(f"Using `actual_num_classes` ({actual_num_classes}) for model definition.")
        num_classes_for_model = actual_num_classes
    else:
        num_classes_for_model = args.num_classes

    # 데이터셋 분할 (Train, Validation, Test)
    # 비율 설정 (예: 80% 학습, 10% 검증, 10% 테스트)
    train_ratio = 0.8
    val_ratio = 0.1
    test_ratio = 0.1

    # 합계가 1이 되는지 확인
    assert train_ratio + val_ratio + test_ratio == 1.0, "학습, 검증, 테스트 비율의 합은 1.0이어야 합니다."

    total_size = len(full_dataset)
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    test_size = total_size - train_size - val_size # 나머지

    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(42) # 재현성을 위해 시드 설정
    )

    print(f"[INFO] 데이터셋 분할: 학습 {len(train_dataset)}개, 검증 {len(val_dataset)}개, 테스트 {len(test_dataset)}개")

    # DataLoader 생성
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False) # 검증은 셔플하지 않음
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False) # 테스트는 셔플하지 않음

    # 모델, 손실 함수, 옵티마이저
    model = ConvGRUClassifier(input_dim, hidden_dims, kernel_size, n_layers, num_classes_for_model).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # WandB에 모델 아키텍처 로깅
    wandb.watch(model, log="all")

    # === 사전 배치/메모리/입력 크기 체크 (train_dataloader 사용) ===
    try:
        first_inputs, first_labels = next(iter(train_dataloader)) # 학습 데이터로 첫 배치 가져오기
    except StopIteration:
        print("⚠️ 학습 데이터셋에서 배치를 가져올 수 없습니다. 데이터가 부족하거나 DataLoader 설정 문제일 수 있습니다.")
        wandb.finish()
        return

    b, t, c, h, w = first_inputs.shape
    print(f"[INFO] 모델 입력(첫 학습 배치): B={b}, T={t}, C={c}, H={h}, W={w}")
    print(f"[INFO] 첫 학습 배치의 매핑된 레이블: {first_labels.tolist()}")

    with torch.no_grad():
        try:
            _ = model(first_inputs.to(device).float()) # .float() 명시적으로 추가
            print("[DEBUG] 모델 포워드 패스 성공")
        except RuntimeError as e:
            print(f"[ERROR] 모델 포워드 패스 중 런타임 오류 발생: {e}")
            print("입력 텐서의 형태, 모델 정의 또는 GPU 메모리 부족을 확인하세요.")
            wandb.finish()
            return

    # 학습 루프
    print("Starting training...")
    best_val_accuracy = -1.0 # 최고 검증 정확도 추적

    for epoch in range(num_epochs):
        model.train() # 모델을 학습 모드로 설정
        train_running_loss = 0.0
        train_correct_predictions = 0
        train_total_predictions = 0

        for i, (inputs, labels) in enumerate(train_dataloader):
            inputs, labels = inputs.to(device).float(), labels.to(device) # .float() 명시적으로 추가

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_running_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            train_total_predictions += labels.size(0)
            train_correct_predictions += (predicted == labels).sum().item()

            # WandB에 배치 손실 로깅
            wandb.log({"batch/train_loss": loss.item()}, step=epoch * len(train_dataloader) + i)

            del inputs, labels, outputs, predicted, loss
            torch.cuda.empty_cache()
            gc.collect()

        # 에포크 학습 결과
        epoch_train_loss = train_running_loss / len(train_dataloader)
        epoch_train_accuracy = 100 * train_correct_predictions / train_total_predictions
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_accuracy:.2f}%")
        wandb.log({"epoch/train_loss": epoch_train_loss, "epoch/train_accuracy": epoch_train_accuracy}, step=epoch)

        # 검증 단계
        val_loss, val_accuracy = validate(model, val_dataloader, criterion, device)
        print(f"Epoch {epoch+1}/{num_epochs} - Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")
        wandb.log({"epoch/val_loss": val_loss, "epoch/val_accuracy": val_accuracy}, step=epoch)

        # 최고 검증 정확도 모델 저장
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), args.model_save_path)
            print(f"👍 Best validation accuracy improved. Model saved to {args.model_save_path}")
            wandb.run.summary["best_val_accuracy"] = best_val_accuracy # WandB summary 업데이트

    print("Training complete.")
    
    # 학습 완료 후 테스트
    print("Starting final testing...")
    # 최고 성능 모델 로드 (선택 사항, 마지막 에포크 모델 사용도 가능)
    # model.load_state_dict(torch.load(args.model_save_path)) # Best 모델로 테스트하려면 이 줄의 주석을 해제
    test_loss, test_accuracy = test_classifier(model, test_dataloader, criterion, device)
    print(f"Final Test Loss: {test_loss:.4f}, Final Test Acc: {test_accuracy:.2f}%")
    wandb.log({"final_test_loss": test_loss, "final_test_accuracy": test_accuracy})

    # WandB 실행 종료
    wandb.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a ConvGRU classifier for optical flow sequences.")
    parser.add_argument('--data_dir', type=str, default='./saved_optical_flow', help='Directory containing the class/seq structured optical flow .npy files')
    parser.add_argument('--num_classes', type=int, default=3, help='Number of classes for classification. Will be automatically adjusted if mismatch with dataset.')
    parser.add_argument('--sequence_length', type=int, default=10, help='Fixed length of optical flow sequences for training')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for optimizer')
    parser.add_argument('--model_save_path', type=str, default='./convgru_classifier.pth', help='Path to save the trained model')
    # img_size 인자는 더 이상 필요하지 않으므로 제거하거나 주석 처리합니다.
    # parser.add_argument('--img_size', type=int, nargs=2, default=[368, 496], help='Height and width of optical flow frames') 

    args = parser.parse_args()
    train_classifier(args)