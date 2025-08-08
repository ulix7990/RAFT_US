
import sys
sys.path.append('core')

import argparse
import os
import cv2
import numpy as np
import torch
from PIL import Image
import glob
from tqdm import tqdm

# RAFT 및 분류기 모델 import
from raft import RAFT
from convgru_classifier import ConvGRUClassifier
from utils.utils import InputPadder

DEVICE = 'cuda'

def load_image(imfile_or_array):
    """이미지 파일 또는 numpy 배열을 RAFT 입력 형식에 맞는 텐서로 변환합니다."""
    if isinstance(imfile_or_array, str):
        img = np.array(Image.open(imfile_or_array)).astype(np.uint8)
    else:
        img = imfile_or_array.astype(np.uint8)

    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)

def calculate_motion_score(flow_data_frame):
    """단일 옵티컬 플로우 프레임의 평균 움직임 강도를 계산합니다."""
    magnitude = np.sqrt(np.sum(flow_data_frame**2, axis=-1))
    return np.mean(magnitude)

def find_best_window(flow_sequence, window_size):
    """전체 옵티컬 플로우 시퀀스에서 가장 움직임이 활발한 윈도우를 찾습니다."""
    num_frames = len(flow_sequence)
    if num_frames < window_size:
        return 0, num_frames

    motion_scores = [calculate_motion_score(frame) for frame in flow_sequence]

    max_score = -1
    best_start_index = 0

    current_window_score = sum(motion_scores[:window_size])
    max_score = current_window_score

    for i in range(1, num_frames - window_size + 1):
        current_window_score = current_window_score - motion_scores[i-1] + motion_scores[i + window_size - 1]
        if current_window_score > max_score:
            max_score = current_window_score
            best_start_index = i
            
    return best_start_index, best_start_index + window_size

def run_inference(args):
    """추론 파이프라인을 실행합니다."""
    
    # 1. 모델 로드
    # RAFT 모델
    raft_model = torch.nn.DataParallel(RAFT(args))
    raft_model.load_state_dict(torch.load(args.raft_model))
    raft_model = raft_model.module
    raft_model.to(DEVICE)
    raft_model.eval()

    # 분류기 모델
    classifier_model = ConvGRUClassifier(
        input_dim=2, 
        hidden_dims=[64], 
        kernel_size=3, 
        n_layers=1, 
        num_classes=args.num_classes
    ).to(DEVICE)
    classifier_model.load_state_dict(torch.load(args.classifier_model))
    classifier_model.eval()
    
    print("--- All models loaded successfully ---")

    # 2. 비디오 처리 및 옵티컬 플로우 추출
    print(f"--- Processing video: {args.video_path} ---")
    cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {args.video_path}")
        return

    all_flows = []
    ret, frame1_bgr = cap.read()
    if not ret:
        print(f"Error: Could not read first frame from {args.video_path}.")
        return

    pbar = tqdm(desc="Extracting Optical Flow", unit="frame")
    while True:
        frame2_bgr = None
        # 학습 때와 동일한 간격(interval)으로 프레임 읽기
        for _ in range(args.interval):
            ret, temp_frame = cap.read()
            if not ret:
                break
            frame2_bgr = temp_frame
        if frame2_bgr is None:
            break

        frame1_rgb = cv2.cvtColor(frame1_bgr, cv2.COLOR_BGR2RGB)
        frame2_rgb = cv2.cvtColor(frame2_bgr, cv2.COLOR_BGR2RGB)
        
        image1_torch = load_image(frame1_rgb)
        image2_torch = load_image(frame2_rgb)
        
        padder = InputPadder(image1_torch.shape)
        image1_padded, image2_padded = padder.pad(image1_torch, image2_torch)
        
        with torch.no_grad():
            _, flow_up = raft_model(image1_padded, image2_padded, iters=20, test_mode=True)
        
        flow_up = padder.unpad(flow_up)[0].permute(1, 2, 0).cpu().numpy()
        all_flows.append(flow_up)
        
        frame1_bgr = frame2_bgr
        pbar.update(1)

    pbar.close()
    cap.release()

    if not all_flows:
        print("Warning: No optical flow frames were generated.")
        return

    # 3. 가장 활동적인 시퀀스 선택
    full_flow_sequence = np.stack(all_flows, axis=0)
    start_idx, end_idx = find_best_window(full_flow_sequence, args.sequence_length)
    best_flow_window = full_flow_sequence[start_idx:end_idx]
    
    print(f"--- Found best sequence of {len(best_flow_window)} frames (from index {start_idx} to {end_idx}) ---")

    # 4. 분류기 입력을 위한 텐서 준비
    flow_tensor = torch.from_numpy(best_flow_window).permute(0, 3, 1, 2).float() # (T, C, H, W)

    # 리사이즈 (학습 때와 동일하게)
    if args.resize_h and args.resize_w:
        H_new, W_new = args.resize_h, args.resize_w
        resized_flows = []
        for t in range(flow_tensor.shape[0]):
            frame = flow_tensor[t].permute(1, 2, 0).numpy() # (H, W, C)
            resized_frame = cv2.resize(frame, (W_new, H_new), interpolation=cv2.INTER_LINEAR)
            resized_flows.append(torch.from_numpy(resized_frame).permute(2, 0, 1)) # (C, H, W)
        flow_tensor = torch.stack(resized_flows, dim=0)

    # 패딩 또는 트리밍 (학습 때와 동일하게)
    current_len = flow_tensor.shape[0]
    if current_len > args.sequence_length:
        flow_tensor = flow_tensor[:args.sequence_length]
    elif current_len < args.sequence_length:
        pad_len = args.sequence_length - current_len
        _, C, H, W = flow_tensor.shape
        pad_tensor = torch.zeros((pad_len, C, H, W), dtype=flow_tensor.dtype)
        flow_tensor = torch.cat([flow_tensor, pad_tensor], dim=0)

    # 배치 차원 추가 및 디바이스로 전송
    flow_tensor = flow_tensor.unsqueeze(0).to(DEVICE) # (1, T, C, H, W)

    # 5. 예측 수행
    with torch.no_grad():
        outputs = classifier_model(flow_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        _, predicted_idx = torch.max(probabilities, 1)
        confidence = probabilities[0][predicted_idx.item()]

    # train_classifier.py의 label_mapping을 참고하여 실제 클래스 값으로 변환 가능
    # 예: label_map = {0: 80, 1: 120, 2: 150} -> predicted_class = label_map[predicted_idx.item()]
    # 여기서는 인덱스를 직접 출력합니다.
    
    print("\n--- Inference Result ---")
    print(f"Predicted Class Index: {predicted_idx.item()}")
    print(f"Confidence: {confidence:.2%}")
    print("------------------------")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run inference pipeline for video classification.")
    # 경로 인자
    parser.add_argument('--video_path', type=str, required=True, help='Path to the input video file for inference.')
    parser.add_argument('--raft_model', type=str, default='./models/raft-sintel.pth', help='Path to the pre-trained RAFT model.')
    parser.add_argument('--classifier_model', type=str, default='./convgru_classifier.pth', help='Path to the trained ConvGRU classifier model.')
    
    # 파라미터 인자 (학습 때와 동일해야 함)
    parser.add_argument('--num_classes', type=int, default=3, help='Number of classes the model was trained on.')
    parser.add_argument('--sequence_length', type=int, default=10, help='Sequence length the model was trained on.')
    parser.add_argument('--interval', type=int, default=5, help='Frame interval used during optical flow extraction.')
    parser.add_argument('--resize_h', type=int, default=256, help='Resize height for input frames (must match training).')
    parser.add_argument('--resize_w', type=int, default=448, help='Resize width for input frames (must match training).')

    # RAFT 모델용 인자
    parser.add_argument('--small', action='store_true', help='use small RAFT model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficient correlation implementation')

    args = parser.parse_args()
    
    run_inference(args)
