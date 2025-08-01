import sys
sys.path.append('core')

import argparse
import os
import cv2
import numpy as np
import torch
from PIL import Image
import shutil
import glob

from raft import RAFT
from utils.utils import InputPadder

DEVICE = 'cuda'

def load_image(imfile_or_array):
    if isinstance(imfile_or_array, str):
        img = np.array(Image.open(imfile_or_array)).astype(np.uint8)
    else: # Assume it's a numpy array (from cv2.imread)
        img = imfile_or_array.astype(np.uint8)

    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)

def process_video(video_path, model, args):
    """Processes a single video file to extract optical flow and save as a single compressed .npz file."""
    # <<<--- 수정된 디렉토리 및 파일명 로직 시작 --->>>
    try:
        video_basename = os.path.splitext(os.path.basename(video_path))[0]
        class_label = video_basename.split('_')[-1]
        class_dir_name = f"class{class_label}"
    except IndexError:
        print(f"Error: Could not parse class label from filename: {video_path}")
        print("Filename must end with '_<number>' (e.g., 'my_video_1.mp4').")
        return

    # 클래스 디렉토리 생성
    class_path = os.path.join(args.output_path, class_dir_name)
    os.makedirs(class_path, exist_ok=True)

    # 출력 파일 경로 설정 (비디오 파일명 기반)
    output_npz_path = os.path.join(class_path, f"{video_basename}.npz")
    
    print(f"\nProcessing video: {video_path}")
    print(f"Detected class: {class_label}")
    print(f"Output will be saved to: {output_npz_path}")
    # <<<--- 수정된 디렉토리 및 파일명 로직 종료 --->>>

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    # <<<--- 모든 Flow를 저장할 리스트 초기화 --->>>
    all_flows = []

    ret, frame1_bgr = cap.read()
    if not ret:
        print(f"Error: Could not read first frame from {video_path}.")
        return

    while True:
        frame2_bgr = None
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
            _, flow_up = model(image1_padded, image2_padded, iters=20, test_mode=True)
            
        flow_up_unpadded = padder.unpad(flow_up)
        flo_numpy = flow_up_unpadded[0].permute(1, 2, 0).cpu().numpy()
        
        # <<<--- Flow를 파일에 바로 저장하는 대신 리스트에 추가 --->>>
        all_flows.append(flo_numpy)

        frame1_bgr = frame2_bgr

    cap.release()

    # <<<--- 비디오 처리 완료 후, 리스트에 쌓인 Flow를 하나의 파일로 압축 저장 --->>>
    if all_flows:
        # 리스트를 하나의 큰 NumPy 배열로 변환 (T, H, W, C)
        flow_sequence = np.stack(all_flows, axis=0)
        # 압축하여 .npz 파일로 저장
        np.savez_compressed(output_npz_path, flow_sequence=flow_sequence)
        print(f"Finished processing and saved {flow_sequence.shape[0]} flows to {output_npz_path}")
    else:
        print(f"Warning: No optical flow frames were generated for {video_path}.")


def run(args):
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))

    model = model.module
    model.to(DEVICE)
    model.eval()

    if os.path.isdir(args.input_path):
        print(f"Input path is a directory. Searching for videos...")
        video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv']
        video_files = []
        for ext in video_extensions:
            video_files.extend(glob.glob(os.path.join(args.input_path, ext)))
        
        if not video_files:
            print(f"No video files found in {args.input_path}")
            return

        print(f"Found {len(video_files)} videos to process.")
        for video_path in sorted(video_files):
            process_video(video_path, model, args)

    elif os.path.isfile(args.input_path):
        print(f"Input path is a single file.")
        process_video(args.input_path, model, args)
        
    else:
        print(f"Error: Input path {args.input_path} is not a valid file or directory.")
        return

    print("\nAll processing complete.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--input_path', help="path to an input video file or a directory of video files")
    parser.add_argument('--output_path', default='saved_optical_flow', help='path to output directory for optical flow data')
    parser.add_argument('--interval', type=int, default=1, help='number of frames to skip between comparisons')
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    args = parser.parse_args()

    run(args)