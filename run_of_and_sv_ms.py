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

from raft import RAFT
from utils.utils import InputPadder
import matplotlib.pyplot as plt

DEVICE = 'cuda'
MAX_FLOW_FRAMES = 600  # Optical flow 계산 쌍 최대 개수 제한

def load_image(imfile_or_array):
    if isinstance(imfile_or_array, str):
        img = np.array(Image.open(imfile_or_array)).astype(np.uint8)
    else:
        img = imfile_or_array.astype(np.uint8)

    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)

def calculate_motion_score(flow_data_frame, roi_width=None, roi_height=None):
    if roi_width is not None and roi_height is not None:
        h, w, _ = flow_data_frame.shape
        start_h = max(0, h - roi_height)
        end_h = h
        start_w = max(0, (w - roi_width) // 2)
        end_w = min(w, start_w + roi_width)
        
        if (end_h - start_h <= 0) or (end_w - start_w <= 0):
            roi_flow_data = flow_data_frame
        else:
            roi_flow_data = flow_data_frame[start_h:end_h, start_w:end_w, :]
    else:
        roi_flow_data = flow_data_frame

    magnitude = np.sqrt(np.sum(roi_flow_data**2, axis=-1))
    return np.mean(magnitude)

def find_best_window(flow_sequence, window_size, roi_width, roi_height):
    num_frames = len(flow_sequence)
    if num_frames < window_size:
        return 0, num_frames

    motion_scores = [calculate_motion_score(frame, None, None) for frame in flow_sequence]

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

def process_video_and_trim(video_path, model, args):
    """
    Processes a single video file to extract optical flow, finds the best sequence,
    and saves that trimmed sequence as one .npz archive.
    """
    try:
        video_basename = os.path.splitext(os.path.basename(video_path))[0]
        class_label = video_basename.split('_')[-1]
        class_dir_name = f"class{class_label}"
    except IndexError:
        print(f"Error: Could not parse class label from filename: {video_path}")
        return

    print(f"\nProcessing video: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    estimated_pairs = min(MAX_FLOW_FRAMES, max(0, (total_frames - 1) // args.interval))

    all_flows = []
    ret, frame1_bgr = cap.read()
    if not ret:
        print(f"Error: Could not read first frame from {video_path}.")
        return

    pbar = tqdm(total=estimated_pairs, desc=f"Processing {os.path.basename(video_path)}", leave=False)

    flow_count = 0
    while flow_count < MAX_FLOW_FRAMES:
        frame2_bgr = None
        for _ in range(args.interval):
            ret, temp_frame = cap.read()
            if not ret:
                break
            frame2_bgr = temp_frame
        if frame2_bgr is None:
            break

        # Optical flow 계산
        frame1_rgb = cv2.cvtColor(frame1_bgr, cv2.COLOR_BGR2RGB)
        frame2_rgb = cv2.cvtColor(frame2_bgr, cv2.COLOR_BGR2RGB)
        image1_torch = load_image(frame1_rgb)
        image2_torch = load_image(frame2_rgb)
        padder = InputPadder(image1_torch.shape)
        image1_padded, image2_padded = padder.pad(image1_torch, image2_torch)
        with torch.no_grad():
            _, flow_up = model(image1_padded, image2_padded, iters=20, test_mode=True)
        flow_up = padder.unpad(flow_up)[0].permute(1, 2, 0).cpu().numpy()
        all_flows.append(flow_up)
        frame1_bgr = frame2_bgr
        flow_count += 1
        pbar.update(1)

    pbar.close()
    cap.release()

    if not all_flows:
        print(f"Warning: No optical flow frames were generated for {video_path}.")
        return

    # best window 선택
    full_flow = np.stack(all_flows, axis=0)
    start_idx, end_idx = find_best_window(full_flow, args.sequence_length, args.roi_width, args.roi_height)
    best_window = full_flow[start_idx:end_idx]

    # === 추가: motion score 계산 및 그래프 저장 ===
    motion_scores = [
        calculate_motion_score(flow_frame, args.roi_width, args.roi_height)
        for flow_frame in full_flow
    ]

    plt.figure(figsize=(10, 4))
    plt.plot(motion_scores, label='Motion Score')
    plt.xlabel('Frame Index')
    plt.ylabel('Motion Score (Magnitude)')
    plt.title(f'Motion Score over Time - {video_basename}')
    plt.grid(True)
    plt.legend()

    motion_plot_path = os.path.join(args.output_path, class_dir_name, f"motion_plot_{video_basename}.png")
    os.makedirs(os.path.dirname(motion_plot_path), exist_ok=True)
    plt.savefig(motion_plot_path)
    plt.close()
    print(f"Saved motion score plot to {motion_plot_path}")

    # 원본 비디오 삭제
    try:
        os.remove(video_path)
        print(f"Deleted original video file: {video_path}")
    except Exception as e:
        print(f"Warning: Could not delete video file {video_path}. Error: {e}")

def run(args):
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))

    model = model.module
    model.to(DEVICE)
    model.eval()

    if os.path.isdir(args.input_path):
        video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv']
        video_files = []
        for ext in video_extensions:
            video_files.extend(glob.glob(os.path.join(args.input_path, ext)))
        
        if not video_files:
            print(f"No video files found in {args.input_path}")
            return

        print(f"Found {len(video_files)} videos to process.")
        for video_path in tqdm(sorted(video_files), desc="Processing videos"):
            process_video_and_trim(video_path, model, args)

    elif os.path.isfile(args.input_path):
        process_video_and_trim(args.input_path, model, args)
        
    else:
        print(f"Error: Input path {args.input_path} is not a valid file or directory.")
        return

    print("\nAll processing complete.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # From run_video_of_save.py
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--input_path', help="path to an input video file or a directory of video files")
    parser.add_argument('--interval', type=int, default=1, help='number of frames to skip between comparisons')
    
    # From trim_sequences.py
    parser.add_argument('--output_path', default='processed_sequences', help='Directory to save the final trimmed sequences')
    parser.add_argument('--sequence_length', type=int, default=10, help='The desired length of the output sequences.')
    parser.add_argument('--roi_width', type=int, default=None, help='Width of the ROI for motion calculation and final saving.')
    parser.add_argument('--roi_height', type=int, default=None, help='Height of the ROI for motion calculation and final saving.')

    # From RAFT args
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    args = parser.parse_args()

    run(args)
