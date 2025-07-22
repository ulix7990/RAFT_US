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
    """Processes a single video file to extract optical flow."""
    # --- Custom directory logic ---
    # 1. Parse class label from filename
    try:
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        class_label = base_name.split('_')[-1]
        class_dir_name = f"class{class_label}"
    except IndexError:
        print(f"Error: Could not parse class label from filename: {video_path}")
        print("Filename must end with '_<number>' (e.g., 'my_video_1.mp4').")
        return

    # 2. Create class directory
    class_path = os.path.join(args.output_path, class_dir_name)
    os.makedirs(class_path, exist_ok=True)

    # 3. Determine next sequence number
    existing_seq_dirs = [d for d in os.listdir(class_path) if os.path.isdir(os.path.join(class_path, d)) and d.startswith('seq')]
    if not existing_seq_dirs:
        next_seq_num = 1
    else:
        last_seq_num = max([int(d.replace('seq', '')) for d in existing_seq_dirs])
        next_seq_num = last_seq_num + 1
    
    # 4. Create sequence directory
    seq_dir_name = f"seq{next_seq_num}"
    final_output_path = os.path.join(class_path, seq_dir_name)
    os.makedirs(final_output_path)
    
    print(f"\nProcessing video: {video_path}")
    print(f"Detected class: {class_label}")
    print(f"Saving to sequence directory: {final_output_path}")
    # --- End of custom directory logic ---

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    frame_idx = 0
    ret, frame1_bgr = cap.read()
    if not ret:
        print(f"Error: Could not read first frame from {video_path}.")
        return

    while True:
        frame2_bgr = None
        for _ in range(args.interval):
            ret, temp_frame = cap.read()
            if not ret:
                break # End of video or error
            frame2_bgr = temp_frame # Keep the last frame read as frame2

        if frame2_bgr is None: # No more frames or error during interval read
            break

        # Convert BGR to RGB for RAFT model
        frame1_rgb = cv2.cvtColor(frame1_bgr, cv2.COLOR_BGR2RGB)
        frame2_rgb = cv2.cvtColor(frame2_bgr, cv2.COLOR_BGR2RGB)

        # Load images as torch tensors
        image1_torch = load_image(frame1_rgb)
        image2_torch = load_image(frame2_rgb)

        padder = InputPadder(image1_torch.shape)
        image1_padded, image2_padded = padder.pad(image1_torch, image2_torch)

        with torch.no_grad():
            _, flow_up = model(image1_padded, image2_padded, iters=20, test_mode=True)
            
        # Unpad the flow to match the original image size
        flow_up_unpadded = padder.unpad(flow_up)

        # Convert flow tensor to numpy array
        flo_numpy = flow_up_unpadded[0].permute(1, 2, 0).cpu().numpy()
        
        # Save the optical flow data as .npy file
        output_filename = f"flow_{frame_idx:04d}.npy"
        output_path_full = os.path.join(final_output_path, output_filename)
        np.save(output_path_full, flo_numpy)

        # Move to the next frame pair
        frame1_bgr = frame2_bgr
        frame_idx += 1

    cap.release()
    print(f"Finished processing {video_path}. Total frames processed: {frame_idx}")

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
