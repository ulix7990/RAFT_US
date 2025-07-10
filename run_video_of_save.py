import sys
sys.path.append('core')

import argparse
import os
import cv2
import numpy as np
import torch
from PIL import Image
import shutil

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

def run(args):
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))

    model = model.module
    model.to(DEVICE)
    model.eval()

    # Create and clear output directory
    if os.path.exists(args.output_path):
        shutil.rmtree(args.output_path)
    os.makedirs(args.output_path)

    cap = cv2.VideoCapture(args.input_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {args.input_path}")
        return

    frame_idx = 0
    ret, frame1_bgr = cap.read()
    if not ret:
        print("Error: Could not read first frame.")
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
        output_path_full = os.path.join(args.output_path, output_filename)
        np.save(output_path_full, flo_numpy)
        print(f"Saved {output_path_full}")

        # Move to the next frame pair
        frame1_bgr = frame2_bgr
        frame_idx += 1

    cap.release()
    print("Video processing complete.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--input_path', help="path to input video file (e.g., .avi)")
    parser.add_argument('--output_path', default='output_of_data', help='path to output directory for optical flow data')
    parser.add_argument('--interval', type=int, default=1, help='number of frames to skip between comparisons')
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    args = parser.parse_args()

    run(args)
