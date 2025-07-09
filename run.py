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
OUTPUT_PATH = 'output'

def load_image(imfile_or_array):
    if isinstance(imfile_or_array, str):
        img = np.array(Image.open(imfile_or_array)).astype(np.uint8)
    else: # Assume it's a numpy array (from cv2.imread)
        img = imfile_or_array.astype(np.uint8)

    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)


def draw_flow_arrows(image, flow, stride=16):
    """ Draw arrows on the image to visualize the flow """
    h, w = image.shape[:2]
    # Ensure flow dimensions match image dimensions for drawing
    h_flow, w_flow = flow.shape[:2]
    h_draw, w_draw = min(h, h_flow), min(w, w_flow)

    for y in range(0, h_draw, stride):
        for x in range(0, w_draw, stride):
            dx, dy = flow[y, x]
            # Draw arrows for significant motion
            if np.sqrt(dx**2 + dy**2) > 1.0:
                pt1 = (x, y)
                pt2 = (int(round(x + dx)), int(round(y + dy)))
                cv2.arrowedLine(image, pt1, pt2, (0, 255, 0), 1, tipLength=0.3)
    return image

def run(args):
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))

    model = model.module
    model.to(DEVICE)
    model.eval()

    # Create and clear output directory
    if os.path.exists(OUTPUT_PATH):
        shutil.rmtree(OUTPUT_PATH)
    os.makedirs(OUTPUT_PATH)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Error: Could not open video file {args.video}")
        return

    frame_idx = 0
    ret, frame1_bgr = cap.read()
    if not ret:
        print("Error: Could not read first frame.")
        return

    while True:
        ret, frame2_bgr = cap.read()
        if not ret:
            break # End of video

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
        
        # Draw arrows on the original frame1 (BGR format)
        img_with_arrows = draw_flow_arrows(frame1_bgr.copy(), flo_numpy, stride=16)

        # Save the result
        output_filename = f"frame_{frame_idx:04d}_arrows.png"
        output_path_full = os.path.join(OUTPUT_PATH, output_filename)
        cv2.imwrite(output_path_full, img_with_arrows)
        print(f"Saved {output_path_full}")

        # Move to the next frame pair
        frame1_bgr = frame2_bgr
        frame_idx += 1

    cap.release()
    print("Video processing complete.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--video', help="path to input video file (e.g., .avi)")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    args = parser.parse_args()

    run(args)
