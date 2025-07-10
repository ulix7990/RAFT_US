
import sys
import argparse
import os
import cv2
import glob
import numpy as np
import torch
from PIL import Image
import shutil

# RAFT core imports
sys.path.append('core')
from raft import RAFT
from utils.utils import InputPadder

# --- Configuration ---
DEVICE = 'cuda'
CROPPED_FRAMES_DIR = 'cropped_frames'
FLOW_OUTPUT_DIR = 'flow_output'
INTERVAL = 1

# --- Functions from save_cropped_frames.py ---

def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]
    iou_score = interArea / float(boxAArea + boxBArea - interArea)
    return iou_score

def detect_and_crop(video_path, output_dir):
    """
    Reads a video, detects the longest-lasting red grid, and saves the cropped frames.
    Returns the path to the output directory if successful, otherwise None.
    """
    print(f"Starting grid detection and cropping for {video_path}...")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return None

    # Clean and create output directory
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    frame_count = 0
    all_streaks = []
    active_streaks = {}
    next_rect_id = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        # Frame processing for grid detection
        blurred_frame = cv2.medianBlur(frame, 5)
        hsv = cv2.cvtColor(blurred_frame, cv2.COLOR_BGR2HSV)
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        lower_red2 = np.array([170, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = cv2.bitwise_or(mask1, mask2)
        
        kernel = np.ones((7, 7), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=10)
        mask = cv2.erode(mask, kernel, iterations=10)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        current_frame_detections = []
        for contour in contours:
            epsilon = 0.05 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            if len(approx) == 4:
                x, y, w, h = cv2.boundingRect(approx)
                if cv2.contourArea(approx) > 20000: # Area threshold
                    current_frame_detections.append([x, y, w, h])

        # Streak tracking logic
        matched_active_streaks_ids = set()
        for current_bbox in current_frame_detections:
            found_match = False
            for rect_id, streak_info in active_streaks.items():
                if iou(current_bbox, streak_info['bboxes'][-1]) > 0.5:
                    streak_info['end_frame'] = frame_count
                    streak_info['bboxes'].append(current_bbox)
                    matched_active_streaks_ids.add(rect_id)
                    found_match = True
                    break
            if not found_match:
                new_streak = {'id': next_rect_id, 'start_frame': frame_count, 'end_frame': frame_count, 'bboxes': [current_bbox], 'active': True}
                active_streaks[next_rect_id] = new_streak
                next_rect_id += 1

        streaks_to_deactivate = [rid for rid in active_streaks if rid not in matched_active_streaks_ids]
        for rect_id in streaks_to_deactivate:
            active_streaks[rect_id]['active'] = False
            all_streaks.append(active_streaks[rect_id])
            del active_streaks[rect_id]

    all_streaks.extend(active_streaks.values())

    # Find the longest streak
    longest_streak = max(all_streaks, key=lambda s: s['end_frame'] - s['start_frame'], default=None)

    if longest_streak:
        print(f"Longest grid streak found: {longest_streak['end_frame'] - longest_streak['start_frame'] + 1} frames.")
        avg_bbox = np.mean(longest_streak['bboxes'], axis=0).astype(int)
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # Reset video capture
        frame_idx_save = 0
        saved_count = 0
        
        while True:
            ret_save, frame_save = cap.read()
            if not ret_save:
                break
            frame_idx_save += 1
            if longest_streak['start_frame'] <= frame_idx_save <= longest_streak['end_frame']:
                x, y, w, h = avg_bbox
                cropped_frame = frame_save[y:y+h, x:x+w]
                output_filename = os.path.join(output_dir, f"frame_{frame_idx_save:05d}.png")
                cv2.imwrite(output_filename, cropped_frame)
                saved_count += 1
        
        print(f"Saved {saved_count} cropped frames to '{output_dir}'.")
        cap.release()
        return output_dir
    else:
        print("No persistent grid detected.")
        cap.release()
        return None

# --- Functions from arrow_vis.py ---

def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)

def viz_flow(img, flo, imfile1, output_dir):
    img_numpy = img[0].permute(1,2,0).cpu().numpy().astype(np.uint8)
    flo_numpy = flo[0].permute(1,2,0).cpu().numpy()
    
    img_bgr = cv2.cvtColor(img_numpy, cv2.COLOR_RGB2BGR)

    stride = 16
    h, w = img_bgr.shape[:2]
    for y in range(0, h, stride):
        for x in range(0, w, stride):
            dx, dy = flo_numpy[y, x]
            if np.sqrt(dx**2 + dy**2) > 1.0: # Draw only significant motion
                pt1 = (x, y)
                pt2 = (int(round(x + dx)), int(round(y + dy)))
                cv2.arrowedLine(img_bgr, pt1, pt2, (0, 255, 0), 1, tipLength=0.3)

    output_filename = os.path.basename(imfile1).replace('.png', '_flow.png')
    output_path = os.path.join(output_dir, output_filename)
    cv2.imwrite(output_path, img_bgr)

def calculate_and_visualize_flow(args, frames_path, output_dir):
    """
    Calculates and visualizes optical flow for a sequence of frames.
    """
    print(f"Starting optical flow calculation for frames in {frames_path}...")
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))

    model = model.module
    model.to(DEVICE)
    model.eval()

    # Clean and create output directory
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    with torch.no_grad():
        images = sorted(glob.glob(os.path.join(frames_path, '*.png')) + glob.glob(os.path.join(frames_path, '*.jpg')))
        
        if len(images) < 2:
            print("Not enough images found for flow calculation.")
            return

        for imfile1, imfile2 in zip(images[:-1], images[INTERVAL:]):
            image1 = load_image(imfile1)
            image2 = load_image(imfile2)

            padder = InputPadder(image1.shape)
            image1_padded, image2_padded = padder.pad(image1, image2)

            _, flow_up = model(image1_padded, image2_padded, iters=20, test_mode=True)
            
            # The flow is for the original image size, no need to unpad flow if we visualize on original image
            viz_flow(image1, flow_up, imfile1, output_dir)
    
    print(f"Saved flow visualizations to '{output_dir}'.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Detects a grid in a video, crops the frames, and then calculates optical flow.")
    parser.add_argument('--input_path', required=True, help="Path to the input video file")
    parser.add_argument('--output_path', required=True, help="Base directory for output frames")
    parser.add_argument('--model', required=True, help="Path to the RAFT model checkpoint")
    parser.add_argument('--interval', type=int, default=1, help="Interval between frames for optical flow calculation")
    parser.add_argument('--small', action='store_true', help='Use small RAFT model')
    parser.add_argument('--mixed_precision', action='store_true', help='Use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='Use efficient correlation implementation')
    args = parser.parse_args()

    # --- Setup output directories ---
    CROPPED_FRAMES_DIR = os.path.join(args.output_path, 'cropped_frames')
    FLOW_OUTPUT_DIR = os.path.join(args.output_path, 'flow_output')
    INTERVAL = args.interval

    # --- Step 1: Detect Grid and Crop Frames ---
    cropped_dir = detect_and_crop(args.input_path, CROPPED_FRAMES_DIR)

    # --- Step 2: Calculate Optical Flow ---
    if cropped_dir:
        calculate_and_visualize_flow(args, cropped_dir, FLOW_OUTPUT_DIR)
        print("Process finished successfully!")
        print(f"Cropped frames are in: '{CROPPED_FRAMES_DIR}'")
        print(f"Flow visualizations are in: '{FLOW_OUTPUT_DIR}'")
    else:
        print("Process failed or was aborted because no grid was detected.")
