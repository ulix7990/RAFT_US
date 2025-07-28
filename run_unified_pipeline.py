import os
import shutil
import argparse
import glob
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import cv2
import wandb
from tqdm import tqdm

# Add 'core' to sys.path to import RAFT and ConvGRUClassifier
sys.path.append(os.path.join(os.path.dirname(__file__), 'core'))

# Import RAFT and ConvGRUClassifier after adding 'core' to path
from raft import RAFT
from convgru_classifier import ConvGRUClassifier
from utils.utils import InputPadder # Assuming utils.utils is in core/utils

# --- Configuration Variables ---
# Base directory of the RAFT_US project (automatically determined)
RAFT_PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

# Step 1: Video Preparation
INPUT_VIDEO_DIR = "/input_video_dir" # !!! IMPORTANT: Set your actual input video directory here !!!
PREPARED_VIDEO_DIR = os.path.join(RAFT_PROJECT_DIR, "data", "prepared_raft_videos")

# Step 2: Optical Flow Extraction
RAFT_MODEL_PATH = os.path.join(RAFT_PROJECT_DIR, "models", "raft-sintel.pth") # !!! IMPORTANT: Set your actual RAFT model path here !!!
OUTPUT_OPTICAL_FLOW_DIR = os.path.join(RAFT_PROJECT_DIR, "data", "saved_optical_flow")
OPTICAL_FLOW_INTERVAL = 1 # Number of frames to skip between comparisons (for run_video_of_save.py)
USE_SMALL_RAFT_MODEL = False # Set to True if using the small RAFT model

# Step 3: Sequence Trimming
PROCESSED_SEQUENCES_DIR = os.path.join(RAFT_PROJECT_DIR, "data", "processed_sequences")
SEQUENCE_LENGTH = 10 # Fixed length of optical flow sequences for training

# Step 4: Classifier Training
NUM_CLASSES = 5 # Number of classes for classification
CLASSIFIER_MODEL_SAVE_PATH = os.path.join(RAFT_PROJECT_DIR, "convgru_classifier.pth")
TRAINING_EPOCHS = 10
TRAINING_BATCH_SIZE = 4
LEARNING_RATE = 0.001

# Wandb Configuration
WANDB_PROJECT_NAME = "optical-flow-classification-unified"


# --- Step 1: Video Preparation Functions ---
def prepare_videos(input_dir, output_dir):
    """
    Prepares video files for the run_video_of_save.py script by renaming them
    to include the class label in the filename and copying them to a flat output directory.
    """
    print(f"--- Step 1: Preparing video files for RAFT ---")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    print(f"Scanning for video files in: {input_dir}")
    processed_count = 0

    for root, _, files in os.walk(input_dir):
        try:
            class_label = os.path.basename(root)
            if not class_label.isdigit():
                continue
        except Exception:
            continue

        for filename in files:
            if filename.lower().endswith(('.avi', '.mp4', '.mov', '.mkv')):
                original_filepath = os.path.join(root, filename)
                base_name, ext = os.path.splitext(filename)

                if base_name.endswith(f"_{class_label}"):
                    new_filename = filename
                else:
                    new_filename = f"{base_name}_{class_label}{ext}"

                destination_filepath = os.path.join(output_dir, new_filename)

                if os.path.exists(destination_filepath) and os.path.getsize(original_filepath) == os.path.getsize(destination_filepath):
                    # print(f"Skipping existing identical file: {new_filename}")
                    continue

                try:
                    shutil.copy2(original_filepath, destination_filepath)
                    print(f"Copied and renamed: {filename} -> {new_filename}")
                    processed_count += 1
                except Exception as e:
                    print(f"Error copying {original_filepath} to {destination_filepath}: {e}")

    print(f"Finished preparing videos. Total processed: {processed_count}")
    print(f"Prepared videos are in: {output_dir}")
    print("Step 1 complete.
")


# --- Step 2: Optical Flow Extraction Functions ---
DEVICE = 'cuda'

def load_image(imfile_or_array):
    if isinstance(imfile_or_array, str):
        img = np.array(Image.open(imfile_or_array)).astype(np.uint8)
    else: # Assume it's a numpy array (from cv2.imread)
        img = imfile_or_array.astype(np.uint8)

    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)

def process_single_video_for_flow(video_path, model, output_base_path, interval):
    """Processes a single video file to extract optical flow."""
    try:
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        class_label = base_name.split('_')[-1]
        class_dir_name = f"class{class_label}"
    except IndexError:
        print(f"Error: Could not parse class label from filename: {video_path}")
        print("Filename must end with '_<number>' (e.g., 'my_video_1.mp4').")
        return

    class_path = os.path.join(output_base_path, class_dir_name)
    os.makedirs(class_path, exist_ok=True)

    existing_seq_dirs = [d for d in os.listdir(class_path) if os.path.isdir(os.path.join(class_path, d)) and d.startswith('seq')]
    if not existing_seq_dirs:
        next_seq_num = 1
    else:
        last_seq_num = max([int(d.replace('seq', '')) for d in existing_seq_dirs])
        next_seq_num = last_seq_num + 1
    
    seq_dir_name = f"seq{next_seq_num}"
    final_output_path = os.path.join(class_path, seq_dir_name)
    os.makedirs(final_output_path)
    
    print(f"
Processing video: {video_path}")
    print(f"Detected class: {class_label}")
    print(f"Saving to sequence directory: {final_output_path}")

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
        for _ in range(interval):
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
        
        output_filename = f"flow_{frame_idx:04d}.npy"
        output_path_full = os.path.join(final_output_path, output_filename)
        np.save(output_path_full, flo_numpy)

        frame1_bgr = frame2_bgr
        frame_idx += 1

    cap.release()
    print(f"Finished processing {video_path}. Total frames processed: {frame_idx}")

def run_optical_flow_extraction(model_path, input_path, output_path, interval, use_small_model):
    print(f"--- Step 2: Extracting optical flow from prepared videos ---")
    parser = argparse.ArgumentParser() # Dummy parser to initialize RAFT model
    parser.add_argument('--model', default=model_path)
    parser.add_argument('--small', action='store_true')
    parser.add_argument('--mixed_precision', action='store_true')
    parser.add_argument('--alternate_corr', action='store_true')
    
    # Manually set args for RAFT model initialization
    raft_args = parser.parse_args([]) # Pass empty list to avoid sys.argv parsing
    raft_args.small = use_small_model
    raft_args.mixed_precision = False # Assuming not using mixed precision by default
    raft_args.alternate_corr = False # Assuming not using alternate corr by default

    model = torch.nn.DataParallel(RAFT(raft_args))
    model.load_state_dict(torch.load(model_path))

    model = model.module
    model.to(DEVICE)
    model.eval()

    if os.path.isdir(input_path):
        print(f"Input path is a directory. Searching for videos...")
        video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv']
        video_files = []
        for ext in video_extensions:
            video_files.extend(glob.glob(os.path.join(input_path, ext)))
        
        if not video_files:
            print(f"No video files found in {input_path}")
            return

        print(f"Found {len(video_files)} videos to process.")
        for video_path in sorted(video_files):
            process_single_video_for_flow(video_path, model, output_path, interval)

    elif os.path.isfile(input_path):
        print(f"Input path is a single file.")
        process_single_video_for_flow(input_path, model, output_path, interval)
        
    else:
        print(f"Error: Input path {input_path} is not a valid file or directory.")
        return

    print("Step 2 complete.
")


# --- Step 3: Sequence Trimming Functions ---
def calculate_motion_score(flow_file):
    """Calculates the motion score for a single optical flow file."""
    flow_data = np.load(flow_file)
    magnitude = np.sqrt(np.sum(flow_data**2, axis=-1))
    return np.mean(magnitude)

def find_best_window(flow_files, window_size):
    """Finds the best window of frames with the highest total motion score."""
    num_files = len(flow_files)
    if num_files < window_size:
        return 0, num_files

    motion_scores = [calculate_motion_score(f) for f in flow_files]

    max_score = -1
    best_start_index = 0

    current_window_score = sum(motion_scores[:window_size])
    max_score = current_window_score

    for i in range(1, num_files - window_size + 1):
        current_window_score = current_window_score - motion_scores[i-1] + motion_scores[i + window_size - 1]
        if current_window_score > max_score:
            max_score = current_window_score
            best_start_index = i
            
    return best_start_index, best_start_index + window_size

def run_sequence_trimming(input_dir, output_dir, sequence_length):
    print(f"--- Step 3: Trimming sequences from optical flow data ---")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Target sequence length: {sequence_length}")

    if not os.path.exists(input_dir):
        print(f"Error: Input directory not found at {input_dir}")
        return

    seq_dirs = glob.glob(os.path.join(input_dir, 'class*', 'seq*'))
    if not seq_dirs:
        print(f"No sequences found in the format 'class*/seq*' inside {input_dir}")
        return

    for seq_dir in tqdm(seq_dirs, desc="Processing sequences"):
        flow_files = sorted(glob.glob(os.path.join(seq_dir, 'flow_*.npy')))
        
        if not flow_files:
            continue

        start_idx, end_idx = find_best_window(flow_files, sequence_length)
        best_window_files = flow_files[start_idx:end_idx]

        relative_path = os.path.relpath(seq_dir, input_dir)
        output_seq_dir = os.path.join(output_dir, relative_path)
        os.makedirs(output_seq_dir, exist_ok=True)

        for i, file_path in enumerate(best_window_files):
            new_filename = f"flow_{i:04d}.npy"
            shutil.copy(file_path, os.path.join(output_seq_dir, new_filename))

    print("Step 3 complete.
")


# --- Step 4: Classifier Training Functions ---
class OpticalFlowDataset(Dataset):
    def __init__(self, data_dir, sequence_length=10):
        self.sequence_length = sequence_length
        self.samples = []
        
        class_dirs = glob.glob(os.path.join(data_dir, 'class*'))
        for class_dir in class_dirs:
            class_label = int(class_dir.split('class')[-1])
            seq_dirs = glob.glob(os.path.join(class_dir, 'seq*'))
            for seq_dir in seq_dirs:
                self.samples.append((seq_dir, class_label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        seq_dir, class_label = self.samples[idx]
        
        flow_files = sorted(glob.glob(os.path.join(seq_dir, 'flow_*.npy')))
        
        flow_arrays = [np.load(f) for f in flow_files]
        
        flow_sequence = torch.from_numpy(np.array(flow_arrays)).permute(0, 3, 1, 2)

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

def run_classifier_training(data_dir, num_classes, sequence_length, epochs, batch_size, learning_rate, model_save_path, wandb_project_name):
    print(f"--- Step 4: Training the classifier ---")
    # Initialize wandb
    wandb.init(project=wandb_project_name, config={
        "data_dir": data_dir,
        "num_classes": num_classes,
        "sequence_length": sequence_length,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "model_save_path": model_save_path
    })

    input_dim = 2
    hidden_dims = [64, 128]
    kernel_size = 3
    n_layers = 2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = OpticalFlowDataset(data_dir=data_dir, sequence_length=sequence_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = ConvGRUClassifier(input_dim, hidden_dims, kernel_size, n_layers, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    wandb.watch(model, log="all")

    print("Starting training...")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(dataloader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            wandb.log({"batch_loss": loss.item()})

        epoch_loss = running_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")
        wandb.log({"epoch_loss": epoch_loss, "epoch": epoch + 1})

    print("Training complete.")

    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")
    
    wandb.finish()
    print("Step 4 complete.
")


# --- Main Pipeline Execution ---
def main():
    # Ensure output directories exist
    os.makedirs(PREPARED_VIDEO_DIR, exist_ok=True)
    os.makedirs(OUTPUT_OPTICAL_FLOW_DIR, exist_ok=True)
    os.makedirs(PROCESSED_SEQUENCES_DIR, exist_ok=True)

    # Step 1: Prepare video files
    prepare_videos(INPUT_VIDEO_DIR, PREPARED_VIDEO_DIR)

    # Step 2: Extract optical flow
    run_optical_flow_extraction(
        RAFT_MODEL_PATH,
        PREPARED_VIDEO_DIR,
        OUTPUT_OPTICAL_FLOW_DIR,
        OPTICAL_FLOW_INTERVAL,
        USE_SMALL_RAFT_MODEL
    )

    # Step 3: Trim sequences
    run_sequence_trimming(
        OUTPUT_OPTICAL_FLOW_DIR,
        PROCESSED_SEQUENCES_DIR,
        SEQUENCE_LENGTH
    )

    # Step 4: Train classifier
    run_classifier_training(
        PROCESSED_SEQUENCES_DIR,
        NUM_CLASSES,
        SEQUENCE_LENGTH,
        TRAINING_EPOCHS,
        TRAINING_BATCH_SIZE,
        LEARNING_RATE,
        CLASSIFIER_MODEL_SAVE_PATH,
        WANDB_PROJECT_NAME
    )

    print("Full unified pipeline execution finished successfully!")

if __name__ == '__main__':
    main()
