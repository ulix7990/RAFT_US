import numpy as np
import os
import glob
import shutil
import argparse
from tqdm import tqdm

def calculate_motion_score(flow_file):
    """Calculates the motion score for a single optical flow file."""
    flow_data = np.load(flow_file)
    # Calculate magnitude of flow vectors for each pixel
    magnitude = np.sqrt(np.sum(flow_data**2, axis=-1))
    # Return the mean magnitude as the score for this frame
    return np.mean(magnitude)

def find_best_window(flow_files, window_size):
    """Finds the best window of frames with the highest total motion score."""
    num_files = len(flow_files)
    if num_files < window_size:
        # If the sequence is shorter than the desired length, return the whole sequence
        return 0, num_files

    # Calculate motion scores for all frames first
    motion_scores = [calculate_motion_score(f) for f in flow_files]

    max_score = -1
    best_start_index = 0

    # Initial window score
    current_window_score = sum(motion_scores[:window_size])
    max_score = current_window_score

    # Slide the window across the sequence
    for i in range(1, num_files - window_size + 1):
        # Efficiently update the window score
        current_window_score = current_window_score - motion_scores[i-1] + motion_scores[i + window_size - 1]
        if current_window_score > max_score:
            max_score = current_window_score
            best_start_index = i
            
    return best_start_index, best_start_index + window_size

def process_sequences(args):
    """Processes all sequences from the input directory and saves the trimmed versions."""
    print(f"Starting sequence trimming...")
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Target sequence length: {args.sequence_length}")

    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory not found at {args.input_dir}")
        return

    # Find all sequence directories
    seq_dirs = glob.glob(os.path.join(args.input_dir, 'class*/seq*'))
    if not seq_dirs:
        print(f"No sequences found in the format 'class*/seq*' inside {args.input_dir}")
        return

    for seq_dir in tqdm(seq_dirs, desc="Processing sequences"):
        flow_files = sorted(glob.glob(os.path.join(seq_dir, 'flow_*.npy')))
        
        if not flow_files:
            continue

        start_idx, end_idx = find_best_window(flow_files, args.sequence_length)
        best_window_files = flow_files[start_idx:end_idx]

        # Create corresponding output directory structure
        # Example: input/class1/seq1 -> output/class1/seq1
        relative_path = os.path.relpath(seq_dir, args.input_dir)
        output_seq_dir = os.path.join(args.output_dir, relative_path)
        os.makedirs(output_seq_dir, exist_ok=True)

        # Copy the best window of files and rename them sequentially
        for i, file_path in enumerate(best_window_files):
            new_filename = f"flow_{i:04d}.npy"
            shutil.copy(file_path, os.path.join(output_seq_dir, new_filename))

    print("\nTrimming complete.")
    print(f"Processed sequences are saved in {args.output_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Trim optical flow sequences to the most significant part based on motion.")
    parser.add_argument('--input_dir', type=str, default='./saved_optical_flow', help='Directory containing the raw sequences (class*/seq* structure).')
    parser.add_argument('--output_dir', type=str, default='./processed_sequences', help='Directory to save the trimmed sequences.')
    parser.add_argument('--sequence_length', type=int, default=10, help='The desired length of the output sequences.')
    
    args = parser.parse_args()
    process_sequences(args)
