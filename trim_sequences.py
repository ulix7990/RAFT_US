import numpy as np
import os
import glob
import shutil
import argparse
from tqdm import tqdm

def calculate_motion_score(flow_file, roi_width=None, roi_height=None):
    """
    Calculates the motion score for a single optical flow file,
    optionally using a specified central-bottom ROI.
    """
    flow_data = np.load(flow_file) # flow_data.shape: (height, width, 2)

    if roi_width is not None and roi_height is not None:
        h, w, _ = flow_data.shape
        
        # Calculate start and end coordinates for the central-bottom ROI
        start_h = max(0, h - roi_height) # Image bottom aligned
        end_h = h                     
        start_w = max(0, (w - roi_width) // 2)
        end_w = min(w, start_w + roi_width)
        
        # Ensure ROI dimensions are not larger than frame dimensions
        # And ensure the calculated ROI is valid
        if (end_h - start_h <= 0) or (end_w - start_w <= 0) or \
           ((end_h - start_h) > h) or ((end_w - start_w) > w):
            print(f"Warning: Invalid or too large ROI ({roi_width}x{roi_height}) for flow data ({w}x{h}). Using full frame for score calculation.")
            roi_flow_data = flow_data
        else:
            roi_flow_data = flow_data[start_h:end_h, start_w:end_w, :]
    else:
        roi_flow_data = flow_data # Use full frame if ROI dimensions are not provided

    # Calculate magnitude of flow vectors for each pixel within the ROI
    magnitude = np.sqrt(np.sum(roi_flow_data**2, axis=-1))
    
    # Return the mean magnitude as the score for this frame
    return np.mean(magnitude)

def find_best_window(flow_files, window_size, roi_width, roi_height):
    """Finds the best window of frames with the highest total motion score."""
    num_files = len(flow_files)
    if num_files < window_size:
        # If the sequence is shorter than the desired length, return the whole sequence
        return 0, num_files

    # Calculate motion scores for all frames first, using the specified ROI
    motion_scores = [calculate_motion_score(f, roi_width, roi_height) for f in flow_files]

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
    """Processes all sequences from the input directory and saves the trimmed versions,
       optionally cropping to the ROI before saving."""
    print(f"Starting sequence trimming...")
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Target sequence length: {args.sequence_length}")
    
    use_roi_for_saving = args.roi_width is not None and args.roi_height is not None
    if use_roi_for_saving:
        print(f"Processing with central-bottom ROI: {args.roi_width}x{args.roi_height} (applied to score calculation AND saving).")
    else:
        print("Processing with full frame (no ROI specified).")

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

        # Find the best window using ROI for score calculation
        start_idx, end_idx = find_best_window(flow_files, args.sequence_length, args.roi_width, args.roi_height)
        best_window_files = flow_files[start_idx:end_idx]

        # Create corresponding output directory structure
        relative_path = os.path.relpath(seq_dir, args.input_dir)
        output_seq_dir = os.path.join(args.output_dir, relative_path)
        os.makedirs(output_seq_dir, exist_ok=True)

        # Copy or save the best window of files, applying ROI if specified
        for i, file_path in enumerate(best_window_files):
            new_filename = f"flow_{i:04d}.npy"
            output_file_path = os.path.join(output_seq_dir, new_filename)

            if use_roi_for_saving:
                # Load the full flow data
                full_flow_data = np.load(file_path)
                h, w, _ = full_flow_data.shape

                # Calculate ROI coordinates again (same as in calculate_motion_score)
                start_h = max(0, h - args.roi_height)
                end_h = h                     
                start_w = max(0, (w - args.roi_width) // 2)
                end_w = min(w, start_w + args.roi_width)

                # Perform the crop
                if (end_h - start_h <= 0) or (end_w - start_w <= 0) or \
                   ((end_h - start_h) > h) or ((end_w - start_w) > w):
                    # Fallback to full frame if ROI is invalid or too large for saving
                    print(f"Warning: Invalid ROI ({args.roi_width}x{args.roi_height}) for saving flow data ({w}x{h}). Saving full frame for {file_path}.")
                    cropped_flow_data = full_flow_data
                else:
                    cropped_flow_data = full_flow_data[start_h:end_h, start_w:end_w, :]
                
                # Save the cropped flow data
                np.save(output_file_path, cropped_flow_data)
            else:
                # If no ROI specified, just copy the original file
                shutil.copy(file_path, output_file_path)

    print("\nTrimming and saving complete.")
    print(f"Processed sequences are saved in {args.output_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Trim optical flow sequences to the most significant part based on motion, with optional ROI processing and saving.")
    parser.add_argument('--input_dir', type=str, default='./saved_optical_flow', help='Directory containing the raw sequences (class*/seq* structure).')
    parser.add_argument('--output_dir', type=str, default='./processed_sequences', help='Directory to save the trimmed sequences.')
    parser.add_argument('--sequence_length', type=int, default=10, help='The desired length of the output sequences.')
    parser.add_argument('--roi_width', type=int, default=None, help='Width of the central-bottom ROI to consider for motion score calculation AND saving. If not specified, the full frame is used.')
    parser.add_argument('--roi_height', type=int, default=None, help='Height of the central-bottom ROI to consider for motion score calculation AND saving. If not specified, the full frame is used.')
    
    args = parser.parse_args()
    process_sequences(args)