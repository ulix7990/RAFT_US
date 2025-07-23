#!/bin/bash

# --- Configuration --- #

# Base directory of the RAFT_US project
RAFT_PROJECT_DIR="/home/ulix7990/rmc_slump_ws/RAFT_US"

# Directory containing your original video files (e.g., where your '03', '04', '07' folders are)
INPUT_VIDEO_DIR="/media/ulix7990/B214294A142912C1/250708_레미콘영상/분류"

# Directory where the prepared videos (renamed) will be stored
PREPARED_VIDEO_DIR="${RAFT_PROJECT_DIR}/prepared_raft_videos"

# Directory where the extracted optical flow data will be saved
OUTPUT_OPTICAL_FLOW_DIR="${RAFT_PROJECT_DIR}/saved_optical_flow_data"

# Path to your trained RAFT model checkpoint file
# !!! IMPORTANT: REPLACE THIS WITH YOUR ACTUAL MODEL PATH !!!
RAFT_MODEL_PATH="${RAFT_PROJECT_DIR}/models/raft-small.pth" # Example path, change this!

# --- Script Execution --- #

echo "Starting video preparation..."

# Step 1: Prepare video files by renaming and copying them
python "${RAFT_PROJECT_DIR}/prepare_videos_for_raft.py" \
    --input_dir "${INPUT_VIDEO_DIR}" \
    --output_dir "${PREPARED_VIDEO_DIR}"

# Check if the preparation step was successful
if [ $? -ne 0 ]; then
    echo "Error: Video preparation failed. Exiting."
    exit 1
fi

echo "Video preparation complete. Starting optical flow extraction..."

# Step 2: Run RAFT to extract optical flow from the prepared videos
python "${RAFT_PROJECT_DIR}/run_video_of_save.py" \
    --model "${RAFT_MODEL_PATH}" \
    --input_path "${PREPARED_VIDEO_DIR}" \
    --output_path "${OUTPUT_OPTICAL_FLOW_DIR}" \
    --small # Add --small if you are using the small model, remove otherwise
    # Add other arguments for run_video_of_save.py as needed, e.g., --interval 2

# Check if the optical flow extraction step was successful
if [ $? -ne 0 ]; then
    echo "Error: Optical flow extraction failed. Exiting."
    exit 1
fi

echo "Optical flow extraction complete. Data saved to: ${OUTPUT_OPTICAL_FLOW_DIR}"
