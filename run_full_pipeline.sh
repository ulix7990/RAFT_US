#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Configuration --- #

# Base directory of the RAFT_US project
RAFT_PROJECT_DIR="."

# Directory containing your original video files (e.g., where your '03', '04', '07' folders are)
INPUT_VIDEO_DIR="${RAFT_PROJECT_DIR}/rmc_video"

# Directory where the prepared videos (renamed) will be stored
PREPARED_VIDEO_DIR="${RAFT_PROJECT_DIR}/data/prepared_raft_videos"

# Directory where the extracted optical flow data will be saved
OUTPUT_OPTICAL_FLOW_DIR="${RAFT_PROJECT_DIR}/data/saved_optical_flow"

# Path to your trained RAFT model checkpoint file
# !!! IMPORTANT: REPLACE THIS WITH YOUR ACTUAL MODEL PATH !!!
RAFT_MODEL_PATH="${RAFT_PROJECT_DIR}/models/raft-sintel.pth" # Example path, change this!

# Directory for processed sequences (output of trim_sequences.py)
PROCESSED_SEQUENCES_DIR="${RAFT_PROJECT_DIR}/data/processed_sequences"

# Path to save the trained classifier model
CLASSIFIER_MODEL_SAVE_PATH="${RAFT_PROJECT_DIR}/convgru_classifier.pth"

# --- Script Execution Logic --- #

# Default values
START_STEP=1
END_STEP=4

# Parse arguments for start and end steps
while getopts s:e: flag
do
    case "${flag}" in
        s) START_STEP=${OPTARG};;
        e) END_STEP=${OPTARG};;
    esac
done

echo "--- Starting pipeline from Step ${START_STEP} to Step ${END_STEP} ---"

# --- Step 1: Preparing video files for RAFT ---
if (( START_STEP <= 1 && END_STEP >= 1 )); then
    echo "--- Step 1: Preparing video files for RAFT ---"
    python "${RAFT_PROJECT_DIR}/prepare_videos_for_raft.py" \
        --input_dir "${INPUT_VIDEO_DIR}" \
        --output_dir "${PREPARED_VIDEO_DIR}"
    echo "Step 1 complete."
fi

# --- Step 2: Extracting optical flow from prepared videos ---
if (( START_STEP <= 2 && END_STEP >= 2 )); then
    echo "--- Step 2: Extracting optical flow from prepared videos ---"
    python "${RAFT_PROJECT_DIR}/run_video_of_save.py" \
        --model "${RAFT_MODEL_PATH}" \
        --input_path "${PREPARED_VIDEO_DIR}" \
        --output_path "${OUTPUT_OPTICAL_FLOW_DIR}" \
        --interval 5
        # Add other arguments for run_video_of_save.py as needed, e.g., --interval 2
    echo "Step 2 complete."
fi

# --- Step 3: Trimming sequences from optical flow data ---
if (( START_STEP <= 3 && END_STEP >= 3 )); then
    echo "--- Step 3: Trimming sequences from optical flow data ---"
    python "${RAFT_PROJECT_DIR}/trim_sequences.py" \
        --input_dir "${OUTPUT_OPTICAL_FLOW_DIR}" \
        --output_dir "${PROCESSED_SEQUENCES_DIR}" \
        --sequence_length 10 \
        --roi_width 128 \
        --roi_height 128 # 여기에 원하는 하단 중앙 ROI의 너비와 높이를 지정
    echo "Step 3 complete."
fi

# --- Step 4: Training the classifier ---
if (( START_STEP <= 4 && END_STEP >= 4 )); then
    echo "--- Step 4: Training the classifier ---"
    python "${RAFT_PROJECT_DIR}/train_classifier.py" \
        --data_dir "${PROCESSED_SEQUENCES_DIR}" \
        --num_classes 3 \
        --sequence_length 10 \
        --epochs 100 \
        --batch_size 1 \
        --model_save_path "${CLASSIFIER_MODEL_SAVE_PATH}" \
        --learning_rate 0.001 \
        --weight_decay 1e-5 # weight_decay 추가
    echo "Step 4 complete."
fi

echo "Full pipeline execution finished successfully for selected steps!"