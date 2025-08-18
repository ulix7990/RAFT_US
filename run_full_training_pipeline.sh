#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Configuration --- #

# Base directory of the RAFT_US project
RAFT_PROJECT_DIR="."
RAFT_DATA_DIR="."

# Directory containing your original video files
INPUT_VIDEO_DIR="/media/cvtech/백업자료/데이터셋/rmc_video"

# Directory where the prepared videos (renamed) will be stored
PREPARED_VIDEO_DIR="${RAFT_DATA_DIR}/data/prepared_raft_videos"

# Directory for processed sequences (output of optical flow extraction)
PROCESSED_SEQUENCES_DIR="${RAFT_DATA_DIR}/data/processed_sequences"

# Path to your trained RAFT model checkpoint file
RAFT_MODEL_PATH="${RAFT_PROJECT_DIR}/models/raft-sintel.pth"

# Path to save the trained classifier model
CLASSIFIER_MODEL_SAVE_PATH="${RAFT_PROJECT_DIR}/convgru_classifier.pth"

# --- Processing & Training Settings --- #
# Height and width for ROI extraction and training input
CROP_H=128
CROP_W=128

# Classifier training settings
PREPROCESSING_MODE="crop"  # Preprocessing mode: 'resize' or 'crop'
CROP_LOCATION="bottom-center"     # Crop location: 'random', 'center', 'top-left', etc.
PATIENCE=10                # Patience for early stopping
DROPOUT_RATE=0.5           # Dropout rate for the classifier
EPOCHS=100
BATCH_SIZE=4
LEARNING_RATE=0.001
WEIGHT_DECAY=1e-5
NUM_CLASSES=3
SEQUENCE_LENGTH=10
N_LAYERS=2
HIDDEN_DIMS="32 64"

# --- Script Execution Logic --- #

# Default values
START_STEP=1
END_STEP=3

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

# --- Step 2: Extracting Optical Flow and Trimming Sequences ---
if (( START_STEP <= 2 && END_STEP >= 2 )); then
    echo "--- Step 2: Extracting Optical Flow and Trimming Sequences ---"
    python "${RAFT_PROJECT_DIR}/run_of_and_trim.py" \
        --model "${RAFT_MODEL_PATH}" \
        --input_path "${PREPARED_VIDEO_DIR}" \
        --output_path "${PROCESSED_SEQUENCES_DIR}" \
        --interval 5 \
        --sequence_length ${SEQUENCE_LENGTH} \
        --roi_width ${CROP_W} \
        --roi_height ${CROP_H}
    echo "Step 2 complete."
fi

# --- Step 3: Training the classifier ---
if (( START_STEP <= 3 && END_STEP >= 3 )); then
    echo "--- Step 3: Training the classifier ---"
    python "${RAFT_PROJECT_DIR}/train_classifier.py" \
        --data_dir "${PROCESSED_SEQUENCES_DIR}" \
        --resize_h ${CROP_H} \
        --resize_w ${CROP_W} \
        --preprocessing_mode "${PREPROCESSING_MODE}" \
        --crop_location "${CROP_LOCATION}" \
        --num_classes ${NUM_CLASSES} \
        --sequence_length ${SEQUENCE_LENGTH} \
        --epochs ${EPOCHS} \
        --batch_size ${BATCH_SIZE} \
        --model_save_path "${CLASSIFIER_MODEL_SAVE_PATH}" \
        --learning_rate ${LEARNING_RATE} \
        --weight_decay ${WEIGHT_DECAY} \
        --patience ${PATIENCE} \
        --dropout_rate ${DROPOUT_RATE} \
        --n_layers ${N_LAYERS} \
        --hidden_dims ${HIDDEN_DIMS}
    echo "Step 3 complete."
fi

echo "Full pipeline execution finished successfully for selected steps!"
