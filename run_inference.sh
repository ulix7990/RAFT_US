#!/bin/bash

# --- Configuration ---

# !!! IMPORTANT: Set the path to the video you want to classify !!!
INPUT_VIDEO="/path/to/your/video.mp4"

# --- Model and Training Parameters (should match your trained model) ---
RAFT_MODEL_PATH="./models/raft-sintel.pth"
CLASSIFIER_MODEL_PATH="./convgru_classifier.pth"

NUM_CLASSES=3
SEQUENCE_LENGTH=10
INTERVAL=5
RESIZE_H=256
RESIZE_W=448

# --- Script Execution ---

# Check if the input video file exists
if [ ! -f "$INPUT_VIDEO" ]; then
    echo "Error: Input video file not found at $INPUT_VIDEO"
    echo "Please update the INPUT_VIDEO variable in this script."
    exit 1
fi

# Check if the model files exist
if [ ! -f "$RAFT_MODEL_PATH" ] || [ ! -f "$CLASSIFIER_MODEL_PATH" ]; then
    echo "Error: One or more model files are missing."
    echo "- RAFT Model expected at: $RAFT_MODEL_PATH"
    echo "- Classifier Model expected at: $CLASSIFIER_MODEL_PATH"
    exit 1
fi


echo "--- Starting Inference Pipeline ---"

python run_inference.py \
    --video_path "$INPUT_VIDEO" \
    --raft_model "$RAFT_MODEL_PATH" \
    --classifier_model "$CLASSIFIER_MODEL_PATH" \
    --num_classes $NUM_CLASSES \
    --sequence_length $SEQUENCE_LENGTH \
    --interval $INTERVAL \
    --resize_h $RESIZE_H \
    --resize_w $RESIZE_W

echo "--- Inference Finished ---"
