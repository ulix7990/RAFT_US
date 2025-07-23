#!/bin/bash

# This script runs the sequence trimming process with default arguments.
# You can override the arguments directly here or on the command line.

python trim_sequences.py \
    --input_dir data/optical_flow_dir \
    --output_dir data/processed_sequences \
    --sequence_length 10

