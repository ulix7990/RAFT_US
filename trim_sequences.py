import numpy as np
import os
import glob
import argparse
from tqdm import tqdm

def calculate_motion_score(flow_data_frame, roi_width=None, roi_height=None):
    """
    Calculates the motion score for a single optical flow frame (a numpy array),
    optionally using a specified central-bottom ROI.
    """
    # <<<--- 입력이 파일 경로가 아닌 NumPy 배열로 변경됨 --->>>
    # flow_data_frame.shape: (height, width, 2)

    if roi_width is not None and roi_height is not None:
        h, w, _ = flow_data_frame.shape
        
        start_h = max(0, h - roi_height)
        end_h = h                     
        start_w = max(0, (w - roi_width) // 2)
        end_w = min(w, start_w + roi_width)
        
        if (end_h - start_h <= 0) or (end_w - start_w <= 0) or \
           ((end_h - start_h) > h) or ((end_w - start_w) > w):
            roi_flow_data = flow_data_frame
        else:
            roi_flow_data = flow_data_frame[start_h:end_h, start_w:end_w, :]
    else:
        roi_flow_data = flow_data_frame

    magnitude = np.sqrt(np.sum(roi_flow_data**2, axis=-1))
    return np.mean(magnitude)

def find_best_window(flow_sequence, window_size, roi_width, roi_height):
    """Finds the best window of frames with the highest total motion score."""
    # <<<--- 입력이 파일 리스트가 아닌 NumPy 배열(시퀀스)로 변경됨 --->>>
    num_frames = len(flow_sequence)
    if num_frames < window_size:
        return 0, num_frames

    # <<<--- 모든 프레임에 대해 모션 점수 계산 --->>>
    motion_scores = [calculate_motion_score(frame, roi_width, roi_height) for frame in flow_sequence]

    max_score = -1
    best_start_index = 0

    current_window_score = sum(motion_scores[:window_size])
    max_score = current_window_score

    for i in range(1, num_frames - window_size + 1):
        current_window_score = current_window_score - motion_scores[i-1] + motion_scores[i + window_size - 1]
        if current_window_score > max_score:
            max_score = current_window_score
            best_start_index = i
            
    return best_start_index, best_start_index + window_size

def process_sequences(args):
    """Processes all .npz sequence files from the input directory and saves the trimmed versions."""
    print(f"Starting sequence trimming from .npz files...")
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Target sequence length: {args.sequence_length}")
    
    use_roi_for_saving = args.roi_width is not None and args.roi_height is not None
    if use_roi_for_saving:
        print(f"Processing with central-bottom ROI: {args.roi_width}x{args.roi_height}")
    else:
        print("Processing with full frame.")

    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory not found at {args.input_dir}")
        return

    # <<<--- `class*/*.npz` 패턴으로 모든 .npz 파일 검색 --->>>
    npz_files = glob.glob(os.path.join(args.input_dir, 'class*/*.npz'))
    if not npz_files:
        print(f"No .npz files found in the format 'class*/*.npz' inside {args.input_dir}")
        return

    for npz_file_path in tqdm(npz_files, desc="Processing .npz files"):
        # <<<--- .npz 파일 로드 --->>>
        try:
            with np.load(npz_file_path) as data:
                # 저장 시 사용한 키('flow_sequence')로 데이터 로드
                full_flow_sequence = data['flow_sequence']
        except (KeyError, IOError) as e:
            print(f"Warning: Could not load or find 'flow_sequence' in {npz_file_path}. Skipping. Error: {e}")
            continue

        if len(full_flow_sequence) == 0:
            continue

        # <<<--- 로드된 시퀀스 데이터로 최적의 윈도우 찾기 --->>>
        start_idx, end_idx = find_best_window(full_flow_sequence, args.sequence_length, args.roi_width, args.roi_height)
        best_window_sequence = full_flow_sequence[start_idx:end_idx]

        # <<<--- 출력 디렉토리 구조 생성 (기존 seq 디렉토리 대신 npz 파일명 기반으로 생성) --->>>
        relative_class_dir = os.path.basename(os.path.dirname(npz_file_path))
        sequence_name = os.path.splitext(os.path.basename(npz_file_path))[0]
        output_seq_dir = os.path.join(args.output_dir, relative_class_dir, f"seq_{sequence_name}")
        os.makedirs(output_seq_dir, exist_ok=True)

        # <<<--- 잘라낸 시퀀스를 개별 .npy 파일로 저장 (다음 단계를 위해) --->>>
        for i, frame_data in enumerate(best_window_sequence):
            new_filename = f"flow_{i:04d}.npy"
            output_file_path = os.path.join(output_seq_dir, new_filename)

            if use_roi_for_saving:
                h, w, _ = frame_data.shape
                start_h = max(0, h - args.roi_height)
                end_h = h                     
                start_w = max(0, (w - args.roi_width) // 2)
                end_w = min(w, start_w + args.roi_width)

                if (end_h - start_h <= 0) or (end_w - start_w <= 0) or \
                   ((end_h - start_h) > h) or ((end_w - start_w) > w):
                    cropped_flow_data = frame_data
                else:
                    cropped_flow_data = frame_data[start_h:end_h, start_w:end_w, :]
                
                np.save(output_file_path, cropped_flow_data)
            else:
                np.save(output_file_path, frame_data)

    print("\nTrimming and saving complete.")
    print(f"Processed sequences are saved in {args.output_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Trim optical flow sequences from .npz files.")
    # <<<--- input_dir 설명 변경 --->>>
    parser.add_argument('--input_dir', type=str, default='./saved_optical_flow', help='Directory containing the compressed .npz files (class*/*.npz structure).')
    parser.add_argument('--output_dir', type=str, default='./processed_sequences', help='Directory to save the final trimmed sequences as individual .npy files.')
    parser.add_argument('--sequence_length', type=int, default=10, help='The desired length of the output sequences.')
    parser.add_argument('--roi_width', type=int, default=None, help='Width of the ROI for motion calculation and final saving.')
    parser.add_argument('--roi_height', type=int, default=None, help='Height of the ROI for motion calculation and final saving.')
    
    args = parser.parse_args()
    process_sequences(args)
