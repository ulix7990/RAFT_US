import os
import shutil
import argparse

def prepare_videos(input_dir, output_dir):
    """
    Prepares video files for the run_video_of_save.py script by renaming them
    to include the class label in the filename and copying them to a flat output directory.

    Args:
        input_dir (str): The root directory containing the original video files
                         structured as /date/class_label/video.avi.
        output_dir (str): The directory where the prepared video files will be saved.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    print(f"Scanning for video files in: {input_dir}")
    processed_count = 0

    for root, _, files in os.walk(input_dir):
        # Extract class label from the immediate parent directory name
        # The structure is .../date_dir/class_label_dir/video.avi
        # So, the class label is the last part of the root path.
        try:
            class_label = os.path.basename(root)
            # Basic check to ensure class_label is numeric
            if not class_label.isdigit():
                # If it's not a digit, it's likely a date directory or the root itself, skip.
                continue
        except Exception:
            continue # Skip if parsing fails

        for filename in files:
            if filename.lower().endswith(('.avi', '.mp4', '.mov', '.mkv')):
                original_filepath = os.path.join(root, filename)
                base_name, ext = os.path.splitext(filename)

                # Check if the filename already ends with the class label
                # e.g., "my_video_120.avi" where class_label is "120"
                if base_name.endswith(f"_{class_label}"):
                    new_filename = filename
                else:
                    new_filename = f"{base_name}_{class_label}{ext}"

                destination_filepath = os.path.join(output_dir, new_filename)

                # Avoid copying if the file already exists and is identical
                if os.path.exists(destination_filepath) and os.path.getsize(original_filepath) == os.path.getsize(destination_filepath):
                    print(f"Skipping existing identical file: {new_filename}")
                    continue

                try:
                    shutil.copy2(original_filepath, destination_filepath)
                    print(f"Copied and renamed: {filename} -> {new_filename}")
                    processed_count += 1
                except Exception as e:
                    print(f"Error copying {original_filepath} to {destination_filepath}: {e}")

    print(f"Finished preparing videos. Total processed: {processed_count}")
    print(f"Prepared videos are in: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare video files for RAFT optical flow processing.")
    parser.add_argument("--input_dir", required=True,
                        help="The root directory containing the original video files (e.g., /path/to/분류).")
    parser.add_argument("--output_dir", default="prepared_videos",
                        help="The directory where the prepared video files will be saved.")
    args = parser.parse_args()

    prepare_videos(args.input_dir, args.output_dir)
