# detect_grid.py

import cv2
import numpy as np
import os

def iou(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou

def detect_red_grid(video_path):
    """
    Reads a video, detects red laser grid, and masks it.
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    frame_count = 0
    all_streaks = [] # Stores all completed and active streaks
    active_streaks = {} # Stores currently active streaks, keyed by their rect_id
    next_rect_id = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video or error reading frame.")
            break
        frame_count += 1

        current_frame_detections = []

        # Apply median blur to reduce noise
        frame = cv2.medianBlur(frame, 5)

        # Convert frame to HSV color space
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Define range for red color (lower red)
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)

        # Define range for red color (upper red)
        lower_red2 = np.array([170, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)

        # Combine the two masks
        mask = cv2.bitwise_or(mask1, mask2)

        # Optional: Morphological operations to clean up the mask
        kernel = np.ones((7, 7), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=10)
        mask = cv2.erode(mask, kernel, iterations=10)

        # Create a black image
        grid_lines_image = np.zeros_like(frame)
        grid_lines_image[mask > 0] = [255, 255, 255]

        grid_lines_image = cv2.medianBlur(grid_lines_image, 5)
        grid_lines_image = cv2.bitwise_not(grid_lines_image)

        kernel_post_invert = np.ones((15, 15), np.uint8)
        grid_lines_image = cv2.erode(grid_lines_image, kernel_post_invert, iterations=10)
        grid_lines_image = cv2.dilate(grid_lines_image, kernel_post_invert, iterations=8)

        gray_grid_lines = cv2.cvtColor(grid_lines_image, cv2.COLOR_BGR2GRAY)
        contours, _ = cv2.findContours(gray_grid_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            epsilon = 0.05 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            if len(approx) == 4:
                area = cv2.contourArea(approx)
                x, y, w, h = cv2.boundingRect(approx)
                aspect_ratio = float(w)/h

                if area > 25000 and area < 45000 and aspect_ratio >= 1.0 and aspect_ratio <= 1.5:
                    current_frame_detections.append([x, y, w, h])

        matched_active_streaks_ids = set()
        new_detections_this_frame = []

        for current_bbox in current_frame_detections:
            found_match_for_current_bbox = False
            for rect_id, streak_info in active_streaks.items():
                if iou(current_bbox, streak_info['bboxes'][-1]) > 0.5:
                    streak_info['end_frame'] = frame_count
                    streak_info['bboxes'].append(current_bbox)
                    matched_active_streaks_ids.add(rect_id)
                    found_match_for_current_bbox = True
                    break
            if not found_match_for_current_bbox:
                new_detections_this_frame.append(current_bbox)

        streaks_to_deactivate = []
        for rect_id, streak_info in active_streaks.items():
            if rect_id not in matched_active_streaks_ids:
                streak_info['active'] = False
                all_streaks.append(streak_info)
                streaks_to_deactivate.append(rect_id)
        for rect_id in streaks_to_deactivate:
            del active_streaks[rect_id]

        for new_bbox in new_detections_this_frame:
            new_streak = {'id': next_rect_id, 'start_frame': frame_count, 'end_frame': frame_count, 'bboxes': [new_bbox], 'active': True}
            active_streaks[next_rect_id] = new_streak
            next_rect_id += 1

    for rect_id, streak_info in active_streaks.items():
        streak_info['active'] = False
        all_streaks.append(streak_info)

    longest_persisting_streak = None
    max_duration = 0

    for streak_info in all_streaks:
        duration = streak_info['end_frame'] - streak_info['start_frame'] + 1
        if duration > max_duration:
            max_duration = duration
            longest_persisting_streak = streak_info

    if longest_persisting_streak:
        print(f"Longest continuously persisting rectangle (ID: {longest_persisting_streak['id']}) found for {max_duration} frames.")
        
        bboxes = np.array(longest_persisting_streak['bboxes'])
        avg_bbox = np.mean(bboxes, axis=0).astype(int)
        longest_persisting_bbox = avg_bbox.tolist()

        print(f"Average Bounding Box: {longest_persisting_bbox}")

        output_dir = "cropped_frames"
        os.makedirs(output_dir, exist_ok=True)

        cap_save = cv2.VideoCapture(video_path)
        frame_idx_save = 0
        
        frames_to_save = range(longest_persisting_streak['start_frame'], longest_persisting_streak['end_frame'] + 1)
        
        while True:
            ret_save, frame_save = cap_save.read()
            if not ret_save:
                break
            
            frame_idx_save += 1
            
            if frame_idx_save in frames_to_save:
                x, y, w, h = longest_persisting_bbox
                h_frame, w_frame, _ = frame_save.shape
                x = max(0, x)
                y = max(0, y)
                w = min(w, w_frame - x)
                h = min(h, h_frame - y)

                cropped_frame = frame_save[y:y+h, x:x+w]
                output_filename = os.path.join(output_dir, f"frame_{frame_idx_save:05d}.png")
                cv2.imwrite(output_filename, cropped_frame)
    else:
        print("No continuously persisting rectangles detected.")

    cap.release()
    cap_save.release()

if __name__ == "__main__":
    video_file = "./input/rmc_01.mp4"
    detect_red_grid(video_file)