#!/usr/bin/env python3

import os
import shutil
import argparse
import cv2
import numpy as np
import glob
import torch
from PIL import Image

# RAFT model import (프로젝트 구조에 맞게 core 폴더를 추가)
import sys
sys.path.append('core')
from raft import RAFT
from utils.utils import InputPadder

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]
    return interArea / float(boxAArea + boxBArea - interArea)


def get_longest_persisting_bbox(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    frame_count = 0
    all_streaks = []
    active_streaks = {}
    next_id = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        # 전처리 및 빨간색 마스크 생성
        blur = cv2.medianBlur(frame, 5)
        hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(hsv, (0,100,100), (10,255,255))
        mask2 = cv2.inRange(hsv, (170,100,100), (180,255,255))
        mask = cv2.bitwise_or(mask1, mask2)
        kernel = np.ones((7,7), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=10)
        mask = cv2.erode(mask, kernel, iterations=10)

        # 그리드 선만 남기고 반전·클린업
        grid = np.zeros_like(frame)
        grid[mask>0] = 255
        grid = cv2.medianBlur(grid,5)
        grid = cv2.bitwise_not(grid)
        post_k = np.ones((15,15), np.uint8)
        grid = cv2.erode(grid, post_k, iterations=10)
        grid = cv2.dilate(grid, post_k, iterations=8)
        gray = cv2.cvtColor(grid, cv2.COLOR_BGR2GRAY)
        cnts, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detections = []
        for cnt in cnts:
            eps = 0.05 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, eps, True)
            if len(approx) == 4:
                area = cv2.contourArea(approx)
                x,y,w,h = cv2.boundingRect(approx)
                ar = w / float(h)
                if 25000 < area < 45000 and 1.0 <= ar <= 1.5:
                    detections.append((x,y,w,h))

        matched = set()
        new_dets = []
        for det in detections:
            matched_flag = False
            for rid, info in active_streaks.items():
                if iou(det, info['bboxes'][-1]) > 0.5:
                    info['end'] = frame_count
                    info['bboxes'].append(det)
                    matched.add(rid)
                    matched_flag = True
                    break
            if not matched_flag:
                new_dets.append(det)

        # 종료된 streaks
        for rid in list(active_streaks):
            if rid not in matched:
                all_streaks.append(active_streaks.pop(rid))

        # 신규 streak 시작
        for det in new_dets:
            active_streaks[next_id] = {
                'start': frame_count,
                'end': frame_count,
                'bboxes': [det]
            }
            next_id += 1

    # 남은 활성 streak 기록
    all_streaks.extend(active_streaks.values())
    cap.release()

    # 가장 오래 지속된 streak 선택
    best = None
    max_dur = 0
    for info in all_streaks:
        dur = info['end'] - info['start'] + 1
        if dur > max_dur:
            max_dur = dur
            best = info

    if not best:
        raise RuntimeError("No grid streak detected.")

    # 평균 bbox 계산
    arr = np.array(best['bboxes'])
    avg = np.mean(arr, axis=0).astype(int)
    return tuple(avg.tolist())  # (x, y, w, h)


def crop_frames(video_path, bbox, out_dir):
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)

    cap = cv2.VideoCapture(video_path)
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        idx += 1
        x,y,w,h = bbox
        h_f, w_f = frame.shape[:2]
        x2 = min(x+w, w_f)
        y2 = min(y+h, h_f)
        crop = frame[y:y2, x:x2]
        cv2.imwrite(os.path.join(out_dir, f"frame_{idx:05d}.png"), crop)
    cap.release()


def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2,0,1).float()
    return img[None].to(DEVICE)


def viz(img, flo, padder, imfile, out_dir):
    img_np = img[0].permute(1,2,0).cpu().numpy().astype(np.uint8)
    flo_np = flo[0].permute(1,2,0).cpu().numpy()
    img_un = padder.unpad(img_np)
    bgr = cv2.cvtColor(img_un, cv2.COLOR_RGB2BGR)
    stride = 16
    h, w = bgr.shape[:2]
    for y in range(0, h, stride):
        for x in range(0, w, stride):
            dx, dy = flo_np[y, x]
            if (dx*dx+dy*dy)**0.5 > 1.0:
                cv2.arrowedLine(bgr, (x,y), (int(round(x+dx)), int(round(y+dy))), (0,255,0), 1, tipLength=0.3)
    fname = os.path.basename(imfile).replace('.png','_flow.png')
    cv2.imwrite(os.path.join(out_dir, fname), bgr)


def run_raft(model_path, input_dir, output_dir, interval):
    # prepare output folder
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    # load model
    parser = argparse.ArgumentParser()
    parser = RAFT.default_args(parser)
    parser.set_defaults(model=model_path, small=False, mixed_precision=False, alternate_corr=False)
    args = parser.parse_args([])
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(model_path))
    model = model.module.to(DEVICE)
    model.eval()

    # process frames
    imgs = sorted(glob.glob(os.path.join(input_dir, '*.png')))
    for im1, im2 in zip(imgs[:-interval], imgs[interval:]):
        image1 = load_image(im1)
        image2 = load_image(im2)
        padder = InputPadder(image1.shape)
        i1, i2 = padder.pad(image1, image2)
        with torch.no_grad():
            _, flow_up = model(i1, i2, iters=20, test_mode=True)
        flow_up = padder.unpad(flow_up)
        viz(image1, flow_up, padder, im1, output_dir)


def main():
    parser = argparse.ArgumentParser(description="Crop red grid region then run RAFT optical flow")
    parser.add_argument('--model',      required=True, help="RAFT checkpoint (.pth)")
    parser.add_argument('--input_path', required=True, help="Input video file")
    parser.add_argument('--output_path', required=True, help="Base output directory")
    parser.add_argument('--interval',   type=int, default=1, help="Frame interval for flow")
    args = parser.parse_args()

    # 1) ROI 검출 및 평균 bbox 계산
    print("Detecting red grid region...")
    bbox = get_longest_persisting_bbox(args.input_path)
    print(f"Detected bounding box: {bbox}")

    # 2) 프레임 크롭
    cropped_dir = os.path.join(args.output_path, 'cropped')
    print("Cropping frames...")
    crop_frames(args.input_path, bbox, cropped_dir)

    # 3) RAFT 실행
    flow_dir = os.path.join(args.output_path, 'flow_arrows')
    print("Running RAFT optical flow...")
    run_raft(args.model, cropped_dir, flow_dir, args.interval)

    print(f"Done. Cropped frames in `{cropped_dir}`, flow visuals in `{flow_dir}`.")


if __name__ == '__main__':
    main()
