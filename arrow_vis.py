import sys
sys.path.append('core')

import argparse
import os
import cv2
import glob
import numpy as np
import torch
from PIL import Image

from raft import RAFT
from utils import flow_viz
from utils.utils import InputPadder



DEVICE = 'cuda'

def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)


def viz(img, flo, padder, imfile1):
    # Convert torch tensors to numpy arrays
    img_numpy = img[0].permute(1,2,0).cpu().numpy().astype(np.uint8)
    flo_numpy = flo[0].permute(1,2,0).cpu().numpy()

    # Unpad the image to its original size
    img_unpadded = padder.unpad(img_numpy)
    
    # Create a BGR version for OpenCV drawing
    img_bgr = cv2.cvtColor(img_unpadded, cv2.COLOR_RGB2BGR)

    # Draw arrows
    stride = 16
    h, w = img_bgr.shape[:2]
    for y in range(0, h, stride):
        for x in range(0, w, stride):
            dx, dy = flo_numpy[y, x]
            # Draw arrows for significant motion
            if np.sqrt(dx**2 + dy**2) > 1.0:
                pt1 = (x, y)
                pt2 = (int(round(x + dx)), int(round(y + dy)))
                cv2.arrowedLine(img_bgr, pt1, pt2, (0, 255, 0), 1, tipLength=0.3)

    # Generate output filename and save
    output_filename = os.path.basename(imfile1).replace('.png', '_arrows.png')
    output_path = os.path.join('demo-frames', output_filename)
    cv2.imwrite(output_path, img_bgr)
    print(f"Saved arrow visualization to: {output_path}")


def demo(args):
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))

    model = model.module
    model.to(DEVICE)
    model.eval()

    with torch.no_grad():
        images = glob.glob(os.path.join(args.path, '*.png')) + \
                 glob.glob(os.path.join(args.path, '*.jpg'))
        
        images = sorted(images)
        for imfile1, imfile2 in zip(images[:-1], images[1:]):
            image1 = load_image(imfile1)
            image2 = load_image(imfile2)

            padder = InputPadder(image1.shape)
            image1_padded, image2_padded = padder.pad(image1, image2)

            flow_low, flow_up = model(image1_padded, image2_padded, iters=20, test_mode=True)
            
            # The flow_up is for the padded image, we need to unpad it
            flow_up_unpadded = padder.unpad(flow_up)

            # Pass the original (unpadded) image, the unpadded flow, the padder, and the filename to viz
            viz(image1, flow_up_unpadded, padder, imfile1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--path', help="dataset for evaluation")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    args = parser.parse_args()

    demo(args)
