import argparse
import os
import sys
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image

from networks.drn_seg import DRNSeg
from utils.tools import *
from utils.visualize import *


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_path", required=True, help="the model input")
    parser.add_argument(
        "--dest_folder", required=True, help="folder to store the results")
    parser.add_argument(
        "--model_path", required=True, help="path to the drn model")
    parser.add_argument(
        "--gpu_id", default='0', help="the id of the gpu to run model on")
    parser.add_argument(
        "--no_crop",
        action="store_true",
        help="do not use a face detector, instead run on the full input image")
    args = parser.parse_args()

    img_path = args.input_path
    dest_folder = args.dest_folder
    model_path = args.model_path
    gpu_id = args.gpu_id

    # Loading the model
    if torch.cuda.is_available():
        device = 'cuda:{}'.format(gpu_id)
    else:
        device = 'cpu'

    model = DRNSeg(2)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict['model'])
    model.to(device)
    model.eval()

    # Data preprocessing
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    im_w, im_h = Image.open(img_path).size
    if args.no_crop:
        face = Image.open(img_path).convert('RGB')
    else:
        faces = face_detection(img_path, verbose=False)
        if len(faces) == 0:
            print("no face detected by dlib, exiting")
            sys.exit()
        face, box = faces[0]
    face = resize_shorter_side(face, 400)[0]
    face_tens = tf(face).to(device)

    # Warping field prediction
    with torch.no_grad():
        flow = model(face_tens.unsqueeze(0))[0].cpu().numpy()
        flow = np.transpose(flow, (1, 2, 0))
        h, w, _ = flow.shape

    # Undoing the warps
    modified = face.resize((w, h), Image.BICUBIC)
    modified_np = np.asarray(modified)
    reverse_np = warp(modified_np, flow)
    reverse = Image.fromarray(reverse_np)

    # Saving the results
    modified.save(
        os.path.join(dest_folder, 'cropped_input.jpg'),
        quality=90)
    reverse.save(
        os.path.join(dest_folder, 'warped.jpg'),
        quality=90)
    flow_magn = np.sqrt(flow[:, :, 0]**2 + flow[:, :, 1]**2)
    save_heatmap_cv(
        modified_np, flow_magn,
        os.path.join(dest_folder, 'heatmap.jpg'))
