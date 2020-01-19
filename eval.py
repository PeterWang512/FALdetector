import glob
import argparse
import torch
import torchvision.transforms as transforms
import numpy as np
from PIL import Image

from networks.drn_seg import DRNSeg, DRNSub
from utils.tools import *
from utils.visualize import *
from sklearn.metrics import average_precision_score, accuracy_score


def load_global_classifier(model_path, gpu_id):
    if torch.cuda.is_available() and gpu_id != -1:
        device = 'cuda:{}'.format(gpu_id)
    else:
        device = 'cpu'
    model = DRNSub(1)
    state_dict = torch.load(model_path, map_location='cpu')
    model.load_state_dict(state_dict['model'])
    model.to(device)
    model.device = device
    model.eval()
    return model


def load_local_detector(model_path, gpu_id):
    if torch.cuda.is_available():
        device = 'cuda:{}'.format(gpu_id)
    else:
        device = 'cpu'

    model = DRNSeg(2)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict['model'])
    model.to(device)
    model.device = device
    model.eval()
    return model


tf = transforms.Compose([transforms.ToTensor(),
                         transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])])
def load_data(img_path, device):
    face = Image.open(img_path).convert('RGB')
    face = resize_shorter_side(face, 400)[0]
    face_tens = tf(face).to(device)
    return face_tens, face


def classify_fake(model, img_path):
    img = load_data(img_path, model.device)[0].unsqueeze(0)
    # Prediction
    with torch.no_grad():
        prob = model(img)[0].sigmoid().cpu().item()
    return prob


def calc_psnr(img0, img1, mask=None):
    return -10 * np.log10(np.mean((img0 - img1)**2) + 1e-6)


def detect_warp(model, img_path):
    img, modified = load_data(img_path, model.device)
    # Warping field prediction
    with torch.no_grad():
        flow = model(img.unsqueeze(0))[0].cpu().numpy()
        flow = np.transpose(flow, (1, 2, 0))

    # Undoing the warps
    flow = flow_resize(flow, modified.size)
    modified_np = np.asarray(modified)
    reverse_np = warp(modified_np, flow)
    original = Image.open(img_path.replace('modified', 'reference')).convert('RGB')
    original_np = np.asarray(original.resize(modified.size, Image.BICUBIC))

    psnr_before = calc_psnr(original_np / 255, modified_np / 255)
    psnr_after = calc_psnr(original_np / 255, reverse_np / 255)
    return psnr_before, psnr_after


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataroot", required=True, help='the root to the dataset')
    parser.add_argument(
        "--global_pth", required=True, help="path to the global model")
    parser.add_argument(
        "--local_pth", required=True, help="path to the local model")
    parser.add_argument(
        "--gpu_id", default='0', help="the id of the gpu to run model on")
    args = parser.parse_args()

    glb_model = load_global_classifier(args.global_pth, args.gpu_id)
    lcl_model = load_local_detector(args.local_pth, args.gpu_id)

    pred_prob, gt_prob, psnr_before, psnr_after = [], [], [], []
    for img_path in glob.glob(args.dataroot + '/original/*'):
        pred_prob.append(classify_fake(glb_model, img_path))
        gt_prob.append(0)

    for img_path in glob.glob(args.dataroot + '/modified/*'):
        pred_prob.append(classify_fake(glb_model, img_path))
        gt_prob.append(1)
        psnrs = detect_warp(lcl_model, img_path)
        psnr_before.append(psnrs[0])
        psnr_after.append(psnrs[1])

    pred_prob, gt_prob, psnr_before, psnr_after = \
        np.array(pred_prob), np.array(gt_prob), np.array(psnr_before), np.array(psnr_after)
    acc = accuracy_score(gt_prob, pred_prob > 0.5)
    avg_precision = average_precision_score(gt_prob, pred_prob)
    delta_psnr = psnr_after.mean() - psnr_before.mean()

    print("Accuracy: ", acc)
    print("Average precision: ", avg_precision)
    print("PSNR increase: ", delta_psnr)
