import os
import cv2
import torch
import numpy as np
import torchvision
from PIL import Image


def unnormalize(tens, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    # assume tensor of shape NxCxHxW
    return tens * torch.Tensor(std)[None, :, None, None] + torch.Tensor(
        mean)[None, :, None, None]


def get_heatmap_cv(img, magn, max_flow_mag):
    min_flow_mag = .5
    cv_magn = np.clip(
        255 * (magn - min_flow_mag) / (max_flow_mag - min_flow_mag),
        a_min=0,
        a_max=255).astype(np.uint8)
    if img.dtype != np.uint8:
        img = (255 * img).astype(np.uint8)

    heatmap_img = cv2.applyColorMap(cv_magn, cv2.COLORMAP_JET)
    heatmap_img = heatmap_img[..., ::-1]

    h, w = magn.shape
    img_alpha = np.ones((h, w), dtype=np.double)[:, :, None]
    heatmap_alpha = np.clip(
        magn / max_flow_mag, a_min=0, a_max=1)[:, :, None]**.7
    heatmap_alpha[heatmap_alpha < .2]**.5
    pm_hm = heatmap_img * heatmap_alpha
    pm_img = img * img_alpha
    cv_out = pm_hm + pm_img * (1 - heatmap_alpha)
    cv_out = np.clip(cv_out, a_min=0, a_max=255).astype(np.uint8)

    return cv_out


def get_heatmap_batch(img_batch, pred_batch):
    imgrid = torchvision.utils.make_grid(img_batch).cpu()
    magn_batch = torch.norm(pred_batch, p=2, dim=1, keepdim=True)
    magngrid = torchvision.utils.make_grid(magn_batch)
    magngrid = magngrid[0, :, :]
    imgrid = unnormalize(imgrid).squeeze_()

    cv_magn = magngrid.detach().cpu().numpy()
    cv_img = imgrid.permute(1, 2, 0).detach().cpu().numpy()
    cv_out = get_heatmap_cv(cv_img, cv_magn, max_flow_mag=9)
    out = np.asarray(cv_out).astype(np.double) / 255.0

    out = torch.from_numpy(out).permute(2, 0, 1)
    return out


def save_heatmap_cv(img, magn, path, max_flow_mag=7):
    cv_out = get_heatmap_cv(img, magn, max_flow_mag)
    out = Image.fromarray(cv_out)
    out.save(path, quality=95)
