import os
import cv2
import torch
import numpy as np
from PIL import Image
from dlib import cnn_face_detection_model_v1 as face_detect_model


def center_crop(im, length):
    w, h = im.size
    left = w//2 - length//2
    right = w//2 + length//2
    top = h//2 - length//2
    bottom = h//2 + length//2
    return im.crop((left, top, right, bottom)), (left, top)


def remove_boundary(img):
    """
    Remove boundary artifacts that FAL causes.
    """
    w, h = img.size
    left = w//80
    top = h//50
    right = w*79//80
    bottom = h*24//25
    return img.crop((left, top, right, bottom))


def resize_shorter_side(img, min_length):
    """
    Resize the shorter side of img to min_length while
    preserving the aspect ratio.
    """
    ow, oh = img.size
    mult = 8
    if ow < oh:
        if ow == min_length and oh % mult == 0:
            return img, (ow, oh)
        w = min_length
        h = int(min_length * oh / ow)
    else:
        if oh == min_length and ow % mult == 0:
            return img, (ow, oh)
        h = min_length
        w = int(min_length * ow / oh)
    return img.resize((w, h), Image.BICUBIC), (w, h)


def flow_resize(flow, sz):
    oh, ow, _ = flow.shape
    w, h = sz
    u_ = cv2.resize(flow[:,:,0], (w, h))
    v_ = cv2.resize(flow[:,:,1], (w, h))
    u_ *= w / float(ow)
    v_ *= h / float(oh)
    return np.dstack((u_,v_))


def warp(im, flow, alpha=1, interp=cv2.INTER_CUBIC):
    height, width, _ = flow.shape
    cart = np.dstack(np.meshgrid(np.arange(width), np.arange(height)))
    pixel_map = (cart + alpha * flow).astype(np.float32)
    warped = cv2.remap(
        im,
        pixel_map[:, :, 0],
        pixel_map[:, :, 1],
        interp,
        borderMode=cv2.BORDER_REPLICATE)
    return warped


cnn_face_detector = None
def face_detection(
        img_path,
        verbose=False,
        model_file='utils/dlib_face_detector/mmod_human_face_detector.dat'):
    """
    Detects faces using dlib cnn face detection, and extend the bounding box
    to include the entire face.
    """
    def shrink(img, max_length=2048):
        ow, oh = img.size
        if max_length >= max(ow, oh):
            return img, 1.0

        if ow > oh:
            mult = max_length / ow
        else:
            mult = max_length / oh
        w = int(ow * mult)
        h = int(oh * mult)
        return img.resize((w, h), Image.BILINEAR), mult

    global cnn_face_detector
    if cnn_face_detector is None:
        cnn_face_detector = face_detect_model(model_file)

    img = Image.open(img_path).convert('RGB')
    w, h = img.size
    img_shrinked, mult = shrink(img)

    im = np.asarray(img_shrinked)
    if len(im.shape) != 3 or im.shape[2] != 3:
        return []

    crop_ims = []
    dets = cnn_face_detector(im, 0)
    for k, d in enumerate(dets):
        top = d.rect.top() / mult
        bottom = d.rect.bottom() / mult
        left = d.rect.left() / mult
        right = d.rect.right() / mult

        wid = right - left
        left = max(0, left - wid // 2.5)
        top = max(0, top - wid // 1.5)
        right = min(w - 1, right + wid // 2.5)
        bottom = min(h - 1, bottom + wid // 2.5)

        if d.confidence > 1:
            if verbose:
                print("%d-th face detected: (%d, %d, %d, %d)" %
                      (k, left, top, right, bottom))
            crop_im = img.crop((left, top, right, bottom))
            crop_ims.append((crop_im, (left, top, right, bottom)))

    return crop_ims


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
