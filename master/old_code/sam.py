import sys
sys.path.append("/home/zzl/segment-anything")
import cv2
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from segment_anything import (
    # for automatic mask generation
    build_sam,
    SamAutomaticMaskGenerator,
    build_sam_vit_b,
    build_sam_vit_l,
    build_sam_vit_h,
    # for mask generation with user input like click
    sam_model_registry,
    SamPredictor,
)
from PIL import Image, ImageDraw
import old_code.clip as clip
import torch
import copy
import numpy as np


# used for storing the click location
from matplotlib.backend_bases import MouseButton
import matplotlib.pyplot as plt
click_point_x, click_point_y = 0, 0
click_points = []
cid, fig = None, None

import open_clip

from scipy.spatial.transform import Rotation as R
from scipy.optimize import linear_sum_assignment


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Loading SAM...")
# mask_generator = SamAutomaticMaskGenerator(
#     build_sam(checkpoint="/home/zzl/Desktop/sam_vit_h_4b8939.pth", device=device)
# )

sam_pred_with_click = False
sam_path = "/home/zzl/Desktop/"
sam_model = ["sam_vit_b_01ec64.pth", "sam_vit_l_0b3195.pth", "sam_vit_h_4b8939.pth"]
build_sam_func = [build_sam_vit_b, build_sam_vit_l, build_sam_vit_h]
sam_idx = 2  # default to use the sam_vit_h
if not sam_pred_with_click:
    mask_generator = SamAutomaticMaskGenerator(
        build_sam_func[sam_idx](
            checkpoint=os.path.join(sam_path, sam_model[sam_idx]), device=device
        )
    )
else:
    sam = sam_model_registry["default"](
        checkpoint=os.path.join(sam_path, sam_model[sam_idx])
    )
    sam.to(device=device)
    mask_generator = SamPredictor(sam)

engine = "openclip"  # "openclip" or "clip"
if engine == "clip":
    print("Loading CLIP...")
    model, preprocess = clip.load("ViT-L/14", device=device)
elif engine == "openclip":
    print("Loading OpenCLIP CLIP...")
    # add offline model for OpenCLIP
    model, _, preprocess = open_clip.create_model_and_transforms(
         "ViT-H-14", device=device, pretrained="laion2b_s32b_b79k"
     )
    open_clip_path = "/home/zzl/Desktop/"
    model_cards = {
        "ViT-B-16": "ViT-B-16_openai.pt",
        "ViT-B-32": "ViT-B-32_openai.pt",
        "ViT-L-14": "ViT-L-14_laion2b_s32b_b82k.pt",
        "ViT-H-14": "open_clip_pytorch_model.bin",
    }
    models = list(model_cards.keys())
    # add offline model for OpenCLIP
    model, _, preprocess = open_clip.create_model_and_transforms(
         "ViT-H-14", device=device, pretrained="laion2b_s32b_b79k"
     )
    model, _, preprocess = open_clip.create_model_and_transforms(
         "ViT-H-14", device=device, pretrained="/home/zzl/Desktop/open_clip_pytorch_model.bin"
     )
    # tokenizer = open_clip.get_tokenizer("ViT-H-14")
    clip_index = 3  # default to use the VIT-H-14
    model, _, preprocess = open_clip.create_model_and_transforms(
        models[clip_index],
        device=device,
        pretrained=os.path.join(open_clip_path, model_cards[models[clip_index]]),
    )
    tokenizer = open_clip.get_tokenizer("/data/openclip_tokenizer", direct_load=True)


def get_rgb(set_filename):
    color_filename ="color_"+set_filename+".png"
    image = cv2.imread(color_filename,1)
    return image

def get_click_point(event):
    global click_points
    global cid, fig
    if event.button is MouseButton.LEFT:
        point_x = int(event.xdata)
        point_y = int(event.ydata)
        click_points.append([point_x, point_y])
    elif event.button is MouseButton.RIGHT:
        # print('disconnecting callback')
        fig.canvas.mpl_disconnect(cid)



    
def remove_boundary(image, boundary_length=4):
    image[0 : int(boundary_length / 2), :, :] = 47
    image[-boundary_length:, :, :] = 47
    image[:, 0:boundary_length, :] = 47
    image[:, -boundary_length:, :] = 47
    return image

def image_preprocess(image):
    # shadow remove to avoid ghost mask
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_img = cv2.GaussianBlur(gray_img, (3, 3), 0)
    gray = cv2.inRange(gray_img, 47, 150)

    kernel = np.ones((3, 3), np.uint8)
    gray = cv2.dilate(gray, kernel, iterations=1)
    kernel = np.ones((5, 5), np.uint8)
    gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    image = cv2.bitwise_and(image, image, mask=gray)

    empty = np.ones(image.shape, dtype=np.uint8) * 47  # 47 is the background color
    background = cv2.bitwise_and(empty, empty, mask=cv2.bitwise_not(gray))
    image = cv2.add(image, background)
    return image

def nms(bboxes, scores, iou_thresh):
    x1 = bboxes[:, 0]
    y1 = bboxes[:, 1]
    x2 = x1 + bboxes[:, 2]
    y2 = y1 + bboxes[:, 3]
    areas = (y2 - y1) * (x2 - x1)

    result = []
    index = scores.argsort()[::-1]
    while index.size > 0:
        i = index[0]
        result.append(i)

        x11 = np.maximum(x1[i], x1[index[1:]])
        y11 = np.maximum(y1[i], y1[index[1:]])
        x22 = np.minimum(x2[i], x2[index[1:]])
        y22 = np.minimum(y2[i], y2[index[1:]])
        w = np.maximum(0, x22 - x11 + 1)
        h = np.maximum(0, y22 - y11 + 1)
        overlaps = w * h
        ious = overlaps / (areas[i] + areas[index[1:]] - overlaps)
        idx = np.where(ious <= iou_thresh)[0]
        index = index[idx + 1]
    return result

def mask_preprocess(MASKS):
    MASKS_filtered = []
    for MASK in MASKS:
        if MASK["bbox"][2] < 10 or MASK["bbox"][3] < 10:
            continue
        if MASK["bbox"][2] > 100 or MASK["bbox"][3] > 100:
            continue
        if MASK["area"] < 100:
            continue

        mask = MASK["segmentation"]
        mask = mask.astype("uint8") * 255
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=1)
        mask = cv2.dilate(mask, kernel, iterations=1)
        if np.count_nonzero(mask) < 50:
            continue  # too small, ignore to avoid empty operation
        MASK["area"] = np.count_nonzero(mask)
        ys, xs = np.nonzero(mask)
        xmin, xmax = np.min(xs), np.max(xs)
        ymin, ymax = np.min(ys), np.max(ys)
        MASK["bbox"] = [xmin, ymin, xmax - xmin, ymax - ymin]
        MASK["segmentation"] = mask.astype("bool")
        MASKS_filtered.append(MASK)

    bboxs = np.asarray([MASK["bbox"] for MASK in MASKS_filtered])
    areas = np.asarray([MASK["area"] for MASK in MASKS_filtered])

    result = nms(bboxs, areas, 0.3)
    MASKS_filtered = [MASKS_filtered[i] for i in result]
    return MASKS_filtered

def unified_mask_representation(masks):
    """
        input: masks: [N, H, W], numpy.ndarray
        output: masks: list(dict(segmentation, bbox, area)) -> bbox XYWH
    """
    MASKS = [] 
    for mask in masks:
        MASK = {}
        MASK["segmentation"] = mask.astype("bool")
        MASK["area"] = np.count_nonzero(mask)
        ys, xs = np.nonzero(mask)
        xmin, xmax = np.min(xs), np.max(xs)
        ymin, ymax = np.min(ys), np.max(ys)
        MASK["bbox"] = [xmin, ymin, xmax - xmin, ymax - ymin]
        MASKS.append(MASK)
    return MASKS

def SAM(image, with_click=sam_pred_with_click, image_preprocess_flag=True, mask_preprocess_flag=True):
    image = remove_boundary(image)
    if image_preprocess_flag:
        image = image_preprocess(image)
    assert (
        with_click == sam_pred_with_click
    ), "the initialzaition of sam is not consistent with the current setting"

    if with_click:
        # use cursor click to guide the SAM
        mask_generator.set_image(image)
        global cid, fig, click_points
        fig = plt.figure(1)
        ax = fig.add_subplot(111)
        ax.imshow(image)
        cid = fig.canvas.mpl_connect("button_press_event", get_click_point)
        plt.show()
        plt.waitforbuttonpress()
        ax.clear()
        plt.close('all') 

        masks_all_click = []
        for point in click_points:
            input_point = np.array([[point[0], point[1]]])
            input_label = np.array([1])
            masks, _, _ = mask_generator.predict(
                point_coords=input_point,
                point_labels=input_label,
                multimask_output=True,
            )
            masks_all_click.extend(masks)
        MASKS = unified_mask_representation(masks_all_click)
    else:
        MASKS = mask_generator.generate(image)
    if mask_preprocess_flag:
        MASKS = mask_preprocess(MASKS)
    return MASKS
