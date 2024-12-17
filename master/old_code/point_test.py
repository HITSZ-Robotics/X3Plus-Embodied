import cv2
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
#import clip
import torch
import copy
import numpy as np

# used for storing the click location
from matplotlib.backend_bases import MouseButton
import matplotlib.pyplot as plt
click_point_x, click_point_y = 0, 0
click_points = []
cid, fig = None, None
#from transformers import AutoProcessor, AutoModelForZeroShotImageClassification, AutoTokenizer
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
    #model, _, preprocess = open_clip.create_model_and_transforms(
     #    "ViT-H-14", device=device, pretrained="laion2b_s32b_b79k"
     #)
    open_clip_path = "/home/zzl/Desktop/"
    model_cards = {
        "ViT-B-16": "ViT-B-16_openai.pt",
        "ViT-B-32": "ViT-B-32_openai.pt",
        "ViT-L-14": "ViT-L-14_laion2b_s32b_b82k.pt",
        "ViT-H-14": "open_clip_pytorch_model.bin",
    }
    models = list(model_cards.keys())
    # add offline model for OpenCLIP
    #model, _, preprocess = open_clip.create_model_and_transforms(
     #    "ViT-H-14", device=device, pretrained="laion2b_s32b_b79k"
     #)
    model, _, preprocess = open_clip.create_model_and_transforms(
         "ViT-H-14", device=device, pretrained="/home/zzl/Desktop/open_clip_pytorch_model.bin"
     )
    tokenizer = open_clip.get_tokenizer("ViT-H-14")
    clip_index = 3  # default to use the VIT-H-14
    model, _, preprocess = open_clip.create_model_and_transforms(
        models[clip_index],
        device=device,
        pretrained=os.path.join(open_clip_path, model_cards[models[clip_index]]),
    )
def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)
  
 
IMAGE = cv2.imread("color_test30.png")
#cv2.imwrite("color_object2.jpg","color_object2.png")
#IMAGE=cv2.imread("color_object2.jpg")
IMAGE = cv2.cvtColor(IMAGE, cv2.COLOR_BGR2RGB)
#cv2.imshow("window",IMAGE)
#cv2.waitKey(0)
mask2=mask_generator.generate(IMAGE)
len(mask2)
#plt.figure(figsize=(20,20))
#plt.imshow(IMAGE)
#show_anns(mask2)
#plt.axis('off')
#plt.show() 
#point=(320.5,450.9)
#x=int(point[0])
#y=int(point[1]) 
#cv2.circle(IMAGE,(x,y),10,(0,0,255),-1)
#cv2.imshow("window",IMAGE)
#cv2.waitKey(0)