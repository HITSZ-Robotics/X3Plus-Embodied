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
base_path = "/home/zzl/Desktop/data/"
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
    #tokenizer = open_clip.get_tokenizer("/data/openclip_tokenizer", direct_load=True)

def ImageCrop(image, masks):
    image = Image.fromarray(image)
    cropped_boxes = []
    used_masks = []
    for mask in masks:
        cropped_boxes.append(
            segment_image(image, mask["segmentation"]).crop(
                convert_box_xywh_to_xyxy(mask["bbox"])
            )
        )
        used_masks.append(mask)
    return cropped_boxes, used_masks

def CLIPRetrieval(objs, query, **kwargs):    #search for the hightest score
    if isinstance(query, str):
        scores = retriev_with_text(objs, query)
    #else:
    #    scores_1 = retriev_with_template_image(objs, query)
    #    scores_2 = retriev_with_template_image(objs, query.rotate(90))
    #    scores = (scores_1 + scores_2) / 2

    obj_idx = get_indices_of_values_above_threshold(scores, 0.1)
    if len(obj_idx) > 1:
        list_remove_element(obj_idx, **kwargs)
    return obj_idx[0]                              


def retriev_with_text(elements: list[Image.Image], search_text: str) -> int:
    if engine == "openai":
        return retriev_clip(elements, search_text)
    elif engine == "openclip":
        return retriev_openclip(elements, search_text)
    else:
        raise Exception("Engine not supported")
    
#!!!!!
@torch.no_grad()
def retriev_openclip(elements: list[Image.Image], search_text: str) -> int:
    preprocessed_images = [preprocess(image).to(device) for image in elements]
    txt = tokenizer(search_text).to(device)
    stacked_images = torch.stack(preprocessed_images)
    img_features = model.encode_image(stacked_images)
    text_features = model.encode_text(txt)
    img_features /= img_features.norm(dim=-1, keepdim=True)  # imgs * 1024
    text_features /= text_features.norm(dim=-1, keepdim=True)  # 1 * 1024
    probs = 100.0 * img_features @ text_features.T
    return probs[:, 0].softmax(dim=0)

@torch.no_grad()
def retriev_clip(elements: list[Image.Image], search_text: str) -> int:
    preprocessed_images = [preprocess(image).to(device) for image in elements]
    tokenized_text = clip.tokenize([search_text]).to(device)
    stacked_images = torch.stack(preprocessed_images)
    image_features = model.encode_image(stacked_images)
    text_features = model.encode_text(tokenized_text)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    probs = 100.0 * image_features @ text_features.T
    return probs[:, 0].softmax(dim=0)

def get_indices_of_values_above_threshold(values, threshold):
    filter_values = {i: v for i, v in enumerate(values) if v > threshold}
    sorted_ids = sorted(filter_values, key=filter_values.get, reverse=True)
    return sorted_ids

def list_remove_element(list_, **kwargs):
    for key in kwargs:
        if "pre_obj" in key:
            try:
                list_.remove(kwargs[key])
            except:
                pass
    return list_


def segment_image(image, segmentation_mask):
    image_array = np.array(image)
    segmented_image_array = np.zeros_like(image_array)
    segmented_image_array[segmentation_mask] = image_array[segmentation_mask]
    segmented_image = Image.fromarray(segmented_image_array)
    black_image = Image.new("RGB", image.size, (0, 0, 0))
    transparency_mask = np.zeros_like(segmentation_mask, dtype=np.uint8)
    transparency_mask[segmentation_mask] = 255
    transparency_mask_image = Image.fromarray(transparency_mask, mode="L")
    black_image.paste(segmented_image, mask=transparency_mask_image)
    return black_image

def convert_box_xywh_to_xyxy(box):
    x1 = box[0]
    y1 = box[1]
    x2 = box[0] + box[2]
    y2 = box[1] + box[3]
    return [x1, y1, x2, y2]


def Pixel2Loc(obj, masks):
    return bbox_to_center(masks[obj]["bbox"])

def bbox_to_center(bbox):
    return (bbox[0] + bbox[2] / 2), (bbox[1] + bbox[3] / 2)  ##outcome the point


##########################################################################  sam aboved



def get_rgb(set_filename):
    color_filename =base_path+set_filename
    image = cv2.imread(color_filename,3)
    image=image.astype("uint8")
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
    #gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #IMAGE = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #gray_img = cv2.GaussianBlur(gray_img, (3, 3), 0)
    #gray = cv2.inRange(gray_img, 47, 150)

    #kernel = np.ones((3, 3), np.uint16)
    #gray = cv2.dilate(gray, kernel, iterations=1)
    #kernel = np.ones((5, 5), np.uint16)
    #gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    #image = cv2.bitwise_and(image, image, mask=gray)

    #empty = np.ones(image.shape, dtype=np.uint8) * 47  # 47 is the background color
    #background = cv2.bitwise_and(empty, empty, mask=cv2.bitwise_not(gray))
    #image = cv2.add(image, background)
    return image
    #eturn IMAGE

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
    #image = remove_boundary(image)
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

def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    print(len(anns))
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([(255,0,0), [0.35]])
        img[m] = color_mask
    ax.imshow(img)
#for i in range (1,5) :
#
#  IMAGE=get_rgb("LLM_test{}").format(i)
#  
#  MASKS=SAM(image=IMAGE)
#  plt.figure(figsize=(20,20))
#  plt.imshow(IMAGE)
#  
#  show_anns(MASKS)
#  plt.axis('off')
#  plt.show() 
#  IMAGE1 =cv2.cvtColor(IMAGE,cv2.COLOR_BGR2RGB)
#  OBJS, MASKS=ImageCrop(image=IMAGE1, masks=MASKS)
#  
#  OBJ0=CLIPRetrieval(objs=OBJS, query='the red square object')
#  LOC0=Pixel2Loc(obj=OBJ0, masks=MASKS)
#  print(LOC0)
#  
#  x=int(LOC0[0])
#  y=int(LOC0[1]) 
#  cv2.circle(IMAGE,(x,y),10,(0,0,255),-1)
#  cv2.imshow("window",IMAGE)
#  cv2.waitKey(0)
def object_detection(set_filename,str)  :
    IMAGE=get_rgb(set_filename)
    
    MASKS=SAM(image=IMAGE)
    #plt.figure(figsize=(20,20))
    #plt.imshow(IMAGE)
    
    #show_anns(MASKS)
    #plt.axis('off')
    #plt.show() 
    IMAGE1 =cv2.cvtColor(IMAGE,cv2.COLOR_BGR2RGB)
    OBJS, MASKS=ImageCrop(image=IMAGE1, masks=MASKS)
    
    OBJ0=CLIPRetrieval(objs=OBJS, query=str)
    LOC0=Pixel2Loc(obj=OBJ0, masks=MASKS)
    print(LOC0)
    
    x=int(LOC0[0])
    y=int(LOC0[1]) 
    cv2.circle(IMAGE,(x,y),10,(0,0,255),-1)
    cv2.imshow("window",IMAGE)
    cv2.waitKey(40000000)
    return x,y
#image=get_rgb("photo.png")
#cv2.imshow("window",image)
#cv2.waitKey(0)
#my_choice="upside down bottle"   
#x,y=object_detection("photo.png",my_choice)
#print(x,y)
