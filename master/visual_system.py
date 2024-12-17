import os
import sys
sys.path.append("/home/zzl/GroundingDINO")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from groundingdino.util.inference import load_model, load_image, predict, annotate
import cv2
import numpy as np
model = load_model("/home/zzl/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py", "/home/zzl/GroundingDINO/weights/groundingdino_swint_ogc.pth")
script_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(script_path)
base_path=parent_directory+"/data/"

def groundingdino(imagefilename,object):
    image_path=base_path+imagefilename


    IMAGE_PATH = image_path
    TEXT_PROMPT = object
    BOX_TRESHOLD = 0.35
    TEXT_TRESHOLD = 0.25
    
    image_source, image = load_image(IMAGE_PATH)
    
    boxes, logits, phrases = predict(
        model=model,
        image=image,
        caption=TEXT_PROMPT,
        box_threshold=BOX_TRESHOLD,
        text_threshold=TEXT_TRESHOLD
    )
    
    annotated_frame,detections = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
    #cv2.imwrite(base_path+"annotated_image.jpg", annotated_frame)
    # 假设已经获取了边界框坐标 (x_min, y_min, x_max, y_max)
   #x_min, y_min, x_max, y_max = 100, 200, 300, 400  # 示例坐标
   
   # 加载原始图像
    image = cv2.imread(IMAGE_PATH)
    x_min=int(detections.xyxy[0][0])-1
    x_max=int(detections.xyxy[0][2])+1
    y_min=int(detections.xyxy[0][1])-1
    y_max=int(detections.xyxy[0][3])+1
   #print(x_max,x_min,y_max,y_min)
   
   # 裁剪边界框区域
    cropped_image = image[y_min:y_max, x_min:x_max]
   
   # 示例：对裁剪后的图像进行灰度化处理
    gray_cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
   #print(gray_cropped_image.shape)
   # 将处理后的图像放回原位置（示例中未展示）
   
   # 保存处理后的图像
    cv2.imwrite(base_path+"cropped_and_processed_image.jpg", gray_cropped_image)
    return x_min,y_min



   


def get_center(image):
   # 使用Canny边缘检测
   edges = cv2.Canny(image, threshold1=100, threshold2=200)
   
   # 使用形态学操作去除噪点并填充边缘间隙
   kernel = np.ones((5, 5), np.uint8)
   edges_dilated = cv2.dilate(edges, kernel, iterations=1)
   edges_closed = cv2.morphologyEx(edges_dilated, cv2.MORPH_CLOSE, kernel)
   
   # 寻找轮廓
   contours, _ = cv2.findContours(edges_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
   
   # 创建一个空图像用于绘制轮廓和重心
   output_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
   
   # 计算所有轮廓的重心
   for contour in contours:
       # 计算轮廓的矩
       M = cv2.moments(contour)
       
       # 计算重心坐标
       if M["m00"] != 0:
           cX = int(M["m10"] / M["m00"])
           cY = int(M["m01"] / M["m00"])
       else:
           cX, cY = 0, 0
       
       # 绘制轮廓
       cv2.drawContours(output_image, [contour], -1, (0, 255, 0), 2)
       
       # 绘制重心
       cv2.circle(output_image, (cX, cY), 2, (255, 0, 0), -1)
       
       print(output_image.shape)
       print(cX,cY)
   # 显示结果
       cv2.imwrite(base_path+"Output_Image.jpg", output_image)
       return cX,cY
#old_image=cv2.imread(base_path+"photo.jpg")
#x_min,y_min=groundingdino("photo.jpg","red square object")
#image = cv2.imread(base_path+"cropped_and_processed_image.jpg", cv2.IMREAD_GRAYSCALE)
#cX,cY=get_center(image)
#cv2.circle(old_image,(x_min+cX,y_min+cY),2,(255, 0, 0), -1)
#cv2.imwrite("test.jpg",old_image)
def get_center1(image):
    a,b=image.shape
    a=int(a/2)
    b=int(b/2)
    return a,b

    

def object_detection(filename,object):
    old_image=cv2.imread(base_path+filename)
    x_min,y_min=groundingdino(filename,object)
    image = cv2.imread(base_path+"cropped_and_processed_image.jpg", cv2.IMREAD_GRAYSCALE)
    cX,cY=get_center1(image)
    X=x_min+cX
    Y=y_min+cY-5
    cv2.circle(old_image,(X,Y),2,(255, 0, 0), -1)
    cv2.imwrite(base_path+"{}.jpg".format(object),old_image)
    return X,Y

object_detection("pick_photo.jpg","blue smaller square object")