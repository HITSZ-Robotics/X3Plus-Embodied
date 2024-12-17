import imageio.v3 as iio
import numpy as np
import matplotlib.pyplot as plt
import cv2
import numpy as np
import cv2
import rawpy
import imageio
def bin2numpy_ex(file_path, shape):
    rawImg = np.fromfile(file_path, dtype=np.uint16)
    rawImg = rawImg[: shape[0]*shape[1]]
    pic_gray = rawImg.reshape(shape)

    return pic_gray

#rawImg = np.fromfile('depth_object1.raw',dtype=np.uint16)

#rawImg = rawImg.reshape()
#print(rawImg)
#mg=rawpy.imread("depth_object3.raw")
  
#rgb=img.postprocess()

  #depth_scale=0.001    #!!!!!!!!!!
#array_data=mg.raw_image_visible##//读取RAW图像数据信息

#print(array_data.min())
##
##path='depth_object2.raw'
##with rawpy.imread(path) as raw:
##    img=raw.raw_image
##bit_depth=np.log2(img.max()+1)
##print(bit_depth)
##
##img=rawpy.imread("depth_object2.raw")
#img = np.fromfile('depth_object2.raw',dtype=np.uint16)
#img=img.reshape((480,640))
#print(img.shape)
#print(img)
#gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
##rgb=img.postprocess()
#

#array_data=img.raw_image_visible##//读取RAW图像数据信息
#width = array_data.shape[0]
##height = array_data.shape[1]
#bit_depth=array_data[:,:,1:2]/256  ##change the channel
#print(bit_depth.shape)####//打印长和宽
#print(bit_depth[3,4])##//打印矩阵信息
#————————————————
#
 #                           版权声明：本文为博主原创文章，遵循 CC 4.0 BY-SA 版权协议，转载请附上原文出处链接和本声明。
                        
#原文链接：https://blog.csdn.net/weixin_44690935/article/details/124345747
## 图像的基本信息
width = 640  # 图像宽度
height = 480  # 图像高度
channels = 1  # 图像通道数，例如3表示RGB
dtype = 'uint16'  # 数据类型，根据实际情况可能是'uint8'或'uint16'等
#
## 使用numpy从RAW文件读取数据
with open('depth_object2.raw', 'rb') as f:
    img_data = np.fromfile(f, dtype=np.uint16)
#
## 根据图像尺寸重塑数组
gray= img_data.reshape(height, width, channels)
print(gray[240,320])
##import open3d as o3d
## Read depth image:
##depth_image = iio.imread('object.png')
#depth_image = iio.imread('color_object2.png')
# 
#
#gray = cv2.cvtColor(depth_image, cv2.COLOR_BGR2GRAY)
#print(gray.shape)
#cv2.imshow('my_picture',gray)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
### print properties:
print(f"Image resolution: {gray.shape}")
print(f"Data type: {gray.dtype}")
print(f"Min value: {np.min(gray)}")
print(f"Max value: {np.max(gray)}")
