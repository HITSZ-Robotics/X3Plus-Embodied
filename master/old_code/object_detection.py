
import numpy as np
import cv2

# 图像的基本信息
width = 640  # 图像宽度
height = 480  # 图像高度
channels = 1  # 图像通道数，例如3表示RGB
dtype = 'uint16'  # 数据类型，根据实际情况可能是'uint8'或'uint16'等

# 使用numpy从RAW文件读取数据
with open('depth_object2.raw', 'rb') as f:
    img_data = np.fromfile(f, dtype=np.uint16)

# 根据图像尺寸重塑数组
#img = img_data.reshape(height, width,channels)

print(img_data)

# 转换为 uint8
#img_uint8 = (img/ 256).astype('uint8')
#print(img_uint8)
# 转换为灰度图像
# apply colormap on depth image(image must be converted to 8-bit per pixel first)
#img_color = cv2.applyColorMap(cv2.convertScaleAbs(img_uint8, alpha=255.0 / img_uint8.max()), cv2.COLORMAP_JET)


#print(img_color[6])
# 保存图像
#cv2.imwrite('object.png', img_color)
#print(img_color)
# 显示图像
#cv2.imshow('Colored Depth Image', img_color)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

## 1.21 图像通道的合并
#img1 = cv2.imread("../images/imgB1.jpg", flags=1)  # flags=1 读取彩色图像(BGR)
#bImg, gImg, rImg = cv2.split(img1)  # 拆分为 BGR 独立通道
## cv2.merge 实现图像通道的合并
#imgMerge = cv2.merge([bImg, gImg, rImg])
#cv2.imshow("cv2Merge", imgMerge)
#
## Numpy 拼接实现图像通道的合并
#imgStack = np.stack((bImg, gImg, rImg), axis=2)
#cv2.imshow("npStack", imgStack)
#
#print(imgMerge.shape, imgStack.shape)
#print("imgMerge is imgStack?", np.array_equal(imgMerge, imgStack))
#
#cv2.waitKey(0)
#cv2.destroyAllWindows()  # 释放所有窗口