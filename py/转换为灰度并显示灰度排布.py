# usr/bin/python
# -*- coding:utf-8 -*-

"""
缩放、近邻插值、 双线性插值
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 读取图像
img = cv2.imread("./img/tst.png")

# 将彩色图像转换为灰度图像
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow("gray", gray)

# 将图像转换为NumPy数组
img_array = np.array(gray)
# iarr = np.ones(img_array.shape, dtype=np.uint8) * 255
# img_array = iarr - img_array

# 获取图像的大小
height, width = img_array.shape

# 创建网格
x = np.linspace(0, width-1, width)
y = np.linspace(0, height-1, height)
x, y = np.meshgrid(x, y)

# 创建三维图
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 绘制曲面图
ax.plot_surface(x, y, img_array, cmap='viridis')

# 设置坐标轴标签
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# 显示图形
plt.show()




# import cv2
# import numpy as np
# import matplotlib.pyplot as plt

# # 读取彩色图像
# image_path = 'tst.png'  # 替换为你的图像路径
# color_image = cv2.imread(image_path)

# # 将彩色图像转换为灰度图像
# gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

# # 显示灰度图像
# plt.figure(figsize=(10, 5))
# plt.subplot(1, 2, 1)
# plt.title('Gray Image')
# plt.imshow(gray_image, cmap='gray')  # 使用灰度颜色映射显示灰度图像
# plt.axis('off')  # 关闭坐标轴

# # 显示灰度值空间分布（灰度直方图）
# plt.subplot(1, 2, 2)
# plt.title('Gray Histogram')
# plt.hist(gray_image.ravel(), bins=256, range=[0, 256], color='gray')
# plt.xlabel('Pixel Value')
# plt.ylabel('Frequency')

# plt.tight_layout()  # 调整子图布局以防止重叠
# plt.show()

