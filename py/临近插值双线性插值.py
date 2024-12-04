"""
缩放、近邻插值、 双线性插值
"""

# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
# import glob

# def resize_image(image_path, new_width, new_height, interpolation_method):
#     # 读取图像
#     img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

#     # 计算缩放比例
#     scale_x = new_width / img.shape[1]
#     scale_y = new_height / img.shape[0]

#     # 创建插值后的图像
#     new_img = np.zeros((new_height, new_width), dtype=np.uint8)

#     # 对每个像素进行插值
#     for i in range(new_height):
#         for j in range(new_width):
#             # 计算原始图像中对应的像素位置
#             x = j / scale_x
#             y = i / scale_y

#             # 计算原始图像中对应的像素位置所在的四个已知点
#             x1 = int(x)
#             y1 = int(y)
#             x2 = min(x1 + 1, img.shape[1] - 1)
#             y2 = min(y1 + 1, img.shape[0] - 1)

#             # 计算原始图像中对应的像素位置与四个已知点的距离
#             dx1 = x - x1
#             dy1 = y - y1
#             dx2 = 1 - dx1
#             dy2 = 1 - dy1

#             # 计算插值后的像素值
#             if interpolation_method == 'bilinear':
#                 new_img[i, j] = (dx2 * dy2 * img[y1, x1] +
#                                  dx1 * dy2 * img[y1, x2] +
#                                  dx2 * dy1 * img[y2, x1] +
#                                  dx1 * dy1 * img[y2, x2])
#             elif interpolation_method == 'nearest':
#                 new_img[i, j] = img[y1, x1]

#     # 显示插值后的图像
#     if interpolation_method == 'bilinear':
#         plt.imshow(new_img, cmap='gray')
#         plt.axis('off')
#         plt.show()
#     elif interpolation_method == 'nearest':
#         plt.imshow(new_img, cmap='gray')
#         plt.axis('off')
#         plt.show()


# if __name__ == '__main__':

#     files = glob.glob('./img/*.png')
#     for file in files:
#         print(file)
#         # 使用双线性插值
#         resize_image(file, 1920, 1080, 'bilinear')

#         # 使用临近插值
#         resize_image(file, 1920, 1080, 'nearest')



"""
缩放、近邻插值、双线性插值 (GPU加速版本)
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
import cupy as cp  # GPU加速库

def resize_image_gpu(image_path, new_width, new_height, interpolation_method):
    # 读取图像
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img_gpu = cp.asarray(img)

    # 计算缩放比例
    scale_x = img.shape[1] / new_width
    scale_y = img.shape[0] / new_height

    # 创建坐标网格
    x = cp.arange(new_width)
    y = cp.arange(new_height)
    X, Y = cp.meshgrid(x, y)

    # 计算原始图像中对应的像素位置
    src_x = X * scale_x
    src_y = Y * scale_y

    # 计算原始图像中对应的像素位置所在的四个已知点
    x1 = cp.floor(src_x).astype(int)
    y1 = cp.floor(src_y).astype(int)
    x2 = cp.minimum(x1 + 1, img.shape[1] - 1)
    y2 = cp.minimum(y1 + 1, img.shape[0] - 1)

    # 计算插值权重
    wx = src_x - x1
    wy = src_y - y1

    if interpolation_method == 'bilinear':
        # 双线性插值
        new_img = (
            (1 - wx) * (1 - wy) * img_gpu[y1, x1] +
            wx * (1 - wy) * img_gpu[y1, x2] +
            (1 - wx) * wy * img_gpu[y2, x1] +
            wx * wy * img_gpu[y2, x2]
        ).astype(cp.uint8)
    elif interpolation_method == 'nearest':
        # 最近邻插值
        new_img = img_gpu[y1, x1].astype(cp.uint8)

    # 将结果从GPU转回CPU
    new_img = cp.asnumpy(new_img)

    # 显示插值后的图像
    plt.imshow(new_img, cmap='gray')
    plt.axis('off')
    plt.title(f'Resized using {interpolation_method} interpolation')
    plt.show()

if __name__ == '__main__':
    files = glob.glob('./img/*.png')
    for file in files:
        print(file)
        # 使用双线性插值
        resize_image_gpu(file, 4096, 3960, 'bilinear')
        # 使用最近邻插值
        resize_image_gpu(file, 1920, 1080, 'nearest')
