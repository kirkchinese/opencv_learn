import cv2
import numpy as np
import matplotlib.pyplot as plt
"""
二值化练习
"""
def otsu_thresholding(image):
    # 将图像转换为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 计算直方图
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    
    total_pixels = gray.size
    current_max, threshold = 0, 0
    sum_total, sum_foreground = 0, 0
    
    # 计算全图的灰度值总和
    for i in range(256):
        sum_total += i * hist[i][0]

    weight_background, weight_foreground = 0, 0
    
    for i in range(256):
        weight_background += hist[i][0]  # 背景权重
        if weight_background == 0:
            continue
        
        weight_foreground = total_pixels - weight_background  # 前景权重
        if weight_foreground == 0:
            break
        
        sum_foreground += i * hist[i][0]  # 前景灰度值总和

        # 计算均值
        mean_background = sum_foreground / weight_background
        mean_foreground = (sum_total - sum_foreground) / weight_foreground
        
        # 计算类间方差
        between_class_variance = weight_background * weight_foreground * (mean_background - mean_foreground) ** 2
        
        # 找到最大类间方差和对应的阈值
        if between_class_variance > current_max:
            current_max = between_class_variance
            threshold = i
    
    # 使用最佳阈值进行二值化处理
    _, thresholded_image = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    
    return thresholded_image, threshold


def bimodal_thresholding(image):
    # 将图像转换为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 计算直方图
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    
    # 找到直方图中的峰值
    peaks = np.where((hist[:-2] < hist[1:-1]) & (hist[1:-1] > hist[2:]))[0]
    
    if len(peaks) < 2:
        raise ValueError("直方图中没有足够的峰值")
    
    peaks=np.sort(peaks)

    # 选择前两个峰值
    peak1, peak2 = peaks[-2], peaks[1]
    
    # 计算阈值（取两个峰值的中点）
    threshold = (peak1 + peak2) // 2
    
    # 使用该阈值进行二值化处理
    _, thresholded_image = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    
    return thresholded_image, threshold, hist

def mean_thresholding(image):
    # 将图像转换为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 计算图像的平均灰度值
    mean_value = np.mean(gray)
    
    # 使用平均值进行二值化处理
    _, thresholded_image = cv2.threshold(gray, mean_value, 255, cv2.THRESH_BINARY)
    
    return thresholded_image, mean_value



def niblack_thresholding(image, window_size=15, k=0.2):
    # 将图像转换为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 计算局部均值和标准差
    mean = cv2.boxFilter(gray, ddepth=-1, ksize=(window_size, window_size))
    sqr_mean = cv2.boxFilter(gray**2, ddepth=-1, ksize=(window_size, window_size))
    std = np.sqrt(sqr_mean - mean**2)

    # 计算局部阈值
    threshold = mean - k * std

    # 二值化处理
    binary_image = (gray > threshold).astype(np.uint8) * 255

    return binary_image


def bernsen_thresholding(image, window_size=4):
    # 将图像转换为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 创建输出图像
    output = np.zeros_like(gray)

    # 定义半窗口大小
    half_window = window_size // 2

    # 遍历图像的每个像素
    for i in range(half_window, gray.shape[0] - half_window):
        for j in range(half_window, gray.shape[1] - half_window):
            # 提取局部区域
            local_region = gray[i-half_window:i+half_window+1, j-half_window:j+half_window+1]
            local_max = np.max(local_region)
            local_min = np.min(local_region)

            # 计算局部阈值
            threshold = (local_max + local_min) / 2

            # 应用阈值
            output[i, j] = 255 if gray[i, j] >= threshold else 0

    return output


def adaptivate_threshold(image,windows_size=5,constant=2):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    adaptive_threshold = cv2.adaptiveThreshold(
    gray,                # 输入图像
    255,                # 最大值
    cv2.ADAPTIVE_THRESH_MEAN_C,  # 自适应方法
    cv2.THRESH_BINARY,  # 阈值类型
    windows_size,                 # 区域大小 (奇数)
    constant                   # 常数
    )
    return adaptive_threshold


# # 读取图像
# image = cv2.imread('img/img.png')  

# # 使用Otsu算法进行阈值处理
# result_image, best_threshold = otsu_thresholding(image)

# # 显示结果
# plt.subplot(121), plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), plt.title('Original Image')
# plt.subplot(122), plt.imshow(result_image, cmap='gray'), plt.title(f'Otsu Threshold: {best_threshold}')
# plt.show()

# # 使用直方图双峰法进行阈值处理
# result_image, best_threshold, hist = bimodal_thresholding(image)

# # 显示结果
# plt.subplot(121), plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), plt.title('Original Image')
# plt.subplot(122), plt.imshow(result_image, cmap='gray'), plt.title(f'Threshold: {best_threshold}')
# plt.figure()
# plt.plot(hist)
# plt.title('Histogram')
# plt.xlabel('Pixel Value')
# plt.ylabel('Frequency')
# plt.show()

# result_image, mean_value = mean_thresholding(image)

# # 显示结果
# plt.subplot(121), plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), plt.title('Original Image')
# plt.subplot(122), plt.imshow(result_image, cmap='gray'), plt.title(f'Mean Threshold: {mean_value:.2f}')
# plt.show()



# result_image = niblack_thresholding(image)

# # 显示结果
# plt.subplot(121), plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), plt.title('Original Image')
# plt.subplot(122), plt.imshow(result_image, cmap='gray'), plt.title('Niblack Thresholding')
# plt.show()


# result_image = bernsen_thresholding(image)

# # 显示结果
# plt.subplot(121), plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), plt.title('Original Image')
# plt.subplot(122), plt.imshow(result_image, cmap='gray'), plt.title('Bernsen Thresholding')
# plt.show()


# adaptivate_threshold_img = adaptivate_threshold(image)

# plt.subplot(121), plt.imshow(cv2.cvtColor(image,cv2.COLOR_BGR2RGB)), plt.title('Original Image')
# plt.subplot(122), plt.imshow(adaptivate_threshold_img, cmap='gray'), plt.title('Adaptive Thresholding')
# plt.show()
