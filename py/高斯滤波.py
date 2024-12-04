import cv2
import numpy as np

def main():
    # 读取图像
    image = cv2.imread('img\IMG.png')
    
    if image is None:
        print("Error: Unable to load image")
        return

    # 显示原始图像
    cv2.imshow('Original Image', image)

    # 应用均值滤波
    mean_filtered = cv2.blur(image, (5, 5))
    cv2.imshow('Mean Filtered Image', mean_filtered)
    cv2.waitKey(0)

    # 将图像转换为灰度图像
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    sobel_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
    sobel = cv2.addWeighted(sobel_x, 0.5, sobel_y, 0.5, 0)
    sobel = np.uint8(np.absolute(sobel))

    # 应用Sobel锐化
    sharpened = cv2.addWeighted(gray_image, 1, sobel, -1, 0)
    cv2.imshow('Sobel Sharpened Image', sharpened)

    # 应用高斯滤波
    gaussian_filtered = cv2.GaussianBlur(image, (5, 5), 0)
    cv2.imshow('Gaussian Filtered Image', gaussian_filtered)
    cv2.waitKey(0)

    # 应用中值滤波
    median_filtered = cv2.medianBlur(image, 5)
    cv2.imshow('Median Filtered Image', median_filtered)
    cv2.waitKey(0)

    # 等待按键按下
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
