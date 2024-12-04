import cv2 as cv 
import numpy as np
import glob
import bitsize
import matplotlib.pyplot as plt

core_w = np.array([-1,0,1,
                   -1,0,1,
                   -1,0,1])
core_h = np.array([-1,-1,-1,
                   0,0,0,
                   1,1,1])

core_laplace = np.array([0,1,0,
                         1,4,1,
                         0,1,0])


def main():
    images = glob.glob('img\*.png')
    i=0
    for img_path in images:
        print(f"正在使用Sobel算子{core_h},{core_w}处理第{i}张图片，图片路径为{img_path}")
        img = cv.imread(img_path)
        img = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
        result_w = cv.filter2D(img, -1, core_w) # 横向滤波
        result_h = cv.filter2D(img, -1, core_h) # 纵向滤波
        result = result_w + result_h
        plt.subplot(2,2,1),plt.imshow(img,cmap='gray'),plt.title('img')
        plt.subplot(2,2,2),plt.imshow(result_w,cmap='gray'),plt.title('result_w')
        plt.subplot(2,2,3),plt.imshow(result_h,cmap='gray'),plt.title('result_h')
        plt.subplot(2,2,4),plt.imshow(result,cmap='gray'),plt.title('result')
        plt.show()
        i+=1
    
    i=0
    for img_path in images:
        print(f"正在使用Laplace算子{core_laplace}处理第{i}张图片，图片路径为{img_path}")
        img = cv.imread(img_path)
        img = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
        result = cv.filter2D(img, -1, core_laplace)
        plt.subplot(1,2,1),plt.imshow(img,cmap='gray'),plt.title('img')
        plt.subplot(1,2,2),plt.imshow(result,cmap='gray'),plt.title('result')
        plt.show()
        i+=1
    i=0
    for img_path in images:
        print(f"正在使用Canny算法处理第{i}张图片，图片路径为{img_path}")
        img = cv.imread(img_path)
        img = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
        result = cv.Canny(img, 100, 200)
        plt.subplot(1,2,1),plt.imshow(img,cmap='gray'),plt.title('img')
        plt.subplot(1,2,2),plt.imshow(result,cmap='gray'),plt.title('result')
        plt.show()
        i+=1


if __name__ == '__main__':
    main()