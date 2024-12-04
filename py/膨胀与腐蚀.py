
import cv2 as cv 
import numpy as np
import glob
import bitsize
import matplotlib.pyplot as plt



# 定义一个函数包装二值化函数
def get_binary(img,type_int = None, window_size=15, k=0.2,constant=2):
    if type_int is None:
        raise Exception ("No type_int specified")
    elif type_int == 0: # otsuch 
        thresholded_image, threshold = bitsize.otsu_thresholding(img)
        return [thresholded_image,threshold]
    elif type_int == 1: # 双峰值法
        thresholded_image, threshold, hist = bitsize.bimodal_thresholding(img)
        return [thresholded_image,threshold,hist]
    elif type_int == 2 :# 全局阈值法 
        thresholded_image, mean_value = bitsize.mean_thresholding(img)
        return [thresholded_image, mean_value]
    elif type_int == 3: # niblack 此函数可以指定窗口大小和k值，所以只有使用这些时我们需要上述的两个参数
        binary = bitsize.niblack_thresholding(img,window_size, k)
        return binary
    elif type_int == 4: # bernsen_thresholding
        binary = bitsize.bernsen_thresholding(img,window_size)
        return binary
    elif type_int == 5: # adaptivate_threshold
        binary = bitsize.adaptivate_threshold(img, window_size,constant)
        return binary

# 膨胀腐蚀
def get_dilation(img,core):
    h,w = img.shape
    dilated_img = np.zeros_like(img)
    core_h,core_w = core.shape
    cx, cy = core_w // 2, core_h // 2
    if cx == 1 and cy == 1:
        cx = 0
        cy = 0
    for i in range(cx,h-cx):
        for j in range(cy,w-cy):
            region = img[i-cx:i+cx+1,j-cy:j+cy+1]
            if np.any(region*core):
                dilated_img[i,j] = 255
    return dilated_img

def get_erosion(img, core):
    # 获取图像的形状
    h, w = img.shape
    eroded_img = np.zeros_like(img)
    kh, kw = core.shape
    cx, cy = kw // 2, kh // 2

    if cx == 1 and cy == 1:
        cx = 0
        cy = 0

    for i in range(cx, h - cx):
        for j in range(cy, w - cy):
            # 获取图像区域
            region = img[i - cx:i + cx + 1, j - cy:j + cy + 1]
            if np.array_equal(region,core):
                eroded_img[i, j] = 255
    return eroded_img
    


images = glob.glob('img\img1.png')
images = glob.glob('img\*.png')

# core = np.array([[0,0,1,0,0],
#                  [0,1,1,1,0],
#                  [1,1,1,1,1],
#                  [0,1,1,1,0],
#                  [0,0,1,0,0]])
# core = np.array([[0,0],
#                  [0,1]])

core = np.array([[0,1,0],
                 [1,1,1],
                 [0,1,0]])
core = np.uint8(core)
# core = np.random.randint(2,size=(3, 3))
# core = np.ones((3, 3), dtype=np.uint8)
print(core)

#
for i in images:
    # img = cv.imread(i)
    # # img_binary = get_binary(img,type_int=5)
    # img_binary = get_binary(img,type_int=1)
    # img_binary=img_binary[0]
    # dilated_img = get_dilation(img_binary,core)
    # erosion_img = cv.erode(img_binary, core, iterations=1)
    # # erosion_img = get_erosion(img_binary,core)
    # # 开运算
    # # open_img = get_dilation(get_erosion(img_binary,core),core)
    # open_img = get_dilation(cv.erode(img_binary, core, iterations=1),core)
    # #close_img = get_erosion(get_dilation(img_binary,core),core)
    # close_img = cv.erode(get_dilation(img_binary,core), core, iterations=1)
    # plt.subplot(231), plt.imshow(img_binary, cmap='gray'), plt.title('binary_img')
    # plt.subplot(232), plt.imshow(dilated_img, cmap='gray'), plt.title("dilate_img")
    # plt.subplot(233), plt.imshow(erosion_img, cmap='gray'), plt.title("erosion_img")

    # plt.subplot(234), plt.imshow(open_img, cmap='gray'), plt.title("open_img")
    # plt.subplot(235), plt.imshow(close_img, cmap='gray'), plt.title("close_img")
    # plt.show()
    img = cv.imread(i,cv.IMREAD_GRAYSCALE)
    img_binary = cv.threshold(img, 127, 255, cv.THRESH_BINARY)[1]  # 使用阈值化函数
    dilated_img = cv.dilate(img_binary, core, iterations=1)
    erosion_img = cv.erode(img_binary, core, iterations=1)

    # 开运算
    open_img = cv.dilate(erosion_img, core, iterations=1)

    # 闭运算
    close_img = cv.erode(dilated_img, core, iterations=1)

    # 顶帽运算
    top_hat = cv.morphologyEx(img_binary, cv.MORPH_TOPHAT, core)

    # 底帽运算
    black_hat = cv.morphologyEx(img_binary, cv.MORPH_BLACKHAT, core)

    plt.subplot(331), plt.imshow(img_binary, cmap='gray'), plt.title('binary_img')
    plt.subplot(332), plt.imshow(dilated_img, cmap='gray'), plt.title("dilate_img")
    plt.subplot(333), plt.imshow(erosion_img, cmap='gray'), plt.title("erosion_img")

    plt.subplot(334), plt.imshow(open_img, cmap='gray'), plt.title("open_img")
    plt.subplot(335), plt.imshow(close_img, cmap='gray'), plt.title("close_img")
    plt.subplot(336), plt.imshow(top_hat, cmap='gray'), plt.title("top_hat")
    plt.subplot(337), plt.imshow(black_hat, cmap='gray'), plt.title("black_hat")
    plt.show()


    
