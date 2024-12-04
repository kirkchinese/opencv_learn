# coding=gb2312

import cv2
import numpy as np
import glob

"""
����궨
"""

# �����̸�ǵ�
# ����Ѱ�������ؽǵ�Ĳ��������õ�ֹͣ׼�������ѭ������30������������0.001
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001) # ��ֵ
#���̸�ģ����
w = 11   # 10 - 1
h = 8   # 7  - 1
# ��������ϵ�е����̸��,����(0,0,0), (1,0,0), (2,0,0) ....,(8,5,0)��ȥ��Z���꣬��Ϊ��ά����
objp = np.zeros((w*h,3), np.float32)
objp[:,:2] = np.mgrid[0:w,0:h].T.reshape(-1,2)
objp = objp*18.1  # 18.1 mm

# �������̸�ǵ�����������ͼ�������
objpoints = [] # ����������ϵ�е���ά��
imgpoints = [] # ��ͼ��ƽ��Ķ�ά��
#����pic�ļ��������е�jpgͼ��
#images = glob.glob('./*.jpg')  #   �����ʮ��������ͼƬ����Ŀ¼

images = glob.glob('img\cut\*.bmp')  #   �����ʮ��������ͼƬ����Ŀ¼

i=0
for fname in images:

    img = cv2.imread(fname)
    # ��ȡ�������ĵ�
    #��ȡͼ��ĳ���
    h1, w1 = img.shape[0], img.shape[1]
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    u, v = img.shape[:2]
    # �ҵ����̸�ǵ�
    ret, corners = cv2.findChessboardCorners(gray, (w,h),None)
    # ����ҵ��㹻��ԣ�����洢����
    if ret == True:
        print("i:", i)
        i = i+1
        # ��ԭ�ǵ�Ļ�����Ѱ�������ؽǵ�
        cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        #׷�ӽ���������ά���ƽ���ά����
        objpoints.append(objp)
        imgpoints.append(corners)
        # ���ǵ���ͼ������ʾ
        cv2.drawChessboardCorners(img, (w,h), corners, ret)
        cv2.namedWindow('findCorners', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('findCorners', 640, 480)
        cv2.imshow('findCorners',img)
        cv2.imwrite("img\cut\imgsave\pic"+str(i-1)+".png",img)
        cv2.waitKey(200)
cv2.destroyAllWindows()
#%% �궨
print('���ڼ���')
#�궨
ret, mtx, dist, rvecs, tvecs = \
    cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)


print("ret:",ret  )
print("mtx:\n",mtx)      # �ڲ�������
print("dist����ֵ:\n",dist   )   # ����ϵ��   distortion cofficients = (k_1,k_2,p_1,p_2,k_3)
print("rvecs��ת�����������:\n",rvecs)   # ��ת����  # �����
print("tvecsƽ�ƣ����������:\n",tvecs  )  # ƽ������  # �����
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (u, v), 0, (u, v))
print('newcameramtx���',newcameramtx)
#�������
camera=cv2.VideoCapture(1)
while True:
    (grabbed,frame)=camera.read()
    h1, w1 = frame.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (u, v), 0, (u, v))
    # ��������
    dst1 = cv2.undistort(frame, mtx, dist, None, newcameramtx)
    #dst2 = cv2.undistort(frame, mtx, dist, None, newcameramtx)
    mapx,mapy=cv2.initUndistortRectifyMap(mtx,dist,None,newcameramtx,(w1,h1),5)
    dst2=cv2.remap(frame,mapx,mapy,cv2.INTER_LINEAR)
    # �ü�ͼ��������������Ժ��ͼƬ
    x, y, w1, h1 = roi
    dst1 = dst1[y:y + h1, x:x + w1]

    #cv2.imshow('frame',dst2)
    #cv2.imshow('dst1',dst1)
    cv2.imshow('dst2', dst2)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # ��q����һ��ͼƬ
        cv2.imwrite("../u4/frame.jpg", dst1)
        break

camera.release()
cv2.destroyAllWindows()


