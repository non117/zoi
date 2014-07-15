# -*- coding: utf-8 -*-
import cv2
import numpy as np
import pylab as plt
from pathlib import Path

#TODO OverflowError: signed integer is less than minimum がなおらない

def show(im):
    # debug用
    b, g, r = cv2.split(im)
    im = cv2.merge([r,g,b])
    plt.imshow(im)
    plt.show()

def rotate(im, angle):
    h, w, _ = im.shape
    mat = cv2.getRotationMatrix2D((h/2, w/2), angle, 1)
    return cv2.warpAffine(im, mat, (w, h))

def point_line(im_edge,hough):
    lines = cv2.HoughLinesP(im_edge,1,np.pi/180,hough)
    h = lines[0][:][:].shape[0]
    x1 = np.empty(h,dtype=np.int)
    y1 = np.empty(h,dtype=np.int)
    x2 = np.empty(h,dtype=np.int)
    y2 = np.empty(h,dtype=np.int)
    # カードの輪郭のみ描画
    for i in range(107):
        x1[i] = lines[0][i][0]
        y1[i] = lines[0][i][1]
        x2[i] = lines[0][i][2]
        y2[i] = lines[0][i][3]
    # 検出したすべての直線の終点座標を返す
    return x1,y1,x2,y2

def drow_line(im,x1,y1,x2,y2,th):
    # 直線を描画
    for i in range(x1.size):
        # 検出した直線の長さがthpixcel以上なら,直線を描く
        if x1[i] > 10e5 or x2[i] > 10e5 or y1[i] > 10e5 or y2[i] > 10e5:
            continue
        if abs(x1[i]-x2[i])>th or abs(y1[i]-y2[i])>th:
            cv2.line(im,(x1[i],y1[i]),(x2[i],y2[i]),(0,0,255), 5, 10)
 
    return im

def houghP(path):
    filename = path.as_posix()
    print(filename)
    im_in = cv2.imread(filename)
    im_out = cv2.imread(filename)
    im_gray = cv2.cvtColor(im_in,cv2.COLOR_BGR2GRAY)
    # ノイズ除去
    #im_gray = cv2.medianBlur(im_gray, 5)
    im_gray = cv2.GaussianBlur(im_gray, (5,5), 0)
    # グレースケール画像を2値化
    im_th = cv2.adaptiveThreshold(im_gray,50,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
    # 2値化画像からエッジを検出
    im_edge = cv2.Canny(im_th,50,150,apertureSize = 3) 
    x1,y1,x2,y2 = point_line(im_edge,hough=30)
    im_out = drow_line(im_out,x1,y1,x2,y2,th=20)
    cv2.imwrite((path.parent/'hough{0}'.format(path.name)).as_posix(), im_out) 

def hough(path):
    filename = path.as_posix()
    print(filename)
    im_in = cv2.imread(filename)
    im_out = cv2.imread(filename)
    im_gray = cv2.cvtColor(im_in,cv2.COLOR_BGR2GRAY)
    # ノイズ除去
    #im_gray = cv2.medianBlur(im_gray, 5)
    im_gray = cv2.GaussianBlur(im_gray, (5,5), 0)
    # グレースケール画像を2値化
    im_th = cv2.adaptiveThreshold(im_gray,50,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
    # 2値化画像からエッジを検出
    im_edge = cv2.Canny(im_th,50,150,apertureSize = 3)
    # エッジ画像から直線の検出
    lines = cv2.HoughLines(im_edge,1,np.pi/180,300)
    thetas = []
    # 直線の描画
    if lines is None:
        return
    for rho,theta in lines[0]:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))
        if rho > 2200 and np.abs(90 - theta*180/np.pi) < 3:
            thetas.append(theta*180/np.pi)
        cv2.line(im_out,(x1,y1),(x2,y2),(0,0,255),2)
    
    cv2.imwrite((path.parent/'hough{0}'.format(path.name)).as_posix(), im_out)
    degree = np.average(thetas) - 90
    if np.abs(degree) > 0.1:
        print(degree, thetas)
        im = rotate(im_in, degree)

def main():
    directory = './yuyushiki2'
    paths = Path(directory).resolve()
    for path in paths.iterdir():
        if path.suffix == '.png': 
            houghP(path)

if __name__ == '__main__':
    main()

