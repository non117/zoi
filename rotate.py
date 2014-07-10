# -*- coding: utf-8 -*-
import cv2
import numpy as np
import pylab as plt
from pathlib import Path

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

def hough(path):
    filename = path.as_posix()
    print(filename)
    im_in = cv2.imread(filename)
    im_out = cv2.imread(filename)
    im_gray = cv2.cvtColor(im_in,cv2.COLOR_BGR2GRAY)
    # ノイズ除去
    im_gray = cv2.medianBlur(im_gray, 5)
    # グレースケール画像を2値化
    im_th = cv2.adaptiveThreshold(im_gray,50,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
    # 2値化画像からエッジを検出
    im_edge = cv2.Canny(im_th,50,150,apertureSize = 3)
    # エッジ画像から直線の検出
    lines = cv2.HoughLines(im_edge,1,np.pi/180,300)
    thetas = []
    # 直線の描画
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
            #print(rho, theta*180/np.pi)
            thetas.append(theta*180/np.pi)
        cv2.line(im_out,(x1,y1),(x2,y2),(0,0,255),2)
    #show(im_out)
    degree = np.average(thetas) - 90
    print(degree)
    if degree > 0.5:
        im = rotate(im_in, degree)
        cv2.imwrite("hoge.png", im)

def main():
    directory = './test'
    paths = Path(directory).resolve()
    for path in paths.iterdir():
        if path.suffix == '.png': 
            hough(path)

if __name__ == '__main__':
    main()

