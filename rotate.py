# -*- coding: utf-8 -*-
import cv2
import numpy as np
import pylab as plt
from PIL import Image

def show(im):
    # debug用
    b, g, r = cv2.split(im)
    im = cv2.merge([r,g,b])
    plt.imshow(im)
    plt.show()

def hough(filename):
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
        if rho > 2200:
            print(rho, theta*180/np.pi)
            thetas.append(theta*180/np.pi)
        cv2.line(im_out,(x1,y1),(x2,y2),(0,0,255),2)
    show(im_out)
    degree = np.average(thetas) - 90
    print(degree)
    Image.open(filename).rotate(degree).show()

def main():
    filename = 'test/test17.png'
    hough(filename)

if __name__ == '__main__':
    main()

