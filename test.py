# -*- coding: utf-8 -*-
import cv2
import numpy as np
import pylab as plt

from pathlib import Path

def show(im):
    b, g, r = cv2.split(im)
    im = cv2.merge([r,g,b])
    plt.imshow(im)
    plt.show()

def diff(seq):
    prev = seq[0]
    new_seq = []
    for x in seq:
        new_seq.append(prev - x)
        prev = x
    return new_seq

def determine2(seq):
    th = max(seq) / 2
    print(th)
    flag = False
    n = len(seq)
    xs = []
    for i in range(n-1):
        if seq[i] > th:
            flag = True
        if flag and seq[i] >= 0 and seq[i+1] < 0:
            xs.append(i+1)
            flag = False
    return xs

def clean(seq):
    xs = []
    temp = []
    prev = seq[0]
    for x in seq[1:]:
        if x - prev < 10:
            temp.append(x)
        else:
            if temp:
                xs.append(int(np.average(temp)))
                temp = []
            else:
                xs.append(x)
        prev = x
    xs.append(int(np.average(temp)))
    return xs

def determine(seq):
    xs = []
    th = max(seq) / 2
    for i, x in enumerate(seq):
        if np.abs(x) > th:
            xs.append(i)
    return clean(xs)

def normalize(seq):
    offset = seq[0]
    return [x - offset for x in seq]

nxs = [0,0,0,0]
nys = [0,0,0,0,0,0,0,0]
n = 0
x1 = []
x2 = []
y1 = []

def store(xs, ys):
    global nxs, nys, n, x1, x2, y1
    y1.append(ys[-1])
    if n == 0:
        x1.append(xs[0])
    elif np.abs(np.average(x1) - xs[0]) < 20:
        x1.append(xs[0])
    else:
        x2.append(xs[0])
    xs = normalize(xs)
    ys = normalize(ys)
    nxs = map(sum, zip(xs, nxs))
    nys = map(sum, zip(ys, nys))
    n += 1

def nearest(xs, ys):
    global x1, x2, y1
    near = 5000
    nx, ny = 0, 0
    for x_ in x1 + x2:
        for x in xs:
            diff = np.abs(x - x_)
            if diff < near:
                near = diff
                nx = x
    near = 5000
    for y_ in y1:
        for y in ys:
            diff = np.abs(y - y_)
            if diff < near:
                near = diff
                ny = y
    return nx, ny

def estimate(xs, ys, h):
    global nxs, nys, n
    if len(ys) == 8 and len(xs) == 4:
        store(xs, ys)
        return xs, ys
    elif n != 0:
        #print(xs[0], ys[-1])
        xs_ = map(lambda x:x/n, nxs)
        ys_ = map(lambda y:y/n, nys)
        yoffset = ys_[7] - ys_[0]
        x0, y_ = nearest(xs, ys)
        y0 = y_ - yoffset
        est_xs = [x0 + x for x in xs_]
        est_ys = [y0 + y for y in ys_]
        return est_xs, est_ys
    else:
        return xs, ys

def crop(path):
    filename = path.as_posix()
    im_in = cv2.imread(filename)
    h, w, _ = im_in.shape
    im_out = cv2.imread(filename)
    im_gray = cv2.cvtColor(im_in, cv2.COLOR_BGR2GRAY)
    im = cv2.GaussianBlur(im_gray, (5,5),0)

    yoko = (im.sum(0)/w).tolist()
    tate = (im.sum(1)/h).tolist()
    xs = determine(diff(yoko))
    ys = determine(diff(tate))
    #print(xs)
    #print(ys)
    #plt.plot(diff(yoko))
    #plt.plot(diff(tate))
    #plt.show()
    xs, ys = estimate(xs, ys, h)
    
    for x in xs:
        cv2.line(im_out, (x, 0), (x, h), (0,0,255), 2)
    for y in ys:
        cv2.line(im_out, (0, y), (w, y), (0,0,255), 2)
    show(im_out)
    
def main():
    for i in range(2,14):
        filename = 'test{0}.png'.format(i)
        path = Path(filename)
        crop(path)

if __name__ == '__main__':
    main()

