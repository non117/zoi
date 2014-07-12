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

def diff(seq):
    # 微分
    prev = seq[0]
    new_seq = []
    for x in seq:
        new_seq.append(prev - x)
        prev = x
    return new_seq

def rotate(im, angle):
    # 大回転
    h, w = im.shape[0], im.shape[1]
    mat = cv2.getRotationMatrix2D((h/2, w/2), angle, 1)
    return cv2.warpAffine(im, mat, (w, h)) 

def hough(im_gray, threshold=300):
    # ノイズ除去
    im_gray = cv2.medianBlur(im_gray, 5)
    # グレースケール画像を2値化
    im_th = cv2.adaptiveThreshold(im_gray,50,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
    # 2値化画像からエッジを検出
    im_edge = cv2.Canny(im_th,50,150,apertureSize = 3)
    # エッジ画像から直線の検出
    lines = cv2.HoughLines(im_edge,1,np.pi/180,threshold)
    return lines

def clean(seq):
    # 連続した値から代表値を抽出する
    xs = []
    temp = []
    prev = seq[0]
    for x in seq:
        if x - prev < 20:
            temp.append(x)
        else:
            if temp:
                xs.append(int(np.average(temp)))
                temp = [x]
            else:
                xs.append(x)
        prev = x
    if temp:
        xs.append(int(np.average(temp)))
    return xs

def determine(seq):
    # ヒストグラムの微分値から濃淡変化の大きい行, 列を探す
    xs = []
    th = max(seq) / 2
    for i, x in enumerate(seq):
        if np.abs(x) > th:
            xs.append(i)
    return clean(xs)

def normalize(seq):
    # 紙面座標系からコマの座標系に正規化
    offset = seq[0]
    return [x - offset for x in seq]

nxs = [0,0,0,0]
nys = [0,0,0,0,0,0,0,0]
n = 0
x1 = []
x2 = []
y1 = []

def store(xs, ys):
    # グローバル変数にコマの座標を貯める
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
    # x, yに最も近いxs, ys内の座標を探す
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
    # コマ座標が抽出できていればそのまま、なければ推定する
    global nxs, nys, n
    if len(ys) == 8 and len(xs) == 4:
    #    store(xs, ys)
        return xs, ys
    elif n != 0:
        #print('estimated.')
        xs_ = map(lambda x:x/n, nxs)
        ys_ = map(lambda y:y/n, nys)
        yoffset = ys_[7] - ys_[0]
        x0, y_ = nearest(xs, ys)
        y0 = y_ - yoffset
        est_xs = [int(x0 + x) for x in xs_]
        est_ys = [int(y0 + y) for y in ys_]
        return est_xs, est_ys
    else:
        return xs, ys

def detect_angle(im_gray):
    global y1
    if y1:
        y = np.average(y1)
    else:
        y = 2267
    lines = hough(im_gray)
    if lines is None:
        return 0
    degs = []
    for rho, theta in lines[0]:
        degree = theta*180/np.pi
        if y - 100 < rho < y + 100 and np.abs(90 - degree) < 3:
            degs.append(degree)
    rot = np.average(degs) - 90
    if np.abs(rot) > 0.1:
        print(rot)
        return rot
    return 0

def statistics(path, exception=False):
    filename = path.as_posix()
    print(filename)
    im_in = cv2.imread(filename)
    h, w, _ = im_in.shape
    im_gray = cv2.cvtColor(im_in, cv2.COLOR_BGR2GRAY)
    im = cv2.GaussianBlur(im_gray, (5,5), 0)
    
    if not exception:
        angle = detect_angle(im_gray)
        if angle != 0:
            im = rotate(im, angle)

    yoko = (im.sum(0)/w).tolist()
    tate = (im.sum(1)/h).tolist()
    xs = determine(diff(yoko))
    ys = determine(diff(tate))

    if not exception and angle == 0 and len(ys) == 8 and len(xs) == 4:
        #xs, ys = estimate(xs, ys, h)
        store(xs, ys) 

def crop(path, exception=False, firstpath=False):
    filename = path.as_posix()
    print(filename)
    im_in = cv2.imread(filename)
    h, w, _ = im_in.shape
    im_gray = cv2.cvtColor(im_in, cv2.COLOR_BGR2GRAY)
    im_out = cv2.imread(filename)
    im = cv2.GaussianBlur(im_gray, (5,5), 0)

    if not exception:
        angle = detect_angle(im_gray)
        if angle != 0:
            im = rotate(im, angle)
            im_out = rotate(im_out, angle)

    yoko = (im.sum(0)/w).tolist()
    tate = (im.sum(1)/h).tolist()
    xs = determine(diff(yoko))
    ys = determine(diff(tate))

    if not exception:
        xs, ys = estimate(xs, ys, h)

    # debug
    DEBUG = True
    #DEBUG = False
    if DEBUG:
        for x in xs:
            cv2.line(im_out, (x, 0), (x, h), (0,0,255), 2)
        for y in ys:
            cv2.line(im_out, (0, y), (w, y), (0,0,255), 2)
        new_path = path.parent / 'test{0}'.format(path.name)
        cv2.imwrite(new_path.as_posix(), im_out)
        return

    name, ext = path.stem, path.suffix
    new_dir = path.parent / 'output' 
    new_suffix = '{0}_{1}'.format(path.parent.name, name)

    cnt = 0
    for i in reversed(range(len(xs) - 1)):
        for j in range(len(ys) -1):
            x1, x2 = xs[i], xs[i+1]
            y1, y2 = ys[j], ys[j+1]
            if 50000 < (x2 - x1) * (y2 - y1) and x2 < w and y2 < h:
                im_trim = im_out[y1:y2, x1:x2]
                new_path = new_dir / '{0}_{1}{2}'.format(new_suffix, cnt, ext)
                cv2.imwrite(new_path.as_posix(), im_trim)
                cnt += 1

def main():
    dirname = 'yuyushiki2'
    exception_pages = range(1,12) + [] + range(120,127)
    errors = [17]
    path = Path(dirname)
    try:
        (path / 'output').mkdir(mode=0o755)
    except OSError:
        pass
    paths = [p for p in path.iterdir()]
    for p in paths:
        if(p.suffix in ('.png', '.jpg')):
            no = int(p.stem.replace('_',''))
            if no in exception_pages:
                statistics(p, exception=True)
            else:
                statistics(p)
    for p in paths:
        if(p.suffix in ('.png', '.jpg')):
            no = int(p.stem.replace('_',''))
            #if no not in errors:
            #    continue
            if no in exception_pages:
                crop(p, exception=True)
            else:
                crop(p)


if __name__ == '__main__':
    main()

