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

def clean(seq):
    xs = []
    temp = []
    prev = seq[0]
    for x in seq[1:]:
        if x - prev < 20:
            temp.append(x)
        else:
            if temp:
                xs.append(int(np.average(temp)))
                temp = []
            else:
                xs.append(x)
        prev = x
    if temp:
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
def store(xs, ys):
    global nxs, nys, n
    xs = normalize(xs)
    ys = normalize(ys)
    nxs = map(sum, zip(xs, nxs))
    nys = map(sum, zip(ys, nys))
    n += 1

def estimate(xs, ys, h):
    global nxs, nys, n
    if len(ys) == 8 and len(xs) == 4:
        store(xs, ys)
        print 10
        return xs, ys
    elif n != 0:
        print 20
        xs_ = map(lambda x:x/n, nxs)
        ys_ = map(lambda y:y/n, nys)
        print ys_
        yoffset = ys_[7] - ys_[0]
        x0 = xs[0]
        y0 = ys[-1] - yoffset
        est_xs = [x0 + x for x in xs_]
        est_ys = [y0 + y for y in ys_]
        return est_xs, est_ys
    else:
        return xs, ys

def crop(path):
    filename = path.as_posix()
    print(filename)
    im_in = cv2.imread(filename)
    h, w, _ = im_in.shape
    im_out = cv2.imread(filename)
    im_gray = cv2.cvtColor(im_in, cv2.COLOR_BGR2GRAY)
    im = cv2.GaussianBlur(im_gray, (5,5), 0)
    
    yoko = (im.sum(0)/w).tolist()
    tate = (im.sum(1)/h).tolist()
    xs = determine(diff(yoko))
    ys = determine(diff(tate))

    #plt.clf()
    #plt.plot(diff(yoko))
    #plt.plot(diff(tate))
    #plt.savefig((path.parent/'graph{0}'.format(path.name)).as_posix())

    xs, ys = estimate(xs, ys, h)
    
    for x in xs:
        cv2.line(im_out, (x, 0), (x, h), (0,0,255), 2)
    for y in ys:
        cv2.line(im_out, (0, y), (w, y), (0,0,255), 2)
    new_path = path.parent / 'test{0}'.format(path.name)
    cv2.imwrite(new_path.as_posix(), im_out)
    return
    
    name, ext = filename.split('.')
    new_dir = path.parent / 'output' 
    new_suffix = '{0}_{1}'.format(path.parent.name, name)

    cnt = 0
    for i in reversed(range(len(xs) - 1)):
        for j in range(len(ys) -1):
            x1, x2 = xs[i], xs[i+1]
            y1, y2 = ys[j], ys[j+1]
            if( 100000 < (x2 - x1) * (y2 - y1)):
                im_trim = im_out[y1:y2, x1:x2]
                new_path = new_dir / '{0}_{1}{2}'.format(new_suffix, cnt, ext)
                cv2.imwrite(new_path.as_posix(), im_trim)
                cnt += 1
    
    #haba = int(np.average([xs[1] - xs[0], xs[3] - xs[2]]))
    #takasa = int(np.average([ys[1] - ys[0], ys[3] - ys[2], ys[5] - ys[4], ys[7] - ys[6]]))

def main():
    dirname = 'yuyu6'
    path = Path(dirname)
    try:
        (path / 'output').mkdir(mode=0o755)
    except OSError:
        pass
    paths = [p for p in path.iterdir()]
    for p in paths:
        if(p.suffix in ('.png', '.jpg')):
            crop(p)

if __name__ == '__main__':
    main()

