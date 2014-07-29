import cv2
import numpy as np
from pathlib import Path

from lib import LUT

lut = LUT(1/1.6)
def main():
    directory = './yuyushiki3'
    paths = Path(directory).resolve()
    for path in paths.iterdir():
        if path.suffix == '.png':
            print(path.as_posix())
            im = cv2.imread(path.as_posix())
            cv2.normalize(im, im, 0, 255, cv2.NORM_MINMAX)
            im_out = cv2.LUT(im, lut)
            cv2.imwrite(path.as_posix(), im_out)

main()
