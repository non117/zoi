# -*- coding: utf-8 -*-
import sys
from PIL import Image

def main():
    filename = 'yuyushiki2/_'
    angle = 0
    a, b = filename.split('.')
    Image.open(filename).rotate(angle).show()
    #Image.open(filename).rotate(angle).save(a + '_' + b)
