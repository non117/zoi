# -*- coding: utf-8 -*-
from pathlib import Path

def main(root):
    root = Path(root)
    for i, dir_ in enumerate(root.iterdir()):
        pages = list(range(12, 120))
        if i == 5:
            pages += [120]
        l = []
        prevname = list(dir_.iterdir())[0].name
        for png in dir_.iterdir():
            comic, page, n = list(filter(bool, png.name.split('_')))
            page = int(page)
            n = int(n.replace('.png',''))
            if not page in pages:
                continue
            if n == 0:
                print(prevname, l)
                l = [0]
            else:
                l.append(n)
            prevname = png.name

if __name__ == '__main__':
    root = 'data'
    main(root)
