import cv2
from pathlib import Path

def main():
    directory = 'newgame'
    paths = Path(directory).resolve()
    for path in paths.iterdir():
        if path.suffix == '.png':
            im = cv2.imread(path.as_posix())
            cv2.imwrite(path.as_posix() ,cv2.bilateralFilter(im, 0, 32, 2))

if __name__ == '__main__':
    main()

