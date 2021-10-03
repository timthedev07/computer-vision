import os
from sys import argv
import cv2


def main():
    if len(argv) < 3:
        print("Usage: python resize.py <path> <percentage>")
    path = argv[1]
    scalePercentage = float(argv[2])
    img = cv2.imread(path)
    width = int(img.shape[1] * scalePercentage / 100)
    height = int(img.shape[0] * scalePercentage / 100)
    dimensions = (width, height)
    resized = cv2.resize(img, dimensions, interpolation=cv2.INTER_AREA)
    os.remove(path)

    cv2.imwrite(path, resized)


if __name__ == "__main__":
    main()
