import cv2
import numpy as np
from PIL import Image


def draw_points(im, points, size=3):
    im_arr = np.array(im)
    points = np.array(points, dtype=int)

    for p in points:
        cv2.circle(im_arr, p, size, (0, 255, 255), -1)

    return Image.fromarray(im_arr)