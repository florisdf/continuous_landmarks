import numpy as np
from PIL import Image
from torchvision.transforms import functional as F, Resize, Compose

from ..utils.face_alignment import align_face


class AlignImageWithPoints:
    def __init__(self, get_eyes_mouth):
        self.get_eyes_mouth = get_eyes_mouth

    def __call__(self, im: Image, points: np.ndarray):
        e0, e1, m0, m1 = self.get_eyes_mouth(points)
        im_arr, points = align_face(np.array(im), points, e0, e1, m0, m1)
        im = Image.fromarray(im_arr)
        return im, points


class ResizeWithPoints(Resize):
    def forward(self, img, points):
        old_size = np.array(img.size)
        new_img = super().forward(img)
        new_size = np.array(new_img.size)

        new_points = points * new_size / old_size
        return new_img, new_points


class ComposeWithPoints(Compose):
    def __call__(self, img, points):
        for t in self.transforms:
            img, points = t(img, points)
        return img, points