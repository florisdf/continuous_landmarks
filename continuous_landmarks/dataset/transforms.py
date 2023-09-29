import math

import numpy as np
from PIL import Image
import torch
from torchvision import transforms

from ..utils.face_alignment import align_face


class Align:
    def __init__(self, get_eyes_mouth):
        self.get_eyes_mouth = get_eyes_mouth

    def __call__(self, im: Image, points: np.ndarray):
        e0, e1, m0, m1 = self.get_eyes_mouth(points)
        im_arr, points = align_face(np.array(im), points, e0, e1, m0, m1)
        im = Image.fromarray(im_arr)
        return im, points


class Resize(transforms.Resize):
    def forward(self, img, points):
        old_size = np.array(img.size)
        new_img = super().forward(img)
        new_size = np.array(new_img.size)

        new_points = points * new_size / old_size
        return new_img, new_points


class ToTensor(transforms.ToTensor):
    def __call__(self, img, points):
        img = super().__call__(img)
        points = torch.tensor(points)
        return img, points


class Compose(transforms.Compose):
    def __call__(self, img, points):
        for t in self.transforms:
            img, points = t(img, points)
        return img, points


class RandomRotation(transforms.RandomRotation):
    def forward(self, img, points):
        curr_seed = torch.seed()
        new_img = super().forward(img)

        old_width, old_height = img.size
        origin = torch.tensor(
            self.center if self.center is not None else (old_width/2,
                                                         old_height/2)
        )
        points = torch.tensor(points)

        torch.manual_seed(curr_seed)
        angle = - torch.tensor(math.radians(self.get_params(self.degrees)))

        R = torch.tensor([[torch.cos(angle), -torch.sin(angle)],
                          [torch.sin(angle),  torch.cos(angle)]])

        new_points = (R @ (points.T - origin.T) + origin.T).T

        return new_img, new_points


class RandomCrop(transforms.RandomCrop):
    def forward(self, img, points):
        curr_seed = torch.seed()
        new_img = super().forward(img)

        torch.manual_seed(curr_seed)
        top, left, _, _ = self.get_params(img, self.size)
        new_points = points.copy()
        new_points[:, 0] -= left
        new_points[:, 1] -= top
        return new_img, new_points


class RandomResizedCrop(transforms.RandomResizedCrop):
    def forward(self, img, points):
        curr_seed = torch.seed()
        new_img = super().forward(img)

        torch.manual_seed(curr_seed)
        top, left, old_height, old_width = self.get_params(img, self.scale,
                                                           self.ratio)
        new_width, new_height = new_img.size
        new_points = points.copy()
        new_points[:, 0] -= left
        new_points[:, 1] -= top
        new_points[:, 0] *= new_width/old_width
        new_points[:, 1] *= new_height/old_height

        return new_img, new_points


class CenterCrop(transforms.CenterCrop):
    def forward(self, img, points):
        new_img = super().forward(img)

        crop_height, crop_width = self.size
        top = int(round((img.height - crop_height) / 2.))
        left = int(round((img.width - crop_width) / 2.))

        new_points = points.copy()
        new_points[:, 0] -= left
        new_points[:, 1] -= top
        return new_img, new_points


class ColorJitter(transforms.ColorJitter):
    def forward(self, img, points):
        new_img = super().forward(img)
        return new_img, points


class Normalize(transforms.Normalize):
    def forward(self, img, points):
        new_img = super().forward(img)
        return new_img, points
