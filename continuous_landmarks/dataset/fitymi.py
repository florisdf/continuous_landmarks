import numpy as np
from pathlib import Path
from PIL import Image
import re
from torch.utils.data import Dataset


class FITYMIDataset(Dataset):
    def __init__(self, data_path, transform=None):
        self.data = [
            (p.parent / f'{p.name.replace("_ldmks.txt", ".png")}', parse_points(p))
            for p in data_path.glob('*_ldmks.txt')
        ]
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, points = self.data[idx]

        im = Image.open(img)

        if self.transform is not None:
            im, points = self.transform(im, points)

        return im, points


def parse_points(file_path):
    lines = file_path.read_text().split('\n')[:-1]
    assert len(lines) == 70

    points = []
    for line in lines:
        x, y = line.split(' ')
        x = float(x)
        y = float(y)
        points.append([x, y])
    return np.array(points)


def get_eyes_mouth(points):
    assert len(points) == 70

    e0 = points[68]
    e1 = points[69]
    m0 = points[48]
    m1 = points[54]

    return e0, e1, m0, m1