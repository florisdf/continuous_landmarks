import numpy as np
from pathlib import Path
from PIL import Image
import re
import torch
from torch.utils.data import Dataset

import pandas as pd

from .facescape import LANDMARKS_300W


class Face300WDataset(Dataset):
    def __init__(
        self, data_path,
        transform=None,
        canon_shape_file='facescape_mouth_stretch.pth',
    ):
        self.df = pd.DataFrame([
            {
                'image': get_image_path_from_pts(p),
                'label': p.stem,
                'keypoints': parse_points(p)
            }
            for p in data_path.glob('**/*.pts')
        ])
        self.transform = transform
        canonical_shape = torch.load(Path(__file__).parent / canon_shape_file)
        self.canonical = canonical_shape[LANDMARKS_300W]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = row['image']
        points = row['keypoints']

        im = Image.open(img).convert('RGB')

        if self.transform is not None:
            im, points = self.transform(im, points)

        return im, points, self.canonical


def get_image_path_from_pts(pts_path):
    png_path = pts_path.parent / f'{pts_path.stem}.png'
    jpg_path = pts_path.parent / f'{pts_path.stem}.jpg'

    if png_path.exists():
        return png_path
    elif jpg_path.exists():
        return jpg_path
    else:
        print(f'No image found for {pts_path}')


def parse_points(file_path):
    lines = file_path.read_text().split('\n')

    if lines[-1] == '':
        lines = lines[:-1]

    m_num_points = re.match(r'n_points:\s+(\d+)', lines[1])
    assert m_num_points
    n_points = int(m_num_points.group(1))

    assert lines[2] == '{'
    assert lines[-1] == '}'

    points = []

    for line in lines[3:-1]:
        x, y = line.strip().split(' ')
        # Minus 1 to compensate for Matlab indexing,
        # see https://ibug.doc.ic.ac.uk/resources/300-W/
        x = float(x) - 1
        y = float(y) - 1
        points.append([x, y])

    assert len(points) == n_points
    return np.array(points)


def get_eyes_mouth(points):
    assert len(points) == 68

    e0 = points[36:42].mean(axis=0)
    e1 = points[42:48].mean(axis=0)
    m0 = points[48]
    m1 = points[54]

    return e0, e1, m0, m1
