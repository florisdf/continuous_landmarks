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
                'image': p.parent / f'{p.stem}.png',
                'label': p.stem,
                'keypoints': parse_points(p)
            }
            for p in data_path.glob('*/*.pts')
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


def parse_points(file_path):
    lines = file_path.read_text().split('\n')[:-1]

    m_num_points = re.match(r'n_points: (\d+)', lines[1])
    assert m_num_points
    n_points = int(m_num_points.group(1))

    assert lines[2] == '{'
    assert lines[-1] == '}'

    points = []

    for line in lines[3:-1]:
        x, y = line.split(' ')
        x = float(x)
        y = float(y)
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
