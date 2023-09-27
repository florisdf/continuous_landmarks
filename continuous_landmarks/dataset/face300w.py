import numpy as np
from pathlib import Path
from PIL import Image
import re
from torch.utils.data import Dataset


class Face300WDataset(Dataset):
    def __init__(self, data_path, transform=None):
        self.data = [
            (p.parent / f'{p.stem}.png', parse_points(p))
            for p in data_path.glob('*/*.pts')
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


def get_eyes_mouth_300w(points):
    assert len(points) == 68

    e0 = points[36:42].mean(axis=0)
    e1 = points[42:48].mean(axis=0)
    m0 = points[48]
    m1 = points[54]

    return e0, e1, m0, m1