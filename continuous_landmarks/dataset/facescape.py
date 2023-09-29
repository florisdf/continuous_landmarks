import pandas as pd
from pathlib import Path
import re

import torch
from torch.utils.data import Dataset

EXPRESSIONS = [
    'anger', 'brow_lower', 'brow_raiser', 'cheek_blowing', 'chin_raiser',
    'dimpler', 'eye_closed', 'grin', 'jaw_forward', 'jaw_left', 'jaw_right',
    'lip_funneler', 'lip_puckerer', 'lip_roll', 'mouth_left', 'mouth_right',
    'mouth_stretch', 'neutral', 'sadness', 'smile'
]

LANDMARKS = [
    23404, 4607, 4615, 4655, 20356, 4643, 5022, 5013, 1681, 1692, 11470, 10441,
    1336, 1343, 1303, 1295, 2372, 6143, 6141, 6126, 6113, 6109, 2844, 2762,
    2765, 2774, 2789, 6053, 6041, 1870, 1855, 4728, 4870, 1807, 1551, 1419,
    3434, 3414, 3447, 3457, 3309, 3373, 3179, 151, 127, 143, 3236, 47, 21018,
    4985, 4898, 6571, 1575, 1663, 1599, 1899, 12138, 5231, 21978, 5101, 21067,
    21239, 11378, 11369, 11553, 12048, 5212, 21892
]


class FaceScapeTUDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        expression: str = None
    ):
        df = pd.DataFrame(
            [
                {
                    'obj_path': p,
                    'expression': re.sub(r'^\d+_', '', p.stem),
                    'subject': int(p.parent.parent.name),
                }
                for p in
                (Path(data_path) / 'facescape_trainset')
                .glob('*/models_reg/*.obj')
             ]
        )

        if expression is not None:
            df = df[df.expression == expression].reset_index(drop=True)

        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        verts = get_obj_verts(row['obj_path'])
        return verts, row.subject, row.expression


def get_obj_verts(obj_path: Path):
    def line_to_vert(line: str):
        if line.startswith('v '):
            _, *vertex = line.split(' ')
            return [float(v) for v in vertex]

    verts = []
    lines = obj_path.read_text().splitlines(keepends=False)
    for line in lines:
        vertex = line_to_vert(line)
        if vertex is not None:
            verts.append(vertex)

    return torch.tensor(verts)
