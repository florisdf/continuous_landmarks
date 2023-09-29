from collections import namedtuple
import pandas as pd
from pathlib import Path

import cv2
import json
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


LANDMARKS_300W = np.array([
    23404, 4607, 4615, 4655, 20356, 4643, 5022, 5013, 1681, 1692, 11470, 10441,
    1336, 1343, 1303, 1295, 2372, 6143, 6141, 6126, 6113, 6109, 2844, 2762,
    2765, 2774, 2789, 6053, 6041, 1870, 1855, 4728, 4870, 1807, 1551, 1419,
    3434, 3414, 3447, 3457, 3309, 3373, 3179, 151, 127, 143, 3236, 47, 21018,
    4985, 4898, 6571, 1575, 1663, 1599, 1899, 12138, 5231, 21978, 5101, 21067,
    21239, 11378, 11369, 11553, 12048, 5212, 21892
])


ImageParams = namedtuple(
    'ImageParams',
    ['view_id', 'expr', 'subj',
     'mv_scale', 'mv_Rt',
     'K', 'Rt', 'dist', 'h', 'w', 'is_valid']
)


SCALE_DICT = json.load((Path(__file__).parent / 'Rt_scale_dict.json').open())


class FaceScapeLandmarkDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        canonical_shape,
        transform=None,
        view=None,
        subject=None,
        expression=None,
        filter_public=False,
    ):
        view = view or '*'
        subject = subject or '*'
        expression = expression or '*'

        data_path = Path(data_path)

        publishable_path = data_path / 'publishable_list_v1.txt'
        publishable = {int(x)
                       for x in publishable_path.read_text().split(', ')}

        self.df = pd.DataFrame(
            [
                {
                    'image': p,
                    'expression': p.parent.name,
                    'view': int(p.stem),
                    'subject': int(p.parent.parent.name),
                }
                for p in
                (data_path / 'fsmview_trainset')
                .glob(f'{subject}/{expression}/{view}.jpg')
             ]
        )

        if filter_public:
            self.df = self.df[self.df['subject'].isin(publishable)]

        self.transform = transform
        self.canonical = canonical_shape

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.iloc[idx]['image']
        im, points = load_img_with_landmarks(img_path)

        if self.transform is not None:
            im, points = self.transform(im, points)

        return im, points


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
                    'expression': p.stem,
                    'subject': int(p.parent.parent.name),
                }
                for p in
                (Path(data_path) / 'facescape_trainset')
                .glob('*/models_reg/*.obj')
             ]
        )

        if expression is not None:
            assert expression in df.expression.values
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


def get_img_params(img_path):
    view_id = img_path.stem
    expr = img_path.parent.name
    subj = img_path.parent.parent.name
    mv_scale, mv_Rt = map(np.array, SCALE_DICT[subj][expr.split('_')[0]])

    params = json.load((img_path.parent / 'params.json').open())
    K = np.array(params[f'{view_id}_K'])
    Rt = np.array(params[f'{view_id}_Rt'])
    dist = np.array(params[f'{view_id}_distortion'], dtype=np.float32)
    h = params[f'{view_id}_height']
    w = params[f'{view_id}_width']
    is_valid = params[f'{view_id}_valid']

    return ImageParams(
        view_id, expr, subj,
        mv_scale, mv_Rt,
        K, Rt, dist, h, w, is_valid
    )


def transform_tu_points_to_pixel(points, mv_Rt, mv_scale, Rt, K, dist):
    # Transform TU points to world coordinates
    proj_points = points - mv_Rt[:3, 3]
    proj_points = (np.linalg.inv(mv_Rt[:3, :3]) @ proj_points.T).T
    proj_points /= mv_scale

    # Transform TU points from world to pixel coordinates
    rot_vec, _ = cv2.Rodrigues(Rt[:3, :3])
    proj_points, _ = cv2.projectPoints(proj_points, rot_vec, Rt[:3, 3],
                                       K, dist)
    return proj_points.squeeze()


def load_img_with_landmarks(img_path):
    expr = img_path.parent.name
    subj = img_path.parent.parent.name
    data_path = img_path.parent.parent.parent.parent

    obj_path = data_path / f'facescape_trainset/{subj}/models_reg/{expr}.obj'
    img_params = get_img_params(img_path)
    points = np.array(get_obj_verts(obj_path))

    landmarks = transform_tu_points_to_pixel(
        points,
        img_params.mv_Rt, img_params.mv_scale,
        img_params.Rt, img_params.K, img_params.dist
    )

    img = cv2.imread(str(img_path))[..., ::-1]
    img = cv2.undistort(img, img_params.K, img_params.dist)

    return Image.fromarray(img), torch.tensor(landmarks)


def get_eyes_mouth(points):
    e0 = points[LANDMARKS_300W[36:42]].mean(axis=0)
    e1 = points[LANDMARKS_300W[42:48]].mean(axis=0)
    m0 = points[LANDMARKS_300W[48]]
    m1 = points[LANDMARKS_300W[54]]

    return e0, e1, m0, m1
