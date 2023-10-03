from pathlib import Path
from PIL import Image

from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, Dataset

from continuous_landmarks.dataset.facescape import load_img_with_landmarks
from continuous_landmarks.dataset.transforms import Resize

size = 512

data_path = Path('/apollo/datasets/FaceScape')

out_path = Path(f'/apollo/datasets/FaceScape_{size}')

img_list = list(data_path.glob('fsmview_trainset/*/*/*.jpg'))

resize = Resize(size)


class DummyDataset(Dataset):
    def __getitem__(self, idx):
        img_path = img_list[idx]
        img_out_path = Path(str(out_path / str(img_path).replace(str(data_path), '.')))
        if img_out_path.exists():
            return idx

        try:
            im, points = load_img_with_landmarks(img_path)
        except FileNotFoundError as e:
            print(e)
            return idx
        im, points = resize(im, points)
        img_out_path.parent.mkdir(exist_ok=True, parents=True)
        im.save(img_out_path)
    
        ldmks_out_path = img_out_path.parent / f'{img_out_path.stem}_ldmks.pth'
        torch.save(points, ldmks_out_path)

        return idx

    def __len__(self):
        return len(img_list)


dl = DataLoader(DummyDataset(), batch_size=10, num_workers=10)

for _ in tqdm(dl):
    pass
