import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from continuous_landmarks.dataset.facescape import FaceScapeTUDataset

parser = argparse.ArgumentParser()
parser.add_argument('expression')
args = parser.parse_args()

expression = args.expression

ds = FaceScapeTUDataset('/apollo/datasets/FaceScape/', expression)
dl = DataLoader(ds, batch_size=100, num_workers=10)

batches = torch.cat([b for b, *_ in tqdm(dl)])
mean_verts = batches.mean(dim=0)

torch.save(mean_verts, f'facescape_{expression}.pth')
