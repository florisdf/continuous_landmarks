from random import shuffle

import numpy as np
from torch.utils.data import Dataset, Sampler


class ConcatDataset(Dataset):
    def __init__(self, datasets):
        self.datasets = datasets
        self.culens = [
            sum(len(ds) for ds in datasets[:i+1])
            for i in range(len(datasets))
        ]

    def __len__(self):
        return self.culens[-1]

    def __getitem__(self, idx):
        ds_idx = next(i for i, culen in enumerate(self.culens)
                      if idx < culen)

        if ds_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.culens[ds_idx - 1]

        return self.datasets[ds_idx][sample_idx]


class ConcatBatchSampler(Sampler):
    def __init__(self, concat_dataset, batch_size, shuffle=True):
        self.concat_dataset = concat_dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.init_batched_idxs()

    def init_batched_idxs(self):
        ds_sample_idxs = [
            np.arange(len(ds))
            for ds in self.concat_dataset.datasets
        ]
        if self.shuffle:
            ds_sample_idxs = [
                np.random.permutation(idxs)
                for idxs in ds_sample_idxs
            ]

        self.batched_idxs = []
        start_idx = 0
        for sample_idxs in ds_sample_idxs:
            for i in range(0, len(sample_idxs), self.batch_size):
                self.batched_idxs.append(
                    sample_idxs[i:i + self.batch_size] + start_idx
                )
            start_idx += len(sample_idxs)

        if self.shuffle:
            shuffle(self.batched_idxs)

    def __iter__(self):
        self.init_batched_idxs()
        for sample_idxs in self.batched_idxs:
            yield sample_idxs

    def __len__(self):
        return len(self.batched_idxs)
