from torch.utils.data import Dataset


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
