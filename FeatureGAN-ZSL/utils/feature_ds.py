import torch
from torch.utils.data import Dataset, DataLoader


class FeatureDS(Dataset):
    def __init__(self, X, Y):
        self.X = torch.tensor(X).float()
        self.Y = torch.tensor(Y)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

    def __len__(self):
        return len(self.X)


def fds_dataloader(X, Y, batch_size, shuffle):
    ds = FeatureDS(X, Y)
    ds_loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=2)
    return ds_loader
