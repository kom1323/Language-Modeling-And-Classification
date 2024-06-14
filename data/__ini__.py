import os
from torch.utils.data import DataLoader, Dataset



class ImdbDataset(Dataset):
    def __init__(self) -> None:
        super().__init__()

