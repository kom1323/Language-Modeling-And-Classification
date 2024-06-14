import os
from torch.utils.data import Dataset
from torchtext.data.utils import get_tokenizer


## ADD COLLATE FUNCTION

tokenizer = get_tokenizer('basic_english')

class ImdbDataset(Dataset):
    def __init__(self, data_dir, transform=None) -> None:
        super(ImdbDataset, self).__init__()
        self.data_dir = data_dir
        self.transform = transform
        self. samples = []
        self._load_data()


    def _load_data(self):
        for label in ['pos', 'neg']:
            dir_path = os.path.join(self.data_dir, label)
            for file_name in os.listdir(dir_path):
                with open(os.path.join(dir_path, file_name), 'r', encoding='utf-8') as file:
                    text = file.read()
                    self.samples.append((text, 1 if label == 'pos' else 0))


    def __len__(self):
        return len(self.samples)
    

    def __getitem__(self, index):
        text, label = self.samples[index]
        if self.transform:
            text = self.transform(text)
        return text, label


    def yield_tokens(data_iter):
        for text, _ in data_iter:
            yield tokenizer(text)