import os
import re
import torch
from torch.utils.data import Dataset, DataLoader
from torchtext.data.utils import get_tokenizer
from collections import Counter


## ADD COLLATE FUNCTION

tokenizer = get_tokenizer('basic_english')

class ImdbDataset(Dataset):
    def __init__(self, data_dir, transform=None) -> None:
        super(ImdbDataset, self).__init__()
        self.data_dir = data_dir
        self.transform = transform
        self. samples = []
        self.vocabulary = None
        self._load_data()
        self._create_vocabulary()

    def _clean_text(self, text):
        # Remove <br> tags
        text = re.sub(r'<br\s*/?>', ' ', text)
        # Remove punctuation and non-alphanumeric characters
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        return text


    def _load_data(self):
        for label in ['pos', 'neg']:
            dir_path = os.path.join(self.data_dir, label)
            for file_name in os.listdir(dir_path):
                with open(os.path.join(dir_path, file_name), 'r', encoding='utf-8') as file:
                    text = file.read()
                    cleaned_text = self._clean_text(text)
                    self.samples.append((cleaned_text, 1 if label == 'pos' else 0))


    def _create_vocabulary(self):
            word_counter = Counter()
            for text, _ in self.samples:
                words = tokenizer(text)
                word_counter.update(words)
            self.vocabulary = {word: idx for idx, (word, _) in enumerate(word_counter.items())}
            self.vocabulary['<unk>'] = len(self.vocabulary) 

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



# Collate function for DataLoader
def collate_fn(batch):
    texts, labels = zip(*batch)
    # Tokenize the texts
    tokenized_texts = [tokenizer(text) for text in texts]
    # Find the maximum length in the batch
    max_length = max(len(text) for text in tokenized_texts)
    # Pad the sequences
    padded_texts = [text + ['<pad>'] * (max_length - len(text)) for text in tokenized_texts]
    # Convert tokens to indices using the vocabulary
    vocab = train_dataset.vocabulary
    indexed_texts = [[vocab[token] if token in vocab else vocab['<unk>'] for token in text] for text in padded_texts]
    # Convert to tensors
    text_tensor = torch.tensor(indexed_texts, dtype=torch.long)
    label_tensor = torch.tensor(labels, dtype=torch.long)
    return text_tensor, label_tensor



if __name__ == "__main__":

    train_dir = 'data/train' 

    train_dataset = ImdbDataset(train_dir)

    # print("First 10 items in the vocabulary:")
    # for word, idx in list(train_dataset.vocabulary.items())[:10]:
    #     print(f"{word}: {idx}")


    # # Print the length of the dataset
    # print(f"Total number of samples in the dataset: {len(train_dataset)}")

    # # Example of using yield_tokens to print tokens
    # print("Sample tokens:")
    # for tokens in train_dataset.yield_tokens():
    #     print(tokens)
    #     break 

    batch_size = 16
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)


    # Example: Iterating through the DataLoader
    for batch in train_dataloader:
        texts, labels = batch
        print("Texts batch shape:", texts.shape)
        print("Labels batch shape:", labels.shape)
        break  # Just to show the first batch