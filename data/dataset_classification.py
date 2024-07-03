import os
import re
import torch
from torch.utils.data import Dataset, DataLoader
from torchtext.data.utils import get_tokenizer
from collections import Counter
import torchtext

torchtext.disable_torchtext_deprecation_warning()

# ADD COLLATE FUNCTION
tokenizer = get_tokenizer('basic_english')

class ImdbDataset(Dataset):
    def __init__(self, data_dir, transform=None, vocabulary=None) -> None:
        super(ImdbDataset, self).__init__()
        self.data_dir = data_dir
        self.transform = transform
        self.samples = []
        self.vocabulary = vocabulary
        self.tokenizer = get_tokenizer('basic_english')
        self._load_data()
        if vocabulary is None:
            self._create_vocabulary()

    def _clean_text(self, text):
        # Remove <br> tags
        text = re.sub(r'<br\s*/?>', ' ', text)
        # Remove punctuation and non-alphanumeric characters
        #text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
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
            words = self.tokenizer(text)
            word_counter.update(words)
        
        total_count = sum(word_counter.values())
        cumulative_count = 0
        threshold = total_count * 0.97
        
        most_frequent_words = []
        for word, count in word_counter.most_common():
            cumulative_count += count
            most_frequent_words.append(word)
            if cumulative_count >= threshold:
                break
        
        self.vocabulary = {word: idx for idx, word in enumerate(most_frequent_words)}
        self.vocabulary['<unk>'] = len(self.vocabulary)
        self.vocabulary['<pad>'] = len(self.vocabulary) + 1

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        text, label = self.samples[index]
        # Convert text to sequence of word indices using vocab
        indexed_text = [self.vocabulary.get(word, self.vocabulary['<unk>']) for word in self.tokenizer(text)]
        return torch.tensor(indexed_text, dtype=torch.long), torch.tensor(label, dtype=torch.long)

# Collate function for DataLoader
def collate_fn(batch, vocab):
    texts, labels = zip(*batch)
    
    # Pad sequences to the same length within the batch
    max_length = max(len(text) for text in texts)
    padded_texts = [text.tolist() + [vocab['<pad>']] * (max_length - len(text)) for text in texts]
    
    # Convert to tensors
    input_tensor = torch.tensor(padded_texts, dtype=torch.long)
    label_tensor = torch.tensor(labels, dtype=torch.long)
    
    return input_tensor, label_tensor
