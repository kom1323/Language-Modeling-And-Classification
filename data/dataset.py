import os
import re
import torch
from torch.utils.data import Dataset, DataLoader
from torchtext.data.utils import get_tokenizer
from collections import Counter
import torchtext 
torchtext.disable_torchtext_deprecation_warning()

## ADD COLLATE FUNCTION
tokenizer = get_tokenizer('basic_english')

class ImdbDataset(Dataset):
    def __init__(self, data_dir, transform=None, vocabulary=None, reverse_vocabulary=None) -> None:
        super(ImdbDataset, self).__init__()
        self.data_dir = data_dir
        self.transform = transform
        self.samples = []
        self.vocabulary = vocabulary
        self.reverse_vocabulary = reverse_vocabulary
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
        self.vocabulary['<pad>'] = len(self.vocabulary)
        self.reverse_vocabulary = {idx: word for word, idx in self.vocabulary.items()}

    def __len__(self):
        return len(self.samples)
    

    def __getitem__(self, index):
        text, _ = self.samples[index]
        # Convert text to sequence of word indices using vocab
        indexed_text = [self.vocabulary[word] if word in self.vocabulary else self.vocabulary['<unk>'] for word in self.tokenizer(text)]

        input_target_pairs = []
        sequence_length = 20

        for i in range(len(indexed_text) - 1):
            # Ensure each input sequence is exactly 20 tokens long
            if i < sequence_length - 1:
                # If the sequence is shorter than 20 tokens, pad the start with <pad>
                input_seq = [self.vocabulary['<pad>']] * (sequence_length - 1 - i) + indexed_text[:i + 1]
            else:
                input_seq = indexed_text[i - (sequence_length - 1):i + 1]
            
            target = indexed_text[i + 1]
            target_one_hot = torch.zeros(len(self.vocabulary))
            target_one_hot[target] = 1
            target_one_hot = target_one_hot.long()
            input_target_pairs.append((input_seq, target_one_hot))

        return input_target_pairs

# Collate function for DataLoader
def collate_fn(batch, vocab):
    input_sequences, targets = [], []
    for input_target_pairs in batch:
        for input_seq, target in input_target_pairs:
            input_sequences.append(input_seq)
            targets.append(target)

    # Pad sequences to the same length within the batch
    max_length = max(len(seq) for seq in input_sequences)
    padded_inputs = [seq + [vocab['<pad>']] * (max_length - len(seq)) for seq in input_sequences]

    # Convert to tensors
    input_tensor = torch.tensor(padded_inputs, dtype=torch.long)
    target_tensor = torch.stack(targets)
    
    return input_tensor, target_tensor

