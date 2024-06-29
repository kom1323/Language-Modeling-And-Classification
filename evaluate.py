import torch
import torch.nn as nn
from model.language_model import GRULanguageModel
from data.dataset import ImdbDataset, collate_fn
from torch.utils.data import DataLoader
import math
import random
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('logs')

def load_model(model_path, vocab_size, embedding_dim, hidden_dim, num_layers, device):
    model = GRULanguageModel(vocab_size, embedding_dim, hidden_dim, num_layers)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model


def test_model(model, test_dataloader, criterion, device, max_samples=1000):
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for iteration, (inputs, targets) in enumerate(test_dataloader):
            if iteration >= max_samples:
                break
            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs, _ = model(inputs)
            loss = criterion(outputs, targets)
            
            total_loss += loss.item() * targets.size(0)
            
            # Compute perplexity
            perplexity = math.exp(loss.item())

            # Write to TensorBoard
            writer.add_scalar('Loss/test', loss.item(), iteration)
            writer.add_scalar('Perplexity/test', perplexity, iteration)
            
            # Print iteration information
            print(f"Iteration: {iteration}, Loss: {loss.item():.4f}, Perplexity: {perplexity:.4f}")



def penalize_repetition(logits, generated_tokens, repetition_penalty=1.2):
    for token in set(generated_tokens):
        logits[0, token] /= repetition_penalty
    return logits

def ngram_blocking(logits, generated_sequence, n=2):
    if len(generated_sequence) < n:
        return logits
    ngram = tuple(generated_sequence[-(n-1):])
    for idx in range(len(logits[0])):
        new_ngram = ngram + (idx,)
        if new_ngram in generated_sequence:
            logits[0, idx] = -float('Inf')
    return logits

def top_k_sampling(logits, k=10):
    top_k_probs, top_k_indices = torch.topk(logits, k)
    top_k_probs = F.softmax(top_k_probs, dim=-1)
    next_token = torch.multinomial(top_k_probs, 1)
    return top_k_indices.gather(-1, next_token)      

def sample_sequence(model, start_sequence, idx_to_word, word_to_idx, max_length=20, device='cpu', temperature=1.0, repetition_penalty=0.5, ngram_size=3):
    model.eval()
    with torch.no_grad():
        if start_sequence:
            input_ids = torch.tensor([word_to_idx.get(word, word_to_idx['<unk>']) for word in start_sequence], dtype=torch.long).unsqueeze(0).to(device)
            generated_sequence = start_sequence[:]
        else:
            # Start with a random word from the vocabulary
            start_word = random.choice(list(word_to_idx.keys()))
            input_ids = torch.tensor([[word_to_idx[start_word]]], dtype=torch.long).to(device)
            generated_sequence = [start_word]

        print(f"Generated Sequence: {start_word} ", end="", flush=True)
        hidden = None
        for _ in range(max_length):
            outputs, hidden = model(input_ids, hidden)
            next_token_logits = outputs[:, -1, :] / temperature # Get the logits for the last timestep
            next_token_logits = penalize_repetition(next_token_logits, input_ids[0], repetition_penalty)

            # Filter out <unk> tokens from the logits
            next_token_logits[:, word_to_idx['<unk>']] = float('-inf')  # Set <unk> token's logit to -infinity

            # Normalize logits to get probabilities
            next_token_probs = F.softmax(next_token_logits, dim=-1)

            # Sample from the distribution
            next_token_id = torch.multinomial(next_token_probs[0], num_samples=1).item()
            next_word = idx_to_word.get(next_token_id, '<unk>')
            
            # Sample from the distribution
            generated_sequence.append(next_word)
            input_ids = torch.tensor([[next_token_id]], dtype=torch.long).to(device)

            # Print the sentence generated so far
            print(f"{next_word} ", end="", flush=True)

    print()
    return ' '.join(generated_sequence)





if __name__ == "__main__":

    train_dir = 'data/train'
    test_dir = 'data/test'

    batch_size = 1
    print("Loading datasets...")
    train_dataset = ImdbDataset(train_dir)


    vocab_size = len(train_dataset.vocabulary)
    embedding_dim = 100
    hidden_dim = 256
    num_layers = 2
    lr = 0.01
    epochs = 15

    #Load and test the model
    # test_dataset = ImdbDataset(test_dir, vocabulary=train_dataset.vocabulary)
    # test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=lambda b: collate_fn(b, test_dataset.vocabulary))


    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("device: ", device)
    print("Vocabulary size: ", vocab_size)
    criterion = nn.CrossEntropyLoss()
    

    print("Loading and testing the model...")
    model = load_model('model/gru_language_model_3500.pth', vocab_size, embedding_dim, hidden_dim, num_layers, device)
    
    #test_model(model, test_dataloader, criterion, device)

    # Generate sentences using the trained model
    idx_to_word = train_dataset.reverse_vocabulary
    word_to_idx = train_dataset.vocabulary

    print("Generating example sequence")
    start_sequence = None
    temperature = 1.5
    repetition_penalty = 0.5
    generated_sentence = sample_sequence(model, start_sequence, idx_to_word, word_to_idx, max_length=100, device=device, temperature=temperature, repetition_penalty=repetition_penalty)
    #print("Final Generated Sentence: ", generated_sentence)



