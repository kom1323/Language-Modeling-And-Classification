import torch
import torch.nn as nn
from model.language_model import GRULanguageModel
from data.dataset import ImdbDataset, collate_fn
from torch.utils.data import DataLoader
import math
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

     # Load and test the model
    test_dataset = ImdbDataset(test_dir, vocabulary=train_dataset.vocabulary)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=lambda b: collate_fn(b, test_dataset.vocabulary))


    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("device: ", device)
    print("Vocabulary size: ", vocab_size)
    criterion = nn.CrossEntropyLoss()
    

    print("Loading and testing the model...")
    model = load_model('model/gru_language_model.pth', vocab_size, embedding_dim, hidden_dim, num_layers, device)
    
    test_model(model, test_dataloader, criterion, device)