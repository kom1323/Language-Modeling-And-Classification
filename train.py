import torch
import torch.nn as nn
import torch.optim as optim
from model.__ini__ import GRULanguageModel 
from data.dataset import ImdbDataset, collate_fn
from torch.utils.data import DataLoader

def train_model(model, criterion, optimizer, train_loader, vocab_size,epochs, device):
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(device)  # Move inputs to GPU
            targets = targets.to(device)  # Move targets to GPU
            
            optimizer.zero_grad()
            outputs, _ = model(inputs)
            loss = criterion(outputs.view(-1, vocab_size), targets.view(-1))
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            if batch_idx % 100 == 99:    # Print every 100 batches
                print(f'Epoch [{epoch+1}/{epochs}], Step [{batch_idx+1}/{len(train_loader)}], Loss: {total_loss / 100:.4f}')
                total_loss = 0.0

    torch.save(model.state_dict(), 'gru_language_model.pth')




if __name__ == "__main__":

    train_dir = 'data/train' 

    train_dataset = ImdbDataset(train_dir)
    
    batch_size = 8
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda b: collate_fn(b, train_dataset.vocabulary))


    # Initialize hyper-parameters
    vocab_size = len(train_dataset.vocabulary)
    embedding_dim = 100
    hidden_dim = 256
    num_layers = 2
    lr = 0.001
    epochs = 5

    # Initialize model, loss function, and optimizer
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("device: ", device)

    model = GRULanguageModel(vocab_size, embedding_dim, hidden_dim, num_layers)
    model = model.to(device)  # Move model to GPU
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_model(model, criterion, optimizer, train_dataloader, vocab_size, epochs, device)
