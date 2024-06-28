import torch
import torch.nn as nn
import torch.optim as optim
from model.language_model import GRULanguageModel 
from data.dataset import ImdbDataset, collate_fn
from torch.utils.data import DataLoader

def train_model(model, criterion, optimizer, train_loader, pad_token_id, epochs, device):
    
    model.train()
    window_size = 20
    for epoch in range(epochs):
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(device)  # Move inputs to GPU
            # targets = targets.to(device)  # Move targets to GPU
            _, seq_len = inputs.size()
            
            total_loss = 0.0
            

            for i in range(1, seq_len):
                hidden = None
                #print(f"Input: {i}/{seq_len}")
                optimizer.zero_grad()

                  # Determine the input sequence
                if i < window_size:
                    input_t = inputs[:, :i]  # Include tokens up to index i
                else:
                    input_t = inputs[:, i - window_size + 1:i]  # Include last window_size tokens
                
                target_t = inputs[:, i]   # Target is the token at index i
                

                # Create a mask to ignore padding tokens
                #target_mask = target_t != pad_token_id

                # Forward pass
                outputs, hidden = model(input_t, hidden)
                output_t = outputs[:, -1, :]

                # Compute loss for the predicted next token, ignoring padding tokens
                loss = criterion(output_t, target_t)

                # Backward pass and optimization step
                loss.backward()
                optimizer.step()


                total_loss += loss.item()
                #in_sample_loss += loss.item()
                # if i % 20 == 19:
                #     print(f'Epoch [{epoch+1}/{epochs}], Step [{batch_idx+1}/{len(train_loader)}]: Input length: {i}/{seq_len}, Sample loss: {in_sample_loss / 20:.4f}')
                #     in_sample_loss = 0.0
                    
            print(f'Epoch [{epoch+1}/{epochs}], Step [{batch_idx+1}/{len(train_loader)}], Loss: {total_loss:.4f}')

    torch.save(model.state_dict(), 'gru_language_model.pth')




if __name__ == "__main__":

    train_dir = 'data/train' 

    print("Loading dataset...")
    train_dataset = ImdbDataset(train_dir)
    
    batch_size = 1
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda b: collate_fn(b, train_dataset.vocabulary))


    # Initialize hyper-parameters
    vocab_size = len(train_dataset.vocabulary)
    embedding_dim = 100
    hidden_dim = 256
    num_layers = 2
    lr = 0.001
    epochs = 1


    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("device: ", device)
    print("Vocabulary size: ", vocab_size)

    
    # Initialize model, loss function, and optimizer
    model = GRULanguageModel(vocab_size, embedding_dim, hidden_dim, num_layers)
    model = model.to(device) 
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    print("Starting training loop...")
    train_model(model, criterion, optimizer, train_dataloader, train_dataset.vocabulary['<pad>'], epochs, device)
