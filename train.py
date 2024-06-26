import torch
import torch.nn as nn
import torch.optim as optim
from model.language_model import GRULanguageModel 
from data.dataset import ImdbDataset, collate_fn
from torch.utils.data import DataLoader

def train_model(model, criterion, optimizer, train_loader, vocab_size,epochs, device):
    
    window_size = 50
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(device)  # Move inputs to GPU
            targets = targets.to(device)  # Move targets to GPU
            
            batch_size, seq_len = inputs.size()
            in_sample_loss = 0.0
            ## PROBLEM WITH THE SOFTMAX OUTPUT TO TOKEN!
            for i in range(1, seq_len):

                model.zero_grad()
                input_t = inputs[:, :i]  # Slice to include tokens up to index i
                target_t = inputs[:, i]    # Target is the token at index i

                # Forward pass
                outputs, _ = model(input_t)  # Assuming model expects (batch_size, seq_len)

                output_t = outputs[:, -1, :] 

                #print(f"Batch {batch_idx}: Inputs shape {input_t.shape}, Targets shape {target_t.shape}, Output shape {output_t.shape}")


                # Flatten outputs and targets for loss calculation
                outputs_flat = output_t.view(-1, vocab_size)
                target_t_flat = target_t.view(-1)



                # Compute loss for the predicted next token
                loss = criterion(outputs_flat, target_t_flat)

                # Backward pass and optimization step
                loss.backward()
                optimizer.step()

                in_sample_loss += loss.item()
                total_loss += loss.item()
                if i % 20 == 19:
                    print(f'Epoch [{epoch+1}/{epochs}], Step [{batch_idx+1}/{len(train_loader)}]: Input length: {input_t.shape[1]}, Sample loss: {in_sample_loss / 20:.4f}')
                    in_sample_loss = 0.0
            
            
            if batch_idx % 100 == 99:    # Print every 100 batches
                print(f'Epoch [{epoch+1}/{epochs}], Step [{batch_idx+1}/{len(train_loader)}], Loss: {total_loss / 100:.4f}')
                total_loss = 0.0

    torch.save(model.state_dict(), 'gru_language_model.pth')




if __name__ == "__main__":

    train_dir = 'data/train' 

    print("Loading dataset...")
    train_dataset = ImdbDataset(train_dir)
    
    batch_size = 32
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

    print("Starting training loop...")
    train_model(model, criterion, optimizer, train_dataloader, vocab_size, epochs, device)
