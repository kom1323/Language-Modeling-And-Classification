import torch
import torch.nn as nn
import torch.optim as optim
from model.language_model import GRULanguageModel 
from data.dataset import ImdbDataset, collate_fn
from torch.utils.data import DataLoader
from evaluate import load_model
from model.classification_model import SentimentClassifier
from torch.utils.tensorboard import SummaryWriter 
import math
import os
writer = SummaryWriter('logs')

def train_model(model, criterion, optimizer, train_dataloader, num_epochs, device):
    
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0

        for iteration, (inputs, targets) in enumerate(train_dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            
            # Forward pass
            outputs, _ = model(inputs)
                        
            # Compute loss
            loss = criterion(outputs, targets)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            # Compute perplexity
            perplexity = math.exp(loss.item())
            
            # Write to TensorBoard
            writer.add_scalar('Loss/train', loss.item(), epoch * len(train_dataloader) + iteration)
            writer.add_scalar('Perplexity/train', perplexity, epoch * len(train_dataloader) + iteration)

            print(f"#iteration: {iteration}/{len(train_dataloader)}, Loss:{loss.item():.4f}, Perplexity: {perplexity:.4f}")
            
            if iteration % 500 == 0:
                print("Saving model...")
                torch.save(model.state_dict(), f'gru_language_model_{iteration}.pth')
                print("Model saved.")

        avg_epoch_loss = epoch_loss / len(train_dataloader)
        avg_epoch_perplexity = math.exp(avg_epoch_loss)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {avg_epoch_loss:.4f}, Avg Perplexity: {avg_epoch_perplexity:.4f}')
        
        # Log epoch loss and perplexity
        writer.add_scalar('Train/Avg_Loss', avg_epoch_loss, epoch)
        writer.add_scalar('Train/Avg_Perplexity', avg_epoch_perplexity, epoch)

    print("Training complete.")
    print("Saving model...")
    torch.save(model.state_dict(), 'gru_language_model.pth')
    print("Model saved.")



def classification_train_model(model, criterion, optimizer, dataloader, num_epochs, device, lr=0.001):
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device).float()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(dataloader)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}')
    
    print("Training complete.")
    return model



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
    lr = 0.01
    epochs = 1


    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("device: ", device)
    print("Vocabulary size: ", vocab_size)

    
    # # Initialize model, loss function, and optimizer
    # model = GRULanguageModel(vocab_size, embedding_dim, hidden_dim, num_layers)
    # model = model.to(device) 
    # criterion = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(model.parameters(), lr=lr)

    # print("Starting training loop...")
    # train_model(model, criterion, optimizer, train_dataloader, epochs, device)

    print("Loading and testing the model...")
    pretrained_model = load_model('experiments/gru_language_model_3500.pth', vocab_size, embedding_dim, hidden_dim, num_layers, device)

    sentiment_classifier = SentimentClassifier(pretrained_model, hidden_dim)
    sentiment_classifier.to(device)
    
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(sentiment_classifier.parameters(), lr=lr)  
    num_epochs = 10

    print("Training classification model.")
    trained_model = classification_train_model(sentiment_classifier, criterion, optimizer, train_dataloader, num_epochs, device)