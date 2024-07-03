import torch
import torch.nn as nn

class SentimentClassifier(nn.Module):
    def __init__(self, pretrained_model, hidden_dim, classifier_input_dim=128):
        super(SentimentClassifier, self).__init__()
        self.feature_extractor = pretrained_model
        
        # Freeze the pre-trained model parameters
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        
        # Classification module
        self.fc1 = nn.Linear(hidden_dim, classifier_input_dim)
        self.fc2 = nn.Linear(classifier_input_dim, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        with torch.no_grad():  # Disable gradient calculation for feature extraction
            features, _ = self.feature_extractor(x)
        
        x = torch.relu(self.fc1(features))
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x