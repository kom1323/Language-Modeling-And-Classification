import torch.nn as nn
from model.gru_cell import GRUCell
import torch



class GRULanguageModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers=1):
        super(GRULanguageModel, self).__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.gru_cells = nn.ModuleList([GRUCell(embedding_dim, hidden_dim) if i == 0 else GRUCell(hidden_dim, hidden_dim) for i in range(num_layers)])
        self.fc = nn.Linear(hidden_dim, vocab_size)


    def forward(self, x, hidden=None):
        if hidden is None:
            hidden = [torch.zeros(x.size(0), self.hidden_dim).to(x.device) for _ in range(self.num_layers)]

        
        embedded = self.embedding(x)
        output_sequence = []
        
        for i in range(embedded.size(1)):  # Loop over sequence length
            input_t = embedded[:, i, :]
            new_hidden = []
            
            for layer in range(self.num_layers):
                h_prev = hidden[layer] if layer == 0 else new_hidden[-1]
                h_t = self.gru_cells[layer](input_t, h_prev)
                input_t = h_t
                new_hidden.append(h_t)
            
            hidden = new_hidden
            output_sequence.append(h_t.unsqueeze(1))
        
        output_sequence = torch.cat(output_sequence, dim=1)
        output = self.fc(output_sequence)
        
        return output, hidden
    