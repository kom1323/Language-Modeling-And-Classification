import torch
import torch.nn as nn



class GRUCell(nn.Module):
    def __init__(self, input_dim, hidden_dim) -> None:
        super(GRUCell, self).__init__()
        self.hidden_dim = hidden_dim
        self.W_z = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.W_r = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.W_h = nn.Linear(input_dim + hidden_dim, hidden_dim)


    def forward(self, x, h_prev):
        combined = torch.cat((x, h_prev), dim=1)
        z_t = torch.sigmoid(self.W_z(combined))
        r_t = torch.sigmoid(self.W_r(combined))
        combined_reset = torch.cat((x, r_t * h_prev), dim=1)
        h_tilde = torch.tanh(self.W_h(combined_reset))
        h_t = (1 - z_t) * h_prev + z_t * h_tilde
        return h_t
    

