import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class QNet(nn.Module):
    
    def __init__(self, state_size=37, action_size=4, hidden_layer_sizes=None, seed=0):
        
        if hidden_layer_sizes is None:
            hidden_layer_sizes = [64, 64]
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_layer_sizes = hidden_layer_sizes
        super(QNet, self).__init__()
        self.input_layer = nn.Linear(self.state_size, self.hidden_layer_sizes[0])
        self.hidden_layers = nn.ModuleList([nn.Linear(l_sz_prev, l_sz)
            for l_sz_prev, l_sz in zip(self.hidden_layer_sizes[0:-1], self.hidden_layer_sizes[1:])
        ])
        self.output_layer = nn.Linear(self.hidden_layer_sizes[-1], action_size)
        self.seed = torch.manual_seed(seed)
        
    def forward(self, state):
        x = F.relu(self.input_layer(state))
        for layer in self.hidden_layers:
            x = F.relu(layer(x))
        
        return self.output_layer(x)

