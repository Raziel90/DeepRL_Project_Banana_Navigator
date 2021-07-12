import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    """
    DQN Module Architecture
    """
    def __init__(self, state_size=37, action_size=4, hidden_layer_sizes=None, seed=None):
        super(DQN, self).__init__()
        if seed is not None:
            self.seed = torch.manual_seed(seed)
        else:
            self.seed = torch.seed()    
        if hidden_layer_sizes is None:
            hidden_layer_sizes = [64, 64]
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_layer_sizes = hidden_layer_sizes

        self.input_layer = nn.Linear(self.state_size, self.hidden_layer_sizes[0])
        self.hidden_layers = nn.ModuleList([nn.Linear(l_sz_prev, l_sz)
            for l_sz_prev, l_sz in zip(self.hidden_layer_sizes[0:-1], self.hidden_layer_sizes[1:])
        ])
        self.output_layer = nn.Linear(self.hidden_layer_sizes[-1], action_size)
        

        
    def forward(self, state):
        x = F.relu(self.input_layer(state))
        for layer in self.hidden_layers:
            x = F.relu(layer(x))
        
        return self.output_layer(x)



class Dueling_DQN(nn.Module):
    """
    Dueling DQN architecture
    """
    def __init__(self, state_size=37, action_size=4, hidden_layer_sizes=None, seed=None):
        super(Dueling_DQN, self).__init__()
        if seed is not None:
            self.seed = torch.manual_seed(seed)
        else:
            self.seed = torch.seed()  

        if hidden_layer_sizes is None:
            hidden_layer_sizes = [64, 64]
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_layer_sizes = hidden_layer_sizes
        
        self.input_layer_val = nn.Linear(self.state_size, self.hidden_layer_sizes[0])
        self.input_layer_adv = nn.Linear(self.state_size, self.hidden_layer_sizes[0])
        
        self.hidden_layers_val = nn.ModuleList([nn.Linear(l_sz_prev, l_sz)
            for l_sz_prev, l_sz in zip(self.hidden_layer_sizes[0:-1], self.hidden_layer_sizes[1:])
        ])

        self.hidden_layers_adv = nn.ModuleList([nn.Linear(l_sz_prev, l_sz)
            for l_sz_prev, l_sz in zip(self.hidden_layer_sizes[0:-1], self.hidden_layer_sizes[1:])
        ])

        self.output_layer_val = nn.Linear(self.hidden_layer_sizes[-1], 1)
        self.output_layer_adv = nn.Linear(self.hidden_layer_sizes[-1], action_size)

    def forward(self, state):
        adv = F.relu(self.input_layer_adv(state))
        val = F.relu(self.input_layer_val(state))
        
        for layer in self.hidden_layers_adv:
            adv = F.relu(layer(adv))
        for layer in self.hidden_layers_val:
            val = F.relu(layer(adv))

        adv = self.output_layer_adv(adv)
        val = self.output_layer_adv(val).expand(state.size(0), self.action_size)
        
        return val + adv - adv.mean(1).unsqueeze(1).expand(state.size(0), self.action_size)