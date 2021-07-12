
import numpy as np
import torch
import torch.optim as optim
from .model import DQN, Dueling_DQN
from .replay_buffers import ReplayBuffer, PrioritizedReplayBuffer
import random

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate 
UPDATE_EVERY = 4        # how often to update the network
PRIORITY_PROBABILITY_A = .9 # Coefficient used to compute the importance of the priority weights during buffer sampling
PRIORITY_CORRECTION_B = 1. # Corrective factor for the loss in case of Priority Replay Buffer
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    def __init__(self, state_size, action_size, hidden_layers=None, seed=0, **kwargs):
        """Basic Agent based on a DQN Network. For Execution only.

        Args:
            state_size (int): size of the state vector
            action_size (int): number of possible actions
            hidden_layers (list of ints, optional): list containing the size of each hidden layer. 
            if None a default architecture will be generated ([64, 64])

            seed (int, optional): seed for the PRNG generatin the initial parameters values. Defaults to 0.
            dueling (bool, optional): if True a Dueling DQN architecture will be used otherwise a classic DQN.
        """
        self.dueling_networks = kwargs.get('dueling', False)
        if self.dueling_networks:
            self.qnetwork_local = Dueling_DQN(state_size, action_size, hidden_layers, seed).to(device)    
        else:
            self.qnetwork_local = DQN(state_size, action_size, hidden_layers, seed).to(device)
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_layer_sizes = self.qnetwork_local.hidden_layer_sizes
    
    @classmethod
    def from_file(cls, filename):
        """Builds the Agent from file

        Args:
            data (dict): dict representation of the agent

        Returns:
            Agent: agent object correspondent to the parameters
        """
        data = torch.load(filename)
        obj = cls.from_dict(data)
        obj.qnetwork_local.load_state_dict(data['weights'])
        return obj
    
    @classmethod
    def from_dict(cls, data):
        """Builds the Agent from a dict representation of the agent

        Args:
            data (dict): dict representation of the agent

        Returns:
            Agent: agent object correspondent to the parameters
        """
        return cls(data['state_size'], data['action_size'], data['hidden_layer_sizes'], **data['kwargs'])

    def to_dict(self):
        """Extrapolates agent's parameter and hyper-parameters and organises them in a dict object

        Returns:
            dict: dict representation of the agent
        """
        data = {
            'state_size': self.state_size,
            'action_size': self.action_size,
            'hidden_layer_sizes': self.hidden_layer_sizes,
            'weights': self.qnetwork_local.state_dict(),
            'seed': self.qnetwork_local.seed.seed(),
            'kwargs': {'dueling': self.dueling_networks}
        }
        return data

    def save_model(self, filename):
        """Saves the agent on a file

        Args:
            filename (str): path to the file to save
        """
        data = self.to_dict()
        torch.save(data, filename)

    def act(self, state, eps=0.):
        """Performs an agent action

        Args:
            state (array of floats): current state vector
            eps (float, optional): ϵ for ϵ-greedy agent. Indicates the probability of a random action. Defaults to 0..

        Returns:
            [type]: [description]
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

class ReplayDDQNAgent(Agent):
    def __init__(self, state_size, action_size, hidden_layers=None, seed=0, **kwargs):
        """Basic Agent based on a DQN Network. For Execution only.

        Args:
            state_size (int): size of the state vector
            action_size (int): number of possible actions
            hidden_layers (list of ints, optional): list containing the size of each hidden layer. 
            if None a default architecture will be generated ([64, 64])

            seed (int, optional): seed for the PRNG generatin the initial parameters values. Defaults to 0.
            dueling (bool, optional): if True a Dueling DQN architecture will be used otherwise a classic DQN.
            rl (float, optional): learning rate of the neural network. Defaults to 5e-4.
            batch_size (float, optional): size of the batch used training the network. Defaults to 64
            gamma (float, optional): discount factor used for the Agent. Defaults to 0.99.
            update_every (float, optional): defines after how many steps the network should be updated. Defaults to 4.
            buffer_size (float, optional): size of the memory for the Memory Replay. Defaults to = int(1e5)
            tau (float, optional): coefficient of (linear) interpolation between
                local and target networks (used in the DDQN soft_update). Defaults to 1e-3.
        """
        
        super().__init__(state_size, action_size, hidden_layers, seed, **kwargs)

        if self.dueling_networks:
            self.qnetwork_target = Dueling_DQN(state_size, action_size, hidden_layers, seed).to(device)    
        else:
            self.qnetwork_target = DQN(state_size, action_size, hidden_layers, seed).to(device)
        
        self.rl = kwargs.get('lr', LR)
        self.batch_size = kwargs.get('batch_size', BATCH_SIZE)
        self.gamma = kwargs.get('gamma', GAMMA)
        self.update_every = kwargs.get('update_every', UPDATE_EVERY)
        self.buffer_size = kwargs.get('buffer_size', BUFFER_SIZE)
        self.tau = kwargs.get('tau', TAU)

        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=self.rl)
        
        self.t_step = 0

        self.memory = ReplayBuffer(action_size, self.buffer_size, self.batch_size, seed)

    def to_dict(self):
        """Extrapolates agent's parameter and hyper-parameters and organises them in a dict object

        Returns:
            dict: dict representation of the agent
        """
        data = super().to_dict()
        data.update({'target_weight': self.qnetwork_target.state_dict()})
        data.update({'replay_memory': self.memory.memory})
        kwargs = {
            'rl': self.rl,
            'batch_size': self.batch_size,
            'gamma': self.gamma,
            'update_every': self.update_every,
            'buffer_size': self.buffer_size,
            'tau': self.tau
        }
        data['kwargs'].update(kwargs)
        return data

    def step(self, state, action, reward, next_state, done):
        """Performs a learning step for the agent.

        Args:
            state ([float array]): array of state vectors 
            action ([float array]): array of action indices
            reward ([float array]): array of rewards
            next_state ([float array]): array of next state vectors 
            done ([boolean array]): array of booleans indicating the end of the episode
        """
        self.memory.add(state, action, reward, next_state, done)
        
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:
            if len(self.memory) > self.batch_size:
                experiences = self.memory.sample()
                self.learn(experiences)
    
    def compute_loss(self, experiences):
        """Computes the loss function for the agent

        Args:
            experiences (tuple): tuple containing the arrays of `state, action, reward, next_state, done`

        Returns:
            torch.FloatTensor: optmisation loss
        """
        states, actions, rewards, next_states, dones = experiences
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        Q_targets = rewards + (self.gamma * Q_targets_next * (1-dones))
        Q_expected = self.qnetwork_local(states).gather(1, actions)
        loss = F.mse_loss(Q_expected, Q_targets)
        return loss

    def learn(self, experiences):
        """This function performs the operations to train the selected network

        Args:
            experiences (tuple): tuple containing the arrays of `state, action, reward, next_state, done`
        """
        loss = self.compute_loss(experiences)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.soft_update()                     

    
    def soft_update(self):
        """Soft update of the model parameters. It applies a linear interpolation based on the self.tau parameter
           target_θ = τ × local_θ + (1 - τ) × target_θ
        """
        local_model, target_model = self.qnetwork_local, self.qnetwork_target

        for local_par, target_par in zip(local_model.parameters(), target_model.parameters()):
            target_par.data.copy_(self.tau * local_par.data + (1-self.tau) * target_par.data)
    


class PriorityReplayDDQNAgent(ReplayDDQNAgent):
    def __init__(self, state_size, action_size, hidden_layers=None, seed=0, **kwargs):
        """Basic Agent based on a DQN Network. For Execution only.

        Args:
            state_size (int): size of the state vector
            action_size (int): number of possible actions
            hidden_layers (list of ints, optional): list containing the size of each hidden layer. 
            if None a default architecture will be generated ([64, 64])

            seed (int, optional): seed for the PRNG generatin the initial parameters values. Defaults to 0.
            dueling (bool, optional): if True a Dueling DQN architecture will be used otherwise a classic DQN.
            rl (float, optional): learning rate of the neural network. Defaults to 5e-4.
            batch_size (float, optional): size of the batch used training the network. Defaults to 64
            gamma (float, optional): discount factor used for the Agent. Defaults to 0.99.
            update_every (float, optional): defines after how many steps the network should be updated. Defaults to 4.
            buffer_size (float, optional): size of the memory for the Memory Replay. Defaults to = int(1e5)
            tau (float, optional): coefficient of (linear) interpolation between
                local and target networks (used in the DDQN soft_update). Defaults to 1e-3.
            priority_probability_a (float, optional): Coefficient used to compute the importance of the priority weights
                during buffer sampling. Defaults to 0.9 .
            priority_correction_b (float, optional): Corrective factor for the loss in case of
                Priority Replay Buffer. Defaults to 1.0
        """
        
        super().__init__(state_size, action_size, hidden_layers=None, seed=0, **kwargs)

        self.beta = kwargs.get('priority_correction_b', PRIORITY_CORRECTION_B)
        self.alpha = kwargs.get('priority_probability_a', PRIORITY_PROBABILITY_A)
        self.memory = PrioritizedReplayBuffer(action_size, self.buffer_size, self.batch_size, seed)

    def to_dict(self):
        """Extrapolates agent's parameter and hyper-parameters and organises them in a dict object

        Returns:
            dict: dict representation of the agent
        """
        data = super().to_dict()
        data.update({'memory_priority': self.memory.priorities})
        kwargs = {
            'priority_correction_b': self.beta,
            'priority_probability_a': self.alpha
        }
        data['kwargs'].update(kwargs)
        return data

    def step(self, state, action, reward, next_state, done):
        
        self.memory.add(state, action, reward, next_state, done)
        
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:
            if len(self.memory) > self.batch_size:
                experiences = self.memory.sample(self.alpha)
                self.learn(experiences)
    
    def compute_loss(self, experiences):
        """Computes the loss function for the agent

        Args:
            experiences (tuple): tuple containing the arrays of `state, action, reward, next_state, done`

        Returns:
            torch.FloatTensor: optmisation loss
        """
        states, actions, rewards, next_states, dones, probs = experiences
        
        w = np.power(self.buffer_size * probs, -self.beta)
        w /= w.max()
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        Q_targets = rewards + (self.gamma * Q_targets_next * (1-dones))
        Q_expected = self.qnetwork_local(states).gather(1, actions)
        loss = (Q_expected - Q_targets).pow(2).squeeze() * torch.Tensor(w).to(device)
        self.memory.update((loss).cpu().detach().numpy() + 1e-5)

        return loss.mean()
