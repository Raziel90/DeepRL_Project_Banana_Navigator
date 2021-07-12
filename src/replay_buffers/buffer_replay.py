from collections import deque, namedtuple
import random
import numpy as np
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ReplayBuffer():
    experience = namedtuple('Experience', field_names=['state', 'action', 'reward', 'next_state', 'done'])
    experience.__qualname__ = 'ReplayBuffer.experience' # necessary for pickling
    def __init__(self, action_size, buffer_size, batch_size, seed):
        self.action_size = action_size
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.seed = random.seed(seed)
        self.memory = deque(maxlen=buffer_size)
    
    def add(self, state, action, reward, next_state, done):
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):


        batch_values = random.sample(self.memory, k=self.batch_size)
        state = torch.from_numpy(np.vstack([e.state for e in batch_values])).float().to(device)
        action = torch.from_numpy(np.vstack([e.action for e in batch_values])).long().to(device)
        reward = torch.from_numpy(np.vstack([e.reward for e in batch_values])).float().to(device)
        next_state = torch.from_numpy(np.vstack([e.next_state for e in batch_values])).float().to(device)
        done = torch.from_numpy(np.vstack([e.done for e in batch_values]).astype(np.uint8)).float().to(device)
        return (state, action, reward, next_state, done)
    
    def __len__(self):
        return len(self.memory)



class PrioritizedReplayBuffer(ReplayBuffer):
    
    def __init__(self, action_size, buffer_size, batch_size, seed):
        super().__init__(action_size, buffer_size, batch_size, seed)
        self.priorities = deque(maxlen=buffer_size)
        self.sampled_idx = None
        
    
    def add(self, state, action, reward, next_state, done):
        super().add(state, action, reward, next_state, done)
        if len(self.priorities) > 0:
            self.priorities.append(max(self.priorities))
        else:
            self.priorities.append(1.0)
    
    def sample(self, a=0.9):
        p = np.power(self.priorities, a)/ np.power(self.priorities, a).sum()
        sample_idx = random.choices(np.arange(len(self.priorities)), k=self.batch_size, weights=p)
        state = torch.from_numpy(np.vstack([self.memory[i].state for i in sample_idx])).float().to(device)
        action = torch.from_numpy(np.vstack([self.memory[i].action for i in sample_idx])).long().to(device)
        reward = torch.from_numpy(np.vstack([self.memory[i].reward for i in sample_idx])).float().to(device)
        next_state = torch.from_numpy(np.vstack([self.memory[i].next_state for i in sample_idx])).float().to(device)
        done = torch.from_numpy(np.vstack([self.memory[i].done for i in sample_idx]).astype(np.uint8)).float().to(device)
        self.sample_idx = sample_idx
        return (state, action, reward, next_state, done, p[sample_idx])
    
    def update(self, priority_vals):
        for idx, p in zip(self.sample_idx, priority_vals):
            self.priorities[idx] = p