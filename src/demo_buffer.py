import numpy as np
import torch as t

class DemoBuffer:
    def __init__(self, demo_paths, device):
        states, actions = [], []
        for path in demo_paths:
            data = np.load(path)
            states.append(data['states'])
            actions.append(data['actions'])
        
        self.states = t.from_numpy(np.concatenate(states)).float()
        self.actions = t.from_numpy(np.concatenate(actions)).long()
        self.device = device
        print(f"Loaded {len(self.states)} demo frames from {len(demo_paths)} files")
    
    def sample(self, batch_size):
        idx = t.randint(0, len(self.states), (batch_size,))
        return self.states[idx].to(self.device), self.actions[idx].to(self.device)
