import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class FastMemoryBuffer:
    def __init__(self, capacity, state_dim=117, action_dim=15):
        self.capacity = capacity
        self.states = np.zeros((capacity, state_dim), dtype=np.float16)
        self.targets = np.zeros((capacity, action_dim), dtype=np.float16)
        self.masks = np.zeros(capacity, dtype=np.int32) # <-- NEW: Store the integer mask
        self.ptr = 0
        self.size = 0

    def push(self, state, target, mask_int): # <-- NEW: Accept mask_int
        self.states[self.ptr] = state
        self.targets[self.ptr] = target
        self.masks[self.ptr] = mask_int # <-- NEW: Save it
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        if self.ptr == 0:
            print(f"Buffer reached capacity ({self.capacity}). Wrapping around to overwrite oldest data.")

    def sample(self, batch_size):
        indices = np.random.randint(0, self.size, size=batch_size)
        return self.states[indices], self.targets[indices], self.masks[indices]
    
    def save(self, folder, name):
        os.makedirs(folder, exist_ok=True)
        # Use only the valid data up to self.size
        np.savez_compressed(
            os.path.join(folder, f"{name}.npz"),
            states=self.states[:self.size],
            targets=self.targets[:self.size],
            masks=self.masks[:self.size],
            ptr = self.ptr
        )

    def load(self, filepath):
        data = np.load(filepath)
        loaded_states = data['states']
        loaded_targets = data['targets']
        loaded_masks = data['masks']
        loaded_size = len(loaded_states)
        
        # Failsafe: If you somehow try to load a file larger than your 
        # current capacity, only take the most recent data from the end.
        if loaded_size > self.capacity:
            loaded_states = loaded_states[-self.capacity:]
            loaded_targets = loaded_targets[-self.capacity:]
            loaded_masks = loaded_masks[-self.capacity:]
            loaded_size = self.capacity

        # Calculate how much space is left before we hit the end of the array
        space_to_end = self.capacity - self.ptr
        
        if loaded_size <= space_to_end:
            # The loaded chunk fits perfectly without wrapping around
            self.states[self.ptr : self.ptr + loaded_size] = loaded_states
            self.targets[self.ptr : self.ptr + loaded_size] = loaded_targets
            self.masks[self.ptr : self.ptr + loaded_size] = loaded_masks
        else:
            # The chunk hits the end of the buffer and needs to wrap around
            # 1. Fill the array up to the very end
            self.states[self.ptr : self.capacity] = loaded_states[:space_to_end]
            self.targets[self.ptr : self.capacity] = loaded_targets[:space_to_end]
            self.masks[self.ptr : self.capacity] = loaded_masks[:space_to_end]
            
            # 2. Wrap around and put the rest at the beginning
            leftover = loaded_size - space_to_end
            self.states[0 : leftover] = loaded_states[space_to_end:]
            self.targets[0 : leftover] = loaded_targets[space_to_end:]
            self.masks[0 : leftover] = loaded_masks[space_to_end:]
        
        # Update the pointers as if we just pushed `loaded_size` individual items
        self.ptr =  data['ptr']
        self.size = min(self.size + loaded_size, self.capacity)

class StrategyNetwork(nn.Module):
    def __init__(self, input_size=117, hidden_size=256, action_space_size=15):
        super(StrategyNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(), # Standard ReLU is fine here, no negative targets
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, action_space_size)
        )

    def forward(self, x, valid_actions_mask=None):
        logits = self.network(x)
        
        # If we provide a mask during gameplay, force invalid actions to -infinity 
        # so they become 0% probability after softmax.
        if valid_actions_mask is not None:
            # Convert boolean mask to tensor if it isn't already
            mask_tensor = torch.as_tensor(valid_actions_mask, dtype=torch.bool, device=logits.device)
            logits = logits.masked_fill(~mask_tensor, float('-inf'))
            
        return F.softmax(logits, dim=-1)
    
class RegretNetwork(nn.Module):
    def __init__(self, input_size=117, hidden_size=512, action_space_size=15):
        """
        A feedforward neural network that predicts the counterfactual regret
        for every abstracted action given the current state.
        """
        super(RegretNetwork, self).__init__()
        
        # We use LeakyReLU to prevent "dead neurons" if regrets go heavily negative early on
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_size // 2, action_space_size) 
            # Note: No activation on the final layer! Regrets can be positive or negative real numbers.
        )

    def forward(self, x):
        """
        Passes the vectorized state through the network.
        Returns a tensor of predicted regrets for each action.
        """
        return self.network(x)


class ValueNetwork(nn.Module):
    def __init__(self, input_dim=117):
        super(ValueNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Tanh() # Forces output to be between -1.0 (lose max) and +1.0 (win max)
        )

    def forward(self, x):
        return self.net(x)

class ValueMemoryBuffer:
    def __init__(self, capacity, state_dim=117):
        self.capacity = capacity
        self.states = np.zeros((capacity, state_dim), dtype=np.float16)
        # Target is just a single float (the expected value)
        self.targets = np.zeros((capacity, 1), dtype=np.float16) 
        self.ptr = 0
        self.size = 0

    def push(self, state, target_ev):
        self.states[self.ptr] = state
        self.targets[self.ptr, 0] = target_ev
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        indices = np.random.randint(0, self.size, size=batch_size)
        return self.states[indices], self.targets[indices]
    
    def save(self, folder, name):
        os.makedirs(folder, exist_ok=True)
        np.savez_compressed(
            os.path.join(folder, f"{name}.npz"),
            states=self.states[:self.size],
            targets=self.targets[:self.size],
            ptr=self.ptr
        )

    def load(self, filepath):
        data = np.load(filepath)
        loaded_states = data['states']
        loaded_targets = data['targets']
        loaded_size = len(loaded_states)
        
        # Failsafe: If the loaded file is somehow larger than the new capacity,
        # only take the most recent entries from the end of the loaded data.
        if loaded_size > self.capacity:
            loaded_states = loaded_states[-self.capacity:]
            loaded_targets = loaded_targets[-self.capacity:]
            loaded_size = self.capacity

        # Calculate how much space is left before hitting the end of the array
        space_to_end = self.capacity - self.ptr
        
        if loaded_size <= space_to_end:
            # The loaded chunk fits perfectly without wrapping around
            self.states[self.ptr : self.ptr + loaded_size] = loaded_states
            self.targets[self.ptr : self.ptr + loaded_size] = loaded_targets
        else:
            # The chunk hits the end of the buffer and needs to wrap around
            # 1. Fill the array up to the very end
            self.states[self.ptr : self.capacity] = loaded_states[:space_to_end]
            self.targets[self.ptr : self.capacity] = loaded_targets[:space_to_end]
            
            # 2. Wrap around and place the remaining data at the beginning
            leftover = loaded_size - space_to_end
            self.states[0 : leftover] = loaded_states[space_to_end:]
            self.targets[0 : leftover] = loaded_targets[space_to_end:]
        
        # Update the pointers to account for the newly added rows
        self.ptr = data['ptr']
        self.size = min(self.size + loaded_size, self.capacity)