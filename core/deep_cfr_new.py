# deep_cfr.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from collections import deque
import copy
from core.model_new import PokerNetwork, encode_state, VERBOSE, set_verbose
from enum import Enum
from core.utilities import get_top_k_keep_indices
from agents.test_agents import RandomAgent

FALLBACK_AGENT = RandomAgent(stream=False)

KEEP_PAIRS = [(i, j) for i in range(5) for j in range(i + 1, 5)]
NUM_DISCARD_CLASSES = len(KEEP_PAIRS)  
def calculate_entropy(logits, mask):
    # Apply mask to ignore illegal actions
    probs = F.softmax(logits, dim=1)
    probs = probs * mask
    probs = probs / probs.sum(dim=1, keepdim=True)
    
    # Entropy = -sum(p * log(p))
    entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
    return entropy.mean().item()

class AgentAction(Enum):
    FOLD = 0
    CHECK = 1
    CALL = 2
    RAISE_MIN = 3
    RAISE_HALF = 4
    RAISE_POT = 5
    RAISE_ALL_IN = 6
    DISCARD_BEST = 7
    DISACRD_2nd = 8
    RAISE_3rd = 9
    RAISE_34 = 10

def _resolve_model_save_path(path_prefix, iteration):
    """Accept either a path prefix or an explicit .pt filename."""
    if str(path_prefix).endswith(".pt"):
        return str(path_prefix)
    return f"{path_prefix}_iteration_{iteration}.pt"

import numpy as np
import random
import torch # Only needed if you have other torch dependencies, but not for saving anymore

class PrioritizedMemory:
    """Enhanced memory buffer with prioritized experience replay and Reservoir Sampling."""
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.states = np.empty((capacity, 123), dtype=np.float32)
        self.actions = np.empty(capacity, dtype=np.int8)    # int8 handles actions 0-9 safely
        self.regrets = np.empty(capacity, dtype=np.float32) 
        self.iters = np.empty(capacity, dtype=np.int16)     # int16 handles up to 32,767 iterations
        self.priorities = np.empty(capacity, dtype=np.float32)
        
        # 1. Changed 'position' to 'total_seen' for Reservoir Sampling
        self.ptr = 0
        self.size = 0
        self._max_priority = 1.0 
    def add_batch(self, batch_s, batch_a, batch_r, batch_i, batch_p):
        """Ingests the worker's dictionary arrays instantly without tuples."""
        batch_size = len(batch_s)
        self._max_priority = max(self._max_priority, np.max(batch_p))
        batch_priorities = np.power(batch_p, self.alpha)

        # 3. Handle batches larger than the whole buffer (edge case)
        if batch_size >= self.capacity:
            self.states[:] = batch_s[-self.capacity:]
            self.actions[:] = batch_a[-self.capacity:]
            self.regrets[:] = batch_r[-self.capacity:]
            self.iters[:] = batch_i[-self.capacity:]
            self.priorities[:] = batch_priorities[-self.capacity:]
            self.ptr = 0
            self.size = self.capacity
            return

        # Calculate where the slice ends
        end_idx = self.ptr + batch_size

        # 4. Fast Array Slicing (Zero Python loops!)
        if end_idx <= self.capacity:
            # Fits perfectly in the current un-wrapped space
            self.states[self.ptr:end_idx] = batch_s
            self.actions[self.ptr:end_idx] = batch_a
            self.regrets[self.ptr:end_idx] = batch_r
            self.iters[self.ptr:end_idx] = batch_i
            self.priorities[self.ptr:end_idx] = batch_priorities
        else:
            # Wraps around the end of the buffer!
            overflow = end_idx - self.capacity
            first_part = batch_size - overflow
            
            # Fill to the end
            self.states[self.ptr:] = batch_s[:first_part]
            self.actions[self.ptr:] = batch_a[:first_part]
            self.regrets[self.ptr:] = batch_r[:first_part]
            self.iters[self.ptr:] = batch_i[:first_part]
            self.priorities[self.ptr:] = batch_priorities[:first_part]
            
            # Wrap to the beginning
            self.states[:overflow] = batch_s[first_part:]
            self.actions[:overflow] = batch_a[first_part:]
            self.regrets[:overflow] = batch_r[first_part:]
            self.iters[:overflow] = batch_i[first_part:]
            self.priorities[:overflow] = batch_priorities[first_part:]

        self.ptr = end_idx % self.capacity
        self.size = min(self.size + batch_size, self.capacity)
            
    def add(self, state, action, regret, iteration, priority=None):
        """Adds a single experience directly into the C-arrays."""
        if priority is None:
            priority = self._max_priority
            
        if priority > self._max_priority:
            self._max_priority = priority
            
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.regrets[self.ptr] = regret
        self.iters[self.ptr] = iteration
        self.priorities[self.ptr] = priority ** self.alpha
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
            
    def sample(self, batch_size, beta=0.4):
        """Samples directly from C-arrays. Returns flat, batched NumPy arrays!"""
        current_size = len(self)
        if current_size == 0:
            return None, [], []

        # 1. Vectorized probability calculation (insanely fast)
        current_priorities = self.priorities[:current_size]
        total_priority = np.sum(current_priorities)
        probabilities = current_priorities / total_priority

        # 2. Sample indices
        if current_size < batch_size:
            indices = np.arange(current_size)
            batch_size = current_size # adjust for weight calc
        else:
            indices = np.random.choice(current_size, batch_size, p=probabilities, replace=False)

        # 3. Vectorized Importance Sampling Weights
        sample_probs = probabilities[indices]
        weights = (current_size * sample_probs) ** -beta
        weights /= weights.max() # Normalize so max is 1.0

        # 4. Return clean slices of our C-arrays
        samples = (
            self.states[indices],
            self.actions[indices],
            self.regrets[indices],
            self.iters[indices]
        )

        return samples, indices, weights.astype(np.float32)

    def update_priority(self, indices, priorities):
        """Accepts either a single index or a NumPy array of indices to update in bulk."""
        priorities = np.maximum(1e-8, priorities)
        
        # Track max priority
        current_max = np.max(priorities)
        if current_max > self._max_priority:
            self._max_priority = current_max
            
        self.priorities[indices] = priorities ** self.alpha

    def __len__(self):
        return self.size

    def get_memory_stats(self):
        current_size = len(self)
        if current_size == 0:
            return {"min": 0, "max": 0, "mean": 0, "median": 0, "size": 0}
            
        raw_priorities = self.priorities[:current_size] ** (1/self.alpha)
        return {
            "min": float(np.min(raw_priorities)),
            "max": float(np.max(raw_priorities)),
            "mean": float(np.mean(raw_priorities)),
            "median": float(np.median(raw_priorities)),
            "size": current_size
        }


class StrategyMemory:
    """Pure Reservoir Sampling buffer for Deep CFR Strategy tracking."""
    def __init__(self, capacity):
        self.capacity = capacity
        self.states = np.empty((capacity, 123), dtype=np.float32)
        self.probs = np.empty((capacity, 11), dtype=np.float32)
        self.iters = np.empty(capacity, dtype=np.int16)
        self.reach_probs = np.zeros(capacity, dtype=np.float32)
        self.ptr = 0
        self.size = 0
    def append(self, state, probs, iteration, reach_prob):
        """Adds a single strategy profile directly into the C-arrays."""
        self.states[self.ptr] = state
        self.probs[self.ptr] = probs
        self.iters[self.ptr] = iteration
        self.reach_probs[self.ptr] = reach_prob
        
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    def add_batch(self, batch_s, batch_p, batch_i, batch_rp):
        """Processes a whole worker chunk instantly without tuples."""
        batch_size = len(batch_s)
        
        # If the batch is somehow larger than the whole buffer, just take the newest part
        if batch_size >= self.capacity:
            self.states[:] = batch_s[-self.capacity:]
            self.probs[:] = batch_p[-self.capacity:]
            self.iters[:] = batch_i[-self.capacity:]
            self.reach_probs[:] = batch_rp[-self.capacity:]
            self.ptr = 0
            self.size = self.capacity
            return

        # Calculate where the slice ends
        end_idx = self.ptr + batch_size

        if end_idx <= self.capacity:
            # Fits perfectly in the current un-wrapped space
            self.states[self.ptr:end_idx] = batch_s
            self.probs[self.ptr:end_idx] = batch_p
            self.iters[self.ptr:end_idx] = batch_i
            self.reach_probs[self.ptr:end_idx] = batch_rp
        else:
            # Wraps around the end of the buffer!
            overflow = end_idx - self.capacity
            first_part = batch_size - overflow
            
            # Fill to the end
            self.states[self.ptr:] = batch_s[:first_part]
            self.probs[self.ptr:] = batch_p[:first_part]
            self.iters[self.ptr:] = batch_i[:first_part]
            self.reach_probs[self.ptr:] = batch_rp[:first_part]
            
            # Wrap to the beginning
            self.states[:overflow] = batch_s[first_part:]
            self.probs[:overflow] = batch_p[first_part:]
            self.iters[:overflow] = batch_i[first_part:]
            self.reach_probs[:overflow] = batch_rp[first_part:]

        self.ptr = end_idx % self.capacity
        self.size = min(self.size + batch_size, self.capacity)
    def __len__(self):
            """Returns the actual number of valid entries, up to capacity."""
            return self.size

    def sample(self, batch_size):
        """Samples directly from the C-arrays. Returns batched NumPy arrays!"""
        current_size = len(self)
        if current_size == 0:
            return np.array([]), np.array([]), np.array([]), np.array([])
            
        if current_size < batch_size:
            indices = np.arange(current_size)
        else:
            indices = np.random.choice(current_size, batch_size, replace=False)
            
        # Instead of a list of tuples, we return clean slices of our C-arrays.
        # PyTorch DataLoaders will love this.
        return (
            self.states[indices],
            self.probs[indices],
            self.iters[indices],
            self.reach_probs[indices]
        )
        
    def save_compressed(self, filename):
        """Instantly slices the valid data and ZIPs it to disk. Zero list unpacking."""
        if self.size == 0:
            print("Buffer is empty, nothing to save.")
            return

        # 1. Slice out only the valid data (ignoring the empty pre-allocated space)
        s_arr = self.states[:self.size]
        p_arr = self.probs[:self.size]
        i_arr = self.iters[:self.size]
        rp_arr = self.reach_probs[:self.size]
        
        # 2. Save capacity, ptr, and size so the Circular Buffer resumes perfectly!
        meta = np.array([self.capacity, self.ptr, self.size], dtype=np.int64)

        # 3. Compress and save
        np.savez_compressed(filename, s=s_arr, p=p_arr, i=i_arr, rp=rp_arr, meta=meta)
        print(f"Successfully zipped {self.size} strategy samples to {filename}")

    def load_compressed(self, filename):
        """Loads the zipped file and drops it straight into the C-arrays."""
        with np.load(filename) as data:
            s_arr, p_arr, i_arr, rp_arr = data['s'], data['p'], data['i'], data['rp']
            
            # Restore the Circular Buffer math
            if 'meta' in data:
                self.capacity = int(data['meta'][0])
                self.ptr = int(data['meta'][1])
                self.size = int(data['meta'][2])
            else:
                self.size = len(s_arr)
                self.ptr = self.size % self.capacity
            
            state_shape = s_arr.shape[1:]
            action_dim = p_arr.shape[1]
            
            # Re-allocate the empty arrays to match the loaded shapes
            self.states = np.empty((self.capacity, *state_shape), dtype=np.float32)
            self.probs = np.empty((self.capacity, action_dim), dtype=np.float32)
            self.iters = np.empty(self.capacity, dtype=np.float32) # Updated to float32 for PyTorch
            self.reach_probs = np.empty(self.capacity, dtype=np.float32)
            
            # Copy the loaded data into the start of our pre-allocated blocks
            self.states[:self.size] = s_arr
            self.probs[:self.size] = p_arr
            self.iters[:self.size] = i_arr
            self.reach_probs[:self.size] = rp_arr
            
            print(f"Loaded {self.size} strategy samples from {filename} (Pointer at: {self.ptr})")
class DeepCFRAgent:
    def __init__(self, player_id=0, memory_size=300000, device='cpu'):
        self.player_id = player_id
        self.device = device
        self.num_actions = 11 # Fold(0), Raise(1), Check(2), Call(3), Discard(4)
        
        self.advantage_net = PokerNetwork().to(device)
        self.strategy_net = PokerNetwork().to(device)
        
        self.optimizer = optim.Adam(self.advantage_net.parameters(), lr=1e-3)
        self.strategy_optimizer = optim.Adam(self.strategy_net.parameters(), lr=1e-4, weight_decay=1e-5)
        
        self.advantage_memory = PrioritizedMemory(memory_size)
        self.strategy_memory = StrategyMemory(capacity=memory_size * 2)
        
        self.iteration_count = 0
        self.min_bet_size = 0.1
        self.max_bet_size = 3.0
        self.metrics = {
        'action_loss': 0.0
        }
        self.action_lookup = {}
        
        # Iterate through all 32 possible combinations of the 5 valid_mask flags
        for f in [0, 1]:     # FOLD
            for r in [0, 1]: # RAISE
                for ch in [0, 1]:# CHECK
                    for ca in [0, 1]:# CALL
                        for d in [0, 1]: # DISCARD
                            
                            actions = []
                            if f: actions.append(AgentAction.FOLD.value)
                            if ch: actions.append(AgentAction.CHECK.value)
                            if ca: actions.append(AgentAction.CALL.value)
                            if d: 
                                actions.extend([
                                    AgentAction.DISCARD_BEST.value, 
                                    AgentAction.DISACRD_2nd.value # Fixed typo here
                                ])
                            if r:
                                actions.extend([
                                    AgentAction.RAISE_MIN.value,
                                    AgentAction.RAISE_HALF.value,
                                    AgentAction.RAISE_POT.value,
                                    AgentAction.RAISE_ALL_IN.value,
                                    AgentAction.RAISE_34.value,
                                    AgentAction.RAISE_3rd.value
                                ])
                                
                            # Prune Dominated Actions
                            if (AgentAction.CHECK.value in actions or AgentAction.DISCARD_BEST.value in actions) and AgentAction.FOLD.value in actions:
                                actions.remove(AgentAction.FOLD.value)
                                
                            # Save as a tuple for memory efficiency and speed
                            self.action_lookup[(f, r, ch, ca, d)] = tuple(actions)
    def get_legal_action_types(self, obs):
        mask_key = tuple(obs['valid_actions'])
        return self.action_lookup[mask_key]
    def get_unaliased_legal_actions(self, obs_dict):
        # 1. Ultra-fast lookup
        base_actions = self.action_lookup[tuple(obs_dict['valid_actions'])]
        
        # 2. Fastest possible exit (No generator overhead, just direct C-level checks)
        if 3 not in base_actions and 4 not in base_actions and 5 not in base_actions:
            return base_actions

        max_raise = obs_dict["max_raise"]
        min_raise = obs_dict['min_raise']
        bad_actions = []

        # 3. Check Action 3 (Min-Raise) immediately
        if 3 in base_actions and obs_dict["min_raise"] >= max_raise:
            bad_actions.append(3)

        # 4. Lazy evaluation: ONLY do the pot math if Action 4 or 5 are actually legal options
        if any(a in base_actions for a in [4, 5, 9, 10, 11]):
            my_bet = obs_dict["my_bet"]
            opp_bet = obs_dict["opp_bet"]
            
            to_call = (opp_bet - my_bet) if opp_bet > my_bet else 0
            
            # 🚨 Make sure you verified the string key for total pot!
            pot_after_call = obs_dict.get("pot", my_bet + opp_bet) + to_call
            
            # Calculate the absolute chip sizes of the fractional raises
            raises = {
                4: 0.50 * pot_after_call,
                5: 1.00 * pot_after_call,
                9: 0.25 * pot_after_call,
                10: 0.75 * pot_after_call,
                11: 0.10 * pot_after_call
            }
            
            # Prune if it hits the ceiling (max_raise) OR the floor (min_raise)
            for action_id, raise_amount in raises.items():
                if action_id in base_actions:
                    if raise_amount >= max_raise or raise_amount <= min_raise:
                        bad_actions.append(action_id)
        # 5. Fast return if no aliases were found
        if not bad_actions:
            return base_actions
            
        # 6. C-speed list comprehension (much faster than a for-loop with .append)
        return [a for a in base_actions if a not in bad_actions]
    def format_gym_action(self, internal_action, obs_dict):
        """
        Translates Internal Agent Logic -> Gym Environment Format.
        Internal Actions: 0:Fold, 1:Check, 2:Call, 3:Min, 4:Half, 5:Pot, 6:AllIn, 7:Discard
        """
        gym_type = 0
        raise_val = 0
        k1, k2 = 0, 0

        # --- FOLD, CHECK, CALL ---
        if internal_action == 0: # Fold
            gym_type = 0
        elif internal_action == 1: # Check
            gym_type = 2
        elif internal_action == 2: # Call
            gym_type = 3

        # --- DISCRETE RAISES (3, 4, 5, 6) ---
        elif internal_action in [3, 4, 5, 6, 9, 10, 11]:
            gym_type = 1 
            
            my_bet = obs_dict["my_bet"]
            opp_bet = obs_dict["opp_bet"]
            to_call = max(0, opp_bet - my_bet)
            
            # The 'Live' pot includes current bets + the amount we must call
            current_pot = obs_dict.get("pot_size", my_bet + opp_bet)
            pot_after_call = current_pot + to_call
            
            # Map indices to strategic multipliers
            # 3: Min-Raise (add 0 to call)
            # 4: 0.5 Pot
            # 5: 1.0 Pot
            # 6: All-In
            
            multipliers = {3: 0.0, 4: 0.5, 5: 1.0, 6: 100.0, 9: .25, 10:.75, 12:.1}
            
            # Robust Calculation: Amount to Call + (Multiplier * New Pot)
            # This ensures 'Pot' is always a real Pot-sized raise.
            target_total = opp_bet + (multipliers[internal_action] * pot_after_call)
        
            # 2. Convert "Total Bet" to "Incremental Raise" (the 'on-top' amount)
             # Because the env does: self.bets[acting] = opp_bet + raise_amount
            incremental_raise = target_total - opp_bet
            
            # Clamp to the Env's strict min/max bounds
            raise_val = int(max(min(incremental_raise, obs_dict["max_raise"]), obs_dict["min_raise"]))
            if raise_val == obs_dict["max_raise"] and internal_action != 6:
                print(f"Warning: Action {internal_action} was clamped into an All-In! Pot: {current_pot}, Stack: {obs_dict['max_raise']}")

        # --- DISCARD (7) ---
        elif internal_action in [7,8]:
            gym_type = 4
            # Fallback to heuristic
            indices = get_top_k_keep_indices(obs_dict, k=2)
            k1, k2 = indices[internal_action % 7]
        return (gym_type, raise_val, k1, k2)

    def cfr_traverse(self, env, iteration, depth=0, reach_prob=1.0):
        current_player = env.acting_agent
        
        # 🚨 Use the synced legal actions directly from the env
            
        obs_tuple = env._get_obs(winner=None)[0] 
        obs_dict = obs_tuple[current_player]
        legal_actions = self.get_unaliased_legal_actions(obs_dict)
        if not legal_actions:
            return 0.0
        # --- OPPONENT'S TURN ---
        if current_player != self.player_id:
            state_tensor = torch.FloatTensor(encode_state(obs_dict)).unsqueeze(0).to(self.device)
            with torch.no_grad():
                # 🚨 1. Query the ADVANTAGE network, getting raw regrets (logits)
                raw_regrets = self.advantage_net(state_tensor)[0].cpu().numpy()
            
            # 2. Filter for only the legal actions in this state
            legal_regrets = np.array([raw_regrets[a] for a in legal_actions])
            
            # 🚨 3. REGRET MATCHING (The core math of CFR)
            # We zero-out all negative regrets (Equivalent to ReLU)
            positive_regrets = np.maximum(legal_regrets, 0)
            sum_positive_regrets = np.sum(positive_regrets)
            
            # 4. Convert to probabilities
            if sum_positive_regrets > 0: 
                legal_probs = positive_regrets / sum_positive_regrets
            else: 
                # If all regrets are negative or zero, play uniform random
                legal_probs = np.ones(len(legal_actions)) / len(legal_actions)
                
            action_idx = np.random.choice(len(legal_actions), p=legal_probs)
            gym_action = self.format_gym_action(legal_actions[action_idx], obs_dict)
            
            # Opponent mutates the environment forward
            _, reward, terminated, _, _ = env.step(gym_action)
            
            if terminated:
                return reward[self.player_id]
            return self.cfr_traverse(env, iteration, depth + 1, reach_prob)

        # --- TRAINED AGENT'S TURN ---
        state_tensor = torch.FloatTensor(encode_state(obs_dict)).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            advantages = self.advantage_net(state_tensor)
            advantages = advantages[0].cpu().numpy()
        # Regret Matching
        advantages_masked = np.zeros(self.num_actions)
        for a in legal_actions:
            advantages_masked[a] = max(advantages[a], 0)
            
        if sum(advantages_masked) > 0:
            strategy = advantages_masked / sum(advantages_masked)
        else:
            strategy = np.zeros(self.num_actions)
            for a in legal_actions:
                strategy[a] = 1.0 / len(legal_actions)

        # Explore all valid branches
        action_values = np.zeros(self.num_actions)
        saved_state = env._get_state()
        for action_type in legal_actions:
            
            gym_action = self.format_gym_action(action_type, obs_dict)
            _, reward, terminated, _, _ = env.step(gym_action)
            
            if terminated:
                action_values[action_type] = reward[self.player_id]
            else:
                next_reach = reach_prob * strategy[action_type]
                action_values[action_type] = self.cfr_traverse(env, iteration, depth + 1, next_reach)
            env._set_state(saved_state)
        
        # Compute EV and Regrets
        ev = np.dot(strategy, action_values)
        for action_type in legal_actions:
            regret = action_values[action_type] - ev
            max_abs_val = max(abs(max(action_values)), abs(min(action_values)), 1.0)
            weighted_regret = (regret / max_abs_val)
            if abs(regret) > 200: # Adjust this threshold based on your stack sizes
                print(f"WARNING: Massive Regret Detected! Action: {action_type}, Regret: {regret}")
                print(f"Action Values: {action_values}, EV: {ev}")
            priority = abs(weighted_regret) + 0.01
            #priority = 1.0
            self.advantage_memory.add(
                encode_state(obs_dict), action_type, weighted_regret, iteration,
                priority
            )

        # 5. Save the SMOOTHED strategy to memory
        self.strategy_memory.append(
            encode_state(obs_dict), strategy, iteration, reach_prob
        )
        
        # Log this occasionally
        return ev

    def train_advantage_network(self, batch_size=128, epochs=3, beta_start=0.4, beta_end=1.0):
        if len(self.advantage_memory) < batch_size: return 0
        self.advantage_net.train()
        total_loss = 0
        progress = min(1.0, self.iteration_count / 500)
        #warm_restart_modifier = .5 + min(1.0, (self.iteration_count % warm_restart_steps)/warm_restart_steps) * .5
        beta = (beta_start + progress * (beta_end - beta_start))
        self.metrics['pos_regret_ratio'] = 0.0
        self.metrics['pred_magnitude'] = 0.0
        self.metrics['action_loss'] = 0.0
        for epoch in range(epochs):
            batch_tuple, indices, weights = self.advantage_memory.sample(batch_size, beta=beta)

            # 2. Unpack the tuple into the individual arrays
            batch_s, batch_a, batch_r, batch_i = batch_tuple

            # 3. Wrap them directly into PyTorch tensors (ZERO overhead!)
            obs_tensors = torch.FloatTensor(batch_s).to(self.device)
            action_type_tensors = torch.LongTensor(batch_a).to(self.device)
            regret_tensors = torch.FloatTensor(batch_r).to(self.device)
            #weight_tensors = torch.FloatTensor(weights).to(self.device)
            iteration_tensors = torch.FloatTensor(batch_i).to(self.device)
            
            iter_weights = iteration_tensors / (iteration_tensors.mean() + 1e-8)
            #combined_weights = weight_tensors * iter_weights
            action_advantages = self.advantage_net(obs_tensors)
            predicted_regrets = action_advantages.gather(1, action_type_tensors.unsqueeze(1)).squeeze(1)
            
            action_loss = F.smooth_l1_loss(predicted_regrets, regret_tensors, reduction='none')
            weighted_action_loss = (action_loss * iter_weights).mean()
        
            combined_loss = weighted_action_loss.clone()
            
            self.optimizer.zero_grad()
            combined_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.advantage_net.parameters(), max_norm=1.0)
            self.optimizer.step()
            with torch.no_grad():
                per_errors = torch.abs(predicted_regrets - regret_tensors).cpu().numpy()
                for i, idx in enumerate(indices):
                    self.advantage_memory.update_priority(idx, per_errors[i] + 0.01)

                pos_ratio = (predicted_regrets > 0).float().mean().item()
                pred_magnitude = action_advantages.abs().mean().item()
            total_loss += combined_loss.item()
            self.metrics['action_loss'] += weighted_action_loss.item()
            self.metrics['pos_regret_ratio'] += pos_ratio
            self.metrics['pred_magnitude'] += pred_magnitude

        self.metrics['pos_regret_ratio'] = self.metrics['pos_regret_ratio'] / epochs
        self.metrics['pred_magnitude'] = self.metrics['pred_magnitude'] / epochs
        self.metrics['action_loss'] = self.metrics['action_loss'] / epochs
        return total_loss / epochs

    def train_strategy_network(self, batch_size=128, epochs=3):
        if len(self.strategy_memory) < batch_size: return 0
        self.strategy_net.train()
        total_loss = 0
        self.metrics['strategy_entropy'] = 0.0
        for _ in range(epochs):
            batch_s, batch_p, batch_i, batch_rp = self.strategy_memory.sample(batch_size)

            # 2. Wrap them directly in PyTorch tensors
            obs_tensors = torch.FloatTensor(batch_s).to(self.device)
            strategy_tensors = torch.FloatTensor(batch_p).to(self.device)
            iteration_tensors = torch.FloatTensor(batch_i).to(self.device)
            reach_tensors = torch.FloatTensor(batch_rp).to(self.device)

            weights = iteration_tensors / (iteration_tensors.mean() + 1e-8)
            #weights = weights * reach_tensors
            weights = weights / (weights.mean() + 1e-8)
            action_logits = self.strategy_net(obs_tensors)
            legal_mask = (strategy_tensors > 0).float()
            masked_logits = action_logits + (1 - legal_mask) * -1e9
            log_probs = F.log_softmax(masked_logits, dim=1)
            per_sample_loss = -torch.sum(strategy_tensors * log_probs, dim=1)
            combined_loss = (per_sample_loss * weights).mean()
            
            self.strategy_optimizer.zero_grad()
            combined_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.strategy_net.parameters(), max_norm=1.0)
            self.strategy_optimizer.step()
            total_loss += combined_loss.item()
            with torch.no_grad():
            # Any action that has a target probability > 0 was a legal action in that state
                mask = (strategy_tensors > 0).float()
            
            # Call the function
                current_entropy = calculate_entropy(action_logits, mask)
                
                # Log it to your metrics
                self.metrics['strategy_entropy'] += current_entropy
        self.metrics['strategy_entropy'] = self.metrics['strategy_entropy'] / epochs
        return total_loss / epochs
    def choose_action(self, obs_tuple):
        obs_dict = obs_tuple[self.player_id]
        legal_actions = self.get_unaliased_legal_actions(obs_dict)
        if not legal_actions: return (0, 0, -1, -1) # Failsafe
            
        state_tensor = torch.FloatTensor(encode_state(obs_dict)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.strategy_net(state_tensor)
            probs = F.softmax(logits, dim=1)[0].cpu().numpy()
        
        # 1. Extract ONLY the probabilities for the legal actions
        filtered_probs = [probs[a] for a in legal_actions]
        
        # 2. Re-normalize just in case of tiny PyTorch rounding errors
        sum_probs = sum(filtered_probs)
        normalized_probs = [p / sum_probs for p in filtered_probs]
        
        # 🚨 3. Play the NASH EQUILIBRIUM! Sample proportionally!
        best_idx = np.random.choice(len(legal_actions), p=normalized_probs)
        
        best_action = legal_actions[best_idx]
        return self.format_gym_action(best_action, obs_dict)
    """
    def choose_action(self, obs_tuple):
        obs_dict = obs_tuple[self.player_id]
        legal_actions = self.get_unaliased_legal_actions(obs_dict)
        if not legal_actions: return (0, 0, -1, -1) 
            
        state_tensor = torch.FloatTensor(encode_state(obs_dict)).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # 🚨 1. USE THE ADVANTAGE NETWORK
            logits = self.advantage_net(state_tensor)
            raw_regrets = logits[0].cpu().numpy()
        
        # 2. Filter for legal actions
        legal_regrets = [raw_regrets[a] for a in legal_actions]
        
        # 🚨 3. GREEDY REGRET: Pick the action with the highest regret value
        best_idx = np.argmax(legal_regrets)
        best_action = legal_actions[best_idx]
        
        return self.format_gym_action(best_action, obs_dict)
    """
    def predict_strategy(self, obs_dict, legal_actions):
        """Fast inference for when this agent is acting as the opponent."""
        state_tensor = torch.FloatTensor(encode_state(obs_dict)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.strategy_net(state_tensor)
            probs = F.softmax(logits, dim=1)[0].cpu().numpy()
            
        legal_probs = np.array([probs[a] for a in legal_actions])
        if np.sum(legal_probs) > 0:
            legal_probs /= np.sum(legal_probs)
        else:
            legal_probs = np.ones(len(legal_actions)) / len(legal_actions)
            
        action_type = np.random.choice(legal_actions, p=legal_probs)
        return action_type

    def save_model(self, path_prefix):
        """Save the model to disk."""
        model_path = _resolve_model_save_path(path_prefix, self.iteration_count)
        torch.save({
            'iteration': self.iteration_count,
            'advantage_net': self.advantage_net.state_dict(), # Fixed from obs_dict()
            'strategy_net': self.strategy_net.state_dict(),   # Fixed from obs_dict()
            'min_bet_size': self.min_bet_size,
            'max_bet_size': self.max_bet_size
        }, model_path)
        
    def load_model(self, path):
        """Load the model from disk."""
        checkpoint = torch.load(path, map_location=self.device)
        self.iteration_count = checkpoint['iteration']
        self.advantage_net.load_state_dict(checkpoint['advantage_net'], strict=False)
        self.strategy_net.load_state_dict(checkpoint['strategy_net'], strict=False)
        
        # Load bet size bounds if available in the checkpoint
        if 'min_bet_size' in checkpoint:
            self.min_bet_size = checkpoint['min_bet_size']
        if 'max_bet_size' in checkpoint:
            self.max_bet_size = checkpoint['max_bet_size']

    def reset_adv(self):
        self.freeze_streams()
                
        self.advantage_net.action_head.reset_parameters()
        self.advantage_net.sizing_head[0].reset_parameters()
        self.advantage_net.sizing_head[2].reset_parameters()
        self.optimizer = optim.Adam(self.advantage_net.parameters(), lr=1e-6, weight_decay=1e-5)

    def freeze_streams(self):
        for name, p in self.advantage_net.named_parameters():
            if "stream" in name: # Adjust name filtering as needed
                p.requires_grad = False


    def cfr_traverse_parallel(self, env, iteration, adv_queue, strat_queue, opponent_agent=None, depth=0):
        current_player = env.acting_agent
        
        obs_tuple = env._get_obs(winner=None)[0] 
        obs_dict = obs_tuple[current_player]
        legal_actions = self.get_legal_action_types(obs_dict)
        
        if not legal_actions:
            return 0.0

        if current_player != self.player_id:
            if opponent_agent is None:
                action_type = random.choice(legal_actions)
                bet_multiplier = 1.0
                gym_action = self.format_gym_action(action_type, obs_dict, bet_multiplier)
                
            else:
                state_tensor = torch.FloatTensor(encode_state(obs_dict)).unsqueeze(0).to(opponent_agent.device)
                with torch.no_grad():
                    logits = opponent_agent.strategy_net(state_tensor)
                    probs = F.softmax(logits, dim=1)[0].cpu().numpy()
                legal_probs = np.array([probs[a] for a in legal_actions])
                if np.sum(legal_probs) > 0: legal_probs = legal_probs / np.sum(legal_probs)
                else: legal_probs = np.ones(len(legal_actions)) / len(legal_actions)
                action_idx = np.random.choice(len(legal_actions), p=legal_probs)
                gym_action = self.format_gym_action(legal_actions[action_idx], obs_dict)
            
            # Opponent mutates the environment forward
            _, reward, terminated, _, _ = env.step(gym_action)
            
            if terminated:
                return reward[self.player_id]
            return self.cfr_traverse(env, iteration, opponent_agent, depth + 1)

        # --- TRAINED AGENT'S TURN ---
        state_tensor = torch.FloatTensor(encode_state(obs_dict)).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            advantages = self.advantage_net(state_tensor)
            advantages = advantages[0].cpu().numpy()
        # Regret Matching
        advantages_masked = np.zeros(self.num_actions)
        for a in legal_actions:
            advantages_masked[a] = max(advantages[a], 0)
            
        if sum(advantages_masked) > 0:
            strategy = advantages_masked / sum(advantages_masked)
        else:
            strategy = np.zeros(self.num_actions)
            for a in legal_actions:
                strategy[a] = 1.0 / len(legal_actions)

        # Explore all valid branches
        action_values = np.zeros(self.num_actions)
        saved_state = env._get_state()

        for action_type in legal_actions:
            
            gym_action = self.format_gym_action(action_type, obs_dict)
            _, reward, terminated, _, _ = env.step(gym_action)
            
            if terminated:
                action_values[action_type] = reward[self.player_id]
            else:
                action_values[action_type] = self.cfr_traverse(env, iteration, opponent_agent, depth + 1)
            env._set_state(saved_state)

        # Compute EV and Regrets
        ev = sum(strategy[a] * action_values[a] for a in legal_actions)
        # --- MODIFIED MEMORY HANDLING ---
        for action_type in legal_actions:
            regret = action_values[action_type] - ev
            weighted_regret = (regret / 200)
            priority = abs(weighted_regret) + 0.01
            
            # Put the tuple into the multiprocessing queue instead of self.memory
            adv_queue.put((
                (encode_state(obs_dict), action_type, weighted_regret, iteration), 
                priority
            ))
            
        strat_queue.put((
            encode_state(obs_dict), strategy, iteration
        ))

        return ev