# deep_cfr.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from collections import deque
import copy
from core.model import PokerNetwork, encode_state, VERBOSE, set_verbose
from enum import Enum

KEEP_PAIRS = [(i, j) for i in range(5) for j in range(i + 1, 5)]
NUM_DISCARD_CLASSES = len(KEEP_PAIRS)  

class ActionType(Enum):
    FOLD = 0
    RAISE = 1
    CHECK = 2
    CALL = 3
    DISCARD = 4
    INVALID = 5

def _resolve_model_save_path(path_prefix, iteration):
    """Accept either a path prefix or an explicit .pt filename."""
    if str(path_prefix).endswith(".pt"):
        return str(path_prefix)
    return f"{path_prefix}_iteration_{iteration}.pt"

class PrioritizedMemory:
    """Enhanced memory buffer with prioritized experience replay."""
    def __init__(self, capacity, alpha=0.6):
        """
        Initialize memory buffer with prioritized experience replay.
        
        Args:
            capacity: Maximum number of experiences to store
            alpha: Controls how much prioritization is used (0 = no prioritization, 1 = full prioritization)
        """
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = []
        self.position = 0
        self._max_priority = 1.0  # Initial max priority for new experiences
        
    def add(self, experience, priority=None):
        """
        Add a new experience to memory with its priority.
        
        Args:
            experience: Tuple of (obs, opponent_features, action_type, bet_size, regret)
            priority: Optional explicit priority value (defaults to max priority if None)
        """
        if priority is None:
            priority = self._max_priority
            
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
            self.priorities.append(priority ** self.alpha)
        else:
            # Replace the oldest entry
            self.buffer[self.position] = experience
            self.priorities[self.position] = priority ** self.alpha
            
        self.position = (self.position + 1) % self.capacity
        
    def sample(self, batch_size, beta=0.4):
        """
        Sample a batch of experiences based on their priorities.
        
        Args:
            batch_size: Number of experiences to sample
            beta: Controls importance sampling correction (0 = no correction, 1 = full correction)
                 Should be annealed from ~0.4 to 1 during training
                 
        Returns:
            Tuple of (samples, indices, importance_sampling_weights)
        """
        if len(self.buffer) < batch_size:
            # If we don't have enough samples, return all with equal weights
            return self.buffer, list(range(len(self.buffer))), np.ones(len(self.buffer))
        
        # Convert priorities to probabilities
        total_priority = sum(self.priorities)
        probabilities = [p / total_priority for p in self.priorities]
        
        # Sample indices based on priorities
        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities, replace=False)
        samples = [self.buffer[idx] for idx in indices]
        
        # Calculate importance sampling weights
        weights = []
        
        for idx in indices:
            # P(i) = p_i^α / sum_k p_k^α
            # weight = (1/N * 1/P(i))^β = (N*P(i))^-β
            sample_prob = self.priorities[idx] / total_priority
            weight = (len(self.buffer) * sample_prob) ** -beta
            weights.append(weight)
        #weights = np.ones(batch_size, dtype=np.float32)
        #weights = np.ones(batch_size, dtype=np.float32)

        # Normalize weights to have maximum weight = 1
        # This ensures we only scale down updates, never up
        max_weight = max(weights)
        weights = [w / max_weight for w in weights]
        
        return samples, indices, np.array(weights, dtype=np.float32)
        
    def update_priority(self, index, priority):
        """
        Update the priority of an experience.
        
        Args:
            index: Index of the experience to update
            priority: New priority value (before alpha adjustment)
        """
        # Clip priority to be positive
        priority = max(1e-8, priority)
        
        # Keep track of max priority for new experience initialization
        self._max_priority = max(self._max_priority, priority)
        
        # Store alpha-adjusted priority
        self.priorities[index] = priority ** self.alpha
        
    def __len__(self):
        """Return the current size of the memory."""
        return len(self.buffer)
        
    def get_memory_stats(self):
        """Get statistics about the current memory buffer."""
        if not self.priorities:
            return {"min": 0, "max": 0, "mean": 0, "median": 0, "size": 0}
            
        raw_priorities = [p ** (1/self.alpha) for p in self.priorities]
        return {
            "min": min(raw_priorities),
            "max": max(raw_priorities),
            "mean": sum(raw_priorities) / len(raw_priorities),
            "median": sorted(raw_priorities)[len(raw_priorities) // 2],
            "size": len(self.buffer)
        }
    def save(self, filepath):
        """
        Saves the memory buffer state to disk.
        """
        state = {
            'capacity': self.capacity,
            'alpha': self.alpha,
            'buffer': self.buffer,
            'priorities': self.priorities,
            'position': self.position,
            '_max_priority': self._max_priority
        }
        torch.save(state, filepath)

    def load(self, filepath):
        """
        Loads the memory buffer state from disk.
        """
        state = torch.load(filepath)
        self.capacity = state['capacity']
        self.alpha = state['alpha']
        self.buffer = state['buffer']
        self.priorities = state['priorities']
        self.position = state['position']
        self._max_priority = state['_max_priority']
class DeepCFRAgent:
    def __init__(self, player_id=0, memory_size=300000, device='cpu'):
        self.player_id = player_id
        self.device = device
        self.num_actions = 5 # Fold(0), Raise(1), Check(2), Call(3), Discard(4)
        
        self.advantage_net = PokerNetwork().to(device)
        self.strategy_net = PokerNetwork().to(device)
        
        self.optimizer = optim.Adam(self.advantage_net.parameters(), lr=1e-6, weight_decay=1e-5)
        self.strategy_optimizer = optim.Adam(self.strategy_net.parameters(), lr=0.00005, weight_decay=1e-5)
        
        self.advantage_memory = PrioritizedMemory(memory_size)
        self.strategy_memory = deque(maxlen=memory_size)
        
        self.iteration_count = 0
        self.min_bet_size = 0.1
        self.max_bet_size = 3.0
        self.metrics = {
        'action_loss': 0.0,
        'bet_loss': 0.0,
        'discard_loss': 0.0,
        }

    def get_legal_action_types(self, obs):
        valid_actions_mask = obs['valid_actions']

        legal_actions = [i for i, is_valid in enumerate(valid_actions_mask) if is_valid == 1]

        # If checking is an option, folding is a purely dominated action. Prune it.
        if (ActionType.CHECK.value in legal_actions or ActionType.DISCARD.value in legal_actions) and ActionType.FOLD.value in legal_actions:
            legal_actions.remove(ActionType.FOLD.value)

        return legal_actions

    def format_gym_action(self, action_type, obs_dict, bet_multiplier=1.0, discard_logits = None):
        """Formats the Tuple exactly like the working RL Agent."""
        raise_amount = 0
        keep_1, keep_2 = 0, 0 # Match the RL agent's default for non-discards
        
        if action_type == 1: # RAISE
            min_r = obs_dict["min_raise"]
            max_r = obs_dict["max_raise"]
            pot = max(1, obs_dict.get("pot_size", obs_dict["my_bet"] + obs_dict["opp_bet"]))
            
            # Safeguard against NaN gradients
            if np.isnan(bet_multiplier): bet_multiplier = 1.0
            
            adjustment = 1.2 if obs_dict["street"] >= 2 else 1.0
            #adjustment = .5
            adjusted_multiplier = max(self.min_bet_size, min(self.max_bet_size, bet_multiplier * adjustment))
            
            target_raise = int(pot * adjusted_multiplier)
            
            # Exact clamping used in the working RL agent
            raise_amount = int(max(min(target_raise, max_r), min_r))
            
        elif action_type == 4: # DISCARD
            if discard_logits is not None:
                discards = torch.distributions.Categorical(logits=discard_logits)
                discard_idx = discards.sample()
                keep_1, keep_2 = KEEP_PAIRS[discard_idx.item() % NUM_DISCARD_CLASSES]
            else:
                my_cards = [c for c in obs_dict["my_cards"] if c != -1]
                comm_cards = [c for c in obs_dict["community_cards"] if c != -1]
                
                if len(my_cards) == 5 and len(comm_cards) >= 3:
                    board = comm_cards[:3] 
                    pot_size = max(1, obs_dict.get("pot_size", obs_dict["my_bet"] + obs_dict["opp_bet"]))
                    
                    best_ev = -1
                    keep_1, keep_2 = 0, 1
                    
                    for i in range(5):
                        for j in range(i+1, 5):
                            five_cards = board + [my_cards[i], my_cards[j]]
                            
                            # Ranks: 2=0, 3=1, 4=2, 5=3, 6=4, 7=5, 8=6, 9=7, A=8
                            ranks = sorted([c % 9 for c in five_cards])
                            suits = [c // 9 for c in five_cards]
                            
                            # Base Facts
                            unique_ranks = sorted(list(set(ranks)))
                            suit_counts = [suits.count(s) for s in set(suits)]
                            max_suit = max(suit_counts)
                            
                            counts = {r: ranks.count(r) for r in set(ranks)}
                            freqs = sorted(counts.values(), reverse=True)
                            
                            paired_ranks = [r for r, count in counts.items() if count >= 2]
                            top_pair_rank = max(paired_ranks) if paired_ranks else 0
                            
                            # --- 1. IDENTIFY MADE HANDS ---
                            is_flush = (max_suit == 5)
                            is_straight = False
                            
                            if len(unique_ranks) == 5:
                                # Standard Straight (e.g., 5-6-7-8-9 or 6-7-8-9-A)
                                if unique_ranks[4] - unique_ranks[0] == 4:
                                    is_straight = True
                                # Low Straight Exception (A-2-3-4-5) -> Ranks: 0,1,2,3,8
                                elif unique_ranks == [0, 1, 2, 3, 8]:
                                    is_straight = True
                                    
                            # --- 2. ASSIGN BASE VALUES (Following the Rulebook) ---
                            if is_flush and is_straight: base_hand_value = 8000 # Straight Flush
                            elif freqs == [3, 2]: base_hand_value = 6000        # Full House
                            elif is_flush: base_hand_value = 5000               # Flush
                            elif is_straight: base_hand_value = 4000            # Straight
                            elif freqs == [3, 1, 1]: base_hand_value = 3000     # Three of a Kind
                            elif freqs == [2, 2, 1]: base_hand_value = 2000     # Two Pair
                            elif freqs == [2, 1, 1, 1]: base_hand_value = 1000  # One Pair
                            else: base_hand_value = 0                           # High Card
                            
                            # --- 3. EVALUATE DRAWS (Only if we don't have a premium made hand) ---
                            hit_prob = 0.0
                            if base_hand_value >= 2000:
                                hit_prob = 1.0
                            if base_hand_value < 4000: # Only chase draws if worse than a Straight
                                
                                # Flush Draw
                                if max_suit == 4:
                                    hit_prob = max(hit_prob, 0.468) 
                                    
                                # Straight Draw
                                temp_ranks = unique_ranks.copy()
                                if 8 in temp_ranks:
                                    temp_ranks.insert(0, -1) # Add A-low (-1) to easily check spans
                                    
                                if len(temp_ranks) >= 4:
                                    for r_idx in range(len(temp_ranks) - 3):
                                        span = temp_ranks[r_idx+3] - temp_ranks[r_idx]
                                        if span == 3:
                                            # Touches the "Wall" (A-2-3-4 or 7-8-9-A) -> Only 3 outs
                                            if temp_ranks[r_idx] == -1 or temp_ranks[r_idx+3] == 8:
                                                hit_prob = max(hit_prob, 0.299) # Gutshot
                                            else:
                                                hit_prob = max(hit_prob, 0.544) # OESD (6 outs)
                                        elif span == 4:
                                            hit_prob = max(hit_prob, 0.299) # Gutshot (hole in middle)
                            
                            # --- 4. CALCULATE IMMEDIATE POT EV ---
                            combination_ev = (hit_prob * pot_size * 100) + base_hand_value + (top_pair_rank * 10) + sum(ranks)
                            
                            if combination_ev > best_ev:
                                best_ev = combination_ev
                                keep_1, keep_2 = i, j
                else:
                    keep_1, keep_2 = 0, 1
            
        return (action_type, raise_amount, keep_1, keep_2)

    def cfr_traverse(self, env, iteration, opponent_agent=None, depth=0):
        current_player = env.acting_agent
        
        # 🚨 Use the synced legal actions directly from the env
            
        obs_tuple = env._get_obs(winner=None)[0] 
        obs_dict = obs_tuple[current_player]
        legal_actions = self.get_legal_action_types(obs_dict)
        if not legal_actions:
            return 0.0
        # --- OPPONENT'S TURN ---
        if current_player != self.player_id:
            if opponent_agent is None:
                action_type = random.choice(legal_actions)
                bet_multiplier = 1.0
                gym_action = self.format_gym_action(action_type, obs_dict, bet_multiplier)
            else:
                gym_action = opponent_agent.act(obs_dict, 0, False, False, None)
            
            # Opponent mutates the environment forward
            _, reward, terminated, _, _ = env.step(gym_action)
            
            if terminated:
                return reward[self.player_id]
            return self.cfr_traverse(env, iteration, opponent_agent, depth + 1)

        # --- TRAINED AGENT'S TURN ---
        state_tensor = torch.FloatTensor(encode_state(obs_dict)).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            advantages, bet_size_pred, discard_logits = self.advantage_net(state_tensor)
            advantages = advantages[0].cpu().numpy()
            bet_size_multiplier = bet_size_pred[0][0].item()
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
        
        for action_type in legal_actions:
            env_branch = copy.deepcopy(env)
            
            gym_action = self.format_gym_action(action_type, obs_dict, bet_size_multiplier, discard_logits)
            _, reward, terminated, _, _ = env_branch.step(gym_action)
            
            if terminated:
                action_values[action_type] = reward[self.player_id]
            else:
                action_values[action_type] = self.cfr_traverse(env_branch, iteration, opponent_agent, depth + 1)

        # Compute EV and Regrets
        ev = sum(strategy[a] * action_values[a] for a in legal_actions)
        
        for action_type in legal_actions:
            regret = action_values[action_type] - ev
            
            max_abs_val = max(abs(max(action_values)), abs(min(action_values)), 1.0)
            weighted_regret = (regret / max_abs_val) * (np.sqrt(iteration) if iteration > 1 else 1.0)
            if abs(regret) > 200: # Adjust this threshold based on your stack sizes
                print(f"WARNING: Massive Regret Detected! Action: {action_type}, Regret: {regret}")
                print(f"Action Values: {action_values}, EV: {ev}")
            priority = abs(weighted_regret) + 0.01
            #priority = 1.0
            bet_val = bet_size_multiplier if action_type == 1 else 0.0
            discard_probs = F.softmax(discard_logits, dim=-1)[0].cpu().numpy()
            self.advantage_memory.add(
                (encode_state(obs_dict), np.zeros(20), action_type, bet_val, weighted_regret, discard_probs),
                priority
            )
            
        self.strategy_memory.append((
            encode_state(obs_dict), np.zeros(20), strategy, 
            bet_size_multiplier if 1 in legal_actions else 0.0, iteration, discard_probs if 4 in legal_actions else np.zeros_like(discard_probs)
        ))
        legal_strategy = np.array([strategy[a] for a in legal_actions])
        legal_strategy = legal_strategy[legal_strategy > 0] # Avoid log(0)
        entropy = -np.sum(legal_strategy * np.log2(legal_strategy))
        
        # Log this occasionally
        if iteration % 10 == 0:
            print(f"Iter {iteration} | Depth {depth} | Entropy: {entropy:.2f} | EV: {ev:.2f}")
        return ev

    def train_advantage_network(self, batch_size=128, epochs=3, beta_start=0.4, beta_end=1.0, warm_restart_steps = 100):
        if len(self.advantage_memory) < batch_size: return 0
        self.advantage_net.train()
        total_loss = 0
        dynamic_bet_weight = .5
        discard_weight = .05

        progress = min(1.0, self.iteration_count / 1500)
        #warm_restart_modifier = .5 + min(1.0, (self.iteration_count % warm_restart_steps)/warm_restart_steps) * .5
        beta = (beta_start + progress * (beta_end - beta_start))
        
        for epoch in range(epochs):
            batch, indices, weights = self.advantage_memory.sample(batch_size, beta=beta)
            obss, opponent_features, action_types, bet_sizes, regrets, discards = zip(*batch)
            
            obs_tensors = torch.FloatTensor(np.array(obss)).to(self.device)
            opponent_feature_tensors = torch.FloatTensor(np.array(opponent_features)).to(self.device)
            action_type_tensors = torch.LongTensor(np.array(action_types)).to(self.device)
            bet_size_tensors = torch.FloatTensor(np.array(bet_sizes)).unsqueeze(1).to(self.device)
            regret_tensors = torch.FloatTensor(np.array(regrets)).to(self.device)
            weight_tensors = torch.FloatTensor(weights).to(self.device)
            discard_tensors = torch.FloatTensor(np.array(discards)).to(self.device)
            
            action_advantages, bet_size_preds, discard_preds = self.advantage_net(obs_tensors, opponent_feature_tensors)
            predicted_regrets = action_advantages.gather(1, action_type_tensors.unsqueeze(1)).squeeze(1)
            
            action_loss = F.mse_loss(predicted_regrets, regret_tensors, reduction='none')
            weighted_action_loss = (action_loss * weight_tensors).mean()
            
            raise_mask = (action_type_tensors == 1) # RAISE is index 1
            discard_mask = (action_type_tensors == 4) 
            combined_loss = weighted_action_loss.clone()
            if torch.any(raise_mask):
                all_bet_losses = F.mse_loss(bet_size_preds, bet_size_tensors, reduction='none')
                masked_bet_losses = all_bet_losses * raise_mask.float().unsqueeze(1)
                raise_count = raise_mask.sum().item()
                if raise_count > 0:
                    weighted_bet_size_loss = dynamic_bet_weight * ((masked_bet_losses.squeeze() * weight_tensors).sum() / raise_count)
                    combined_loss = weighted_action_loss + weighted_bet_size_loss
            if torch.any(discard_mask):
                discard_loss = F.cross_entropy(
                    discard_preds[discard_mask], 
                    discard_tensors[discard_mask], 
                    reduction='none'
                )
                weighted_discard_loss = discard_weight * (discard_loss * weight_tensors[discard_mask]).mean()
                combined_loss = combined_loss + (weighted_discard_loss)
            
            self.optimizer.zero_grad()
            combined_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.advantage_net.parameters(), max_norm=0.5)
            self.optimizer.step()
            with torch.no_grad():
                new_action_errors = F.mse_loss(predicted_regrets, regret_tensors, reduction='none')
                combined_errors = new_action_errors.clone()
                if torch.any(raise_mask):
                    raise_indices = torch.where(raise_mask)[0]
                    combined_errors = new_action_errors.clone()
                    for i in raise_indices:
                        bet_err = F.mse_loss(bet_size_preds[i], bet_size_tensors[i], reduction='mean')
                        combined_errors[i] += dynamic_bet_weight * bet_err
                if torch.any(discard_mask):
                    discard_indices = torch.where(discard_mask)[0]
                    for i in discard_indices:
                        disc_err = F.cross_entropy(discard_preds[i], discard_tensors[i], reduction='mean')
                        combined_errors[i] += dynamic_bet_weight * disc_err
                
                for i, idx in enumerate(indices):
                    self.advantage_memory.update_priority(idx, combined_errors.cpu().numpy()[i] + 0.01)
            total_loss += combined_loss.item()
            self.metrics['action_loss'] += weighted_action_loss.item()
            if torch.any(raise_mask):
                self.metrics['bet_loss'] += weighted_bet_size_loss.item()
            if torch.any(discard_mask):
                self.metrics['discard_loss'] += weighted_discard_loss.item()
        return total_loss / epochs

    def train_strategy_network(self, batch_size=128, epochs=3):
        if len(self.strategy_memory) < batch_size: return 0
        self.strategy_net.train()
        total_loss = 0
        dynamic_bet_weight = .5
        discard_weight = .05
        for _ in range(epochs):
            batch = random.sample(self.strategy_memory, batch_size)
            obss, opponent_features, strategies, bet_sizes, iterations, discards = zip(*batch)
            
            obs_tensors = torch.FloatTensor(np.array(obss)).to(self.device)
            strategy_tensors = torch.FloatTensor(np.array(strategies)).to(self.device)
            bet_size_tensors = torch.FloatTensor(np.array(bet_sizes)).unsqueeze(1).to(self.device)
            iteration_tensors = torch.FloatTensor(iterations).to(self.device).unsqueeze(1)
            discard_tensors = torch.FloatTensor(np.array(discards)).to(self.device)
            
            weights = iteration_tensors / torch.sum(iteration_tensors)
            action_logits, bet_size_preds, discard_preds = self.strategy_net(obs_tensors)
            predicted_strategies = F.softmax(action_logits, dim=1)
            
            action_loss = -torch.sum(weights * torch.sum(strategy_tensors * torch.log(predicted_strategies + 1e-8), dim=1))
            
            raise_mask = (strategy_tensors[:, 1] > 0) # RAISE is index 1
            discard_mask = (strategy_tensors[:, 4] > 0)
            combined_loss = action_loss
            if torch.any(raise_mask) > 0:
                raise_indices = torch.nonzero(raise_mask).squeeze(1)
                raise_bet_preds = bet_size_preds[raise_indices]
                raise_bet_targets = bet_size_tensors[raise_indices]
                raise_weights = weights[raise_indices]
                
                bet_size_loss = F.mse_loss(raise_bet_preds, raise_bet_targets, reduction='none')
                weighted_bet_size_loss = torch.sum(raise_weights.squeeze() * bet_size_loss.squeeze())
                combined_loss = action_loss + dynamic_bet_weight * weighted_bet_size_loss
            if torch.any(discard_mask):
                discard_indices = torch.nonzero(discard_mask).squeeze(1)
                discard_preds_masked = discard_preds[discard_indices]
                discard_targets_masked = discard_tensors[discard_indices]
                discard_weights = weights[discard_indices]
                
                discard_loss = F.cross_entropy(
                    discard_preds_masked, 
                    discard_targets_masked, 
                    reduction='none'
                )
                
                weighted_discard_loss = torch.sum(discard_weights.squeeze() * discard_loss.squeeze())
                combined_loss = combined_loss + (discard_weight * weighted_discard_loss)
            
            self.strategy_optimizer.zero_grad()
            combined_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.strategy_net.parameters(), max_norm=0.5)
            self.strategy_optimizer.step()
            total_loss += combined_loss.item()
            
        return total_loss / epochs

    def choose_action(self, obs_tuple):
        obs_dict = obs_tuple[self.player_id]
        legal_actions = self.get_legal_action_types(obs_dict)
        if not legal_actions: return (0, 0, -1, -1) # Failsafe
            
        state_tensor = torch.FloatTensor(encode_state(obs_dict)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits, bet_size_pred, discard_pred = self.strategy_net(state_tensor)
            probs = F.softmax(logits, dim=1)[0].cpu().numpy()
            bet_size_multiplier = bet_size_pred[0][0].item()
        legal_probs = np.array([probs[a] for a in legal_actions])
        if np.sum(legal_probs) > 0: legal_probs = legal_probs / np.sum(legal_probs)
        else: legal_probs = np.ones(len(legal_actions)) / len(legal_actions)
        action_idx = np.random.choice(len(legal_actions), p=legal_probs)
        return self.format_gym_action(legal_actions[action_idx], obs_dict, bet_size_multiplier)
    
    def predict_strategy(self, obs_dict, legal_actions):
        """Fast inference for when this agent is acting as the opponent."""
        state_tensor = torch.FloatTensor(encode_state(obs_dict)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits, bet_size_pred = self.strategy_net(state_tensor)
            probs = F.softmax(logits, dim=1)[0].cpu().numpy()
            bet_size_multiplier = bet_size_pred[0][0].item()
            
        legal_probs = np.array([probs[a] for a in legal_actions])
        if np.sum(legal_probs) > 0:
            legal_probs /= np.sum(legal_probs)
        else:
            legal_probs = np.ones(len(legal_actions)) / len(legal_actions)
            
        action_type = np.random.choice(legal_actions, p=legal_probs)
        return action_type, bet_size_multiplier

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


    def clear_metrics(self):
        self.metrics = {
        'action_loss': 0.0,
        'bet_loss': 0.0,
        'discard_loss': 0.0,
        }
