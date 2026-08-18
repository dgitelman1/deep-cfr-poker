"""Deep CFR self-play trainer for the bitboard poker prototype."""

import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm

from fastpoker import FastPokerState
from networks import RegretNetwork, StrategyNetwork, FastMemoryBuffer

DECK_SIZE = 27
MAX_PLAYER_BET = 100.0
INPUT_SIZE = 118

TOTAL_ITERATIONS = 1000
GAMES_PER_ITERATION = 500
BATCH_SIZE = 4096
TRAINING_STEPS = 1000
LEARNING_RATE = 0.0005
BUFFER_CAPACITY = 10_000_000
CHECKPOINT_EVERY = 10
CHECKPOINT_DIR = "checkpoints"

MASK_LUT = np.zeros((32768, 15), dtype=bool)
for _i in range(32768):
    for _j in range(15):
        if _i & (1 << _j):
            MASK_LUT[_i, _j] = True

DUMMY_TRANSLATION = np.arange(15, dtype=np.intp)

DISCARD_COMBOS = [
    (0, 1), (0, 2), (0, 3), (0, 4),
    (1, 2), (1, 3), (1, 4),
    (2, 3), (2, 4),
    (3, 4),
]

SUIT_PERMUTATIONS = [
    (0, 1, 2), (0, 2, 1),
    (1, 0, 2), (1, 2, 0),
    (2, 0, 1), (2, 1, 0),
]


def vectorize_observation(obs):
    """Converts the observation dict into a flat 118-dim NumPy array."""
    my_cards_vec = np.zeros(DECK_SIZE, dtype=np.float32)
    comm_cards_vec = np.zeros(DECK_SIZE, dtype=np.float32)
    my_discard_vec = np.zeros(DECK_SIZE, dtype=np.float32)
    opp_discard_vec = np.zeros(DECK_SIZE, dtype=np.float32)

    my_cards_vec[sorted([c for c in obs["my_cards"] if c != -1])] = 1.0
    comm_cards_vec[sorted([c for c in obs["community_cards"] if c != -1])] = 1.0
    my_discard_vec[sorted([c for c in obs["my_discarded_cards"] if c != -1])] = 1.0
    opp_discard_vec[sorted([c for c in obs["opp_discarded_cards"] if c != -1])] = 1.0

    street_vec = np.zeros(4, dtype=np.float32)
    street_vec[obs["street"]] = 1.0

    agent_vec = np.array([1.0 if obs["acting_agent"] == 1 else -1.0], dtype=np.float32)

    total_pot = obs["my_bet"] + obs["opp_bet"]
    bets_vec = np.array([
        obs["my_bet"] / MAX_PLAYER_BET,
        obs["opp_bet"] / MAX_PLAYER_BET,
        total_pot / (MAX_PLAYER_BET * 2),
        obs["min_raise"] / MAX_PLAYER_BET,
        obs["max_raise"] / MAX_PLAYER_BET,
    ], dtype=np.float32)

    return np.concatenate([
        my_cards_vec, comm_cards_vec, my_discard_vec, opp_discard_vec,
        street_vec, agent_vec, bets_vec,
    ])


def regret_matching(predicted_regrets, valid_mask_int):
    regrets = predicted_regrets.ravel()
    valid_mask = MASK_LUT[valid_mask_int]

    pos_regrets = np.maximum(regrets, 0.0)
    pos_regrets[~valid_mask] = 0.0
    sum_pos = pos_regrets.sum()

    if sum_pos > 0:
        pos_regrets /= sum_pos
        return pos_regrets

    valid_regrets = regrets[valid_mask]
    exp_regrets = np.exp(valid_regrets - np.max(valid_regrets))
    probs = exp_regrets / exp_regrets.sum()

    final_probs = np.zeros(15, dtype=np.float32)
    final_probs[valid_mask] = probs
    return final_probs


def map_index_to_env_action(idx, obs):
    min_r = int(obs.get("min_raise", 2))
    max_r = int(obs.get("max_raise", 100))
    my_bet = obs.get("my_bet", 0)
    opp_bet = obs.get("opp_bet", 0)

    if idx == 0:
        return (0, 0, -1, -1)

    if idx == 1:
        return (2 if my_bet == opp_bet else 3, 0, -1, -1)

    if idx in (2, 3, 4):
        if idx == 2:
            raise_amt = min_r
        elif idx == 3:
            amount_to_call = opp_bet - my_bet
            pot_after_call = my_bet + opp_bet + amount_to_call
            raise_amt = opp_bet + (pot_after_call // 2)
        else:
            raise_amt = max_r
        final_raise = int(max(min_r, min(raise_amt, max_r)))
        return (1, final_raise, -1, -1)

    combo_idx = idx - 5
    k1, k2 = DISCARD_COMBOS[combo_idx]
    return (4, 0, k1, k2)


def get_canonical_state_and_perm(state_np):
    best_state = None
    best_hash = None
    best_perm = None

    for p in SUIT_PERMUTATIONS:
        permuted_vec = state_np.copy()
        for offset in (0, 27, 54, 81):
            original_suits = (
                state_np[offset:offset + 9],
                state_np[offset + 9:offset + 18],
                state_np[offset + 18:offset + 27],
            )
            permuted_vec[offset:offset + 9] = original_suits[p[0]]
            permuted_vec[offset + 9:offset + 18] = original_suits[p[1]]
            permuted_vec[offset + 18:offset + 27] = original_suits[p[2]]

        h = permuted_vec.tobytes()
        if best_hash is None or h > best_hash:
            best_hash = h
            best_state = permuted_vec
            best_perm = p

    return best_state, best_perm


def get_action_translation_map(real_cards, perm):
    """Maps canonical action indices back to the real environment's action indices."""
    sorted_real = sorted(real_cards)
    canonical_cards = [(perm[c // 9] * 9) + (c % 9) for c in sorted_real]

    sorted_canon_with_orig_idx = sorted(enumerate(canonical_cards), key=lambda x: x[1])
    orig_idx_to_canon_idx = {
        orig_idx: canon_idx
        for canon_idx, (orig_idx, _) in enumerate(sorted_canon_with_orig_idx)
    }

    translation_map = list(range(15))
    for i, (k1, k2) in enumerate(DISCARD_COMBOS):
        real_action_idx = i + 5
        new_k1 = orig_idx_to_canon_idx[k1]
        new_k2 = orig_idx_to_canon_idx[k2]
        if new_k1 > new_k2:
            new_k1, new_k2 = new_k2, new_k1
        canon_combo_idx = DISCARD_COMBOS.index((new_k1, new_k2))
        translation_map[real_action_idx] = canon_combo_idx + 5

    return np.array(translation_map, dtype=np.intp)


class FastNumpyMLP:
    """Extracts weights from a PyTorch MLP for fast NumPy inference."""

    def __init__(self, pytorch_model):
        self.layers = []
        for module in pytorch_model.modules():
            if isinstance(module, nn.Linear):
                W = module.weight.detach().cpu().numpy().T
                b = module.bias.detach().cpu().numpy()
                self.layers.append(('linear', W, b))
            elif isinstance(module, nn.LeakyReLU):
                self.layers.append(('leaky_relu', module.negative_slope, None))
            elif isinstance(module, nn.ReLU):
                self.layers.append(('relu', None, None))
            elif isinstance(module, nn.Tanh):
                self.layers.append(('tanh', None, None))

    def __call__(self, x):
        out = x.astype(np.float32, copy=False)
        for layer_type, param1, _param2 in self.layers:
            if layer_type == 'linear':
                out = np.dot(out, param1) + _param2
            elif layer_type == 'leaky_relu':
                out = np.where(out > 0, out, out * param1)
            elif layer_type == 'relu':
                out = np.maximum(0, out)
            elif layer_type == 'tanh':
                out = np.tanh(out)
        return out


def external_sampling_mccfr(env, traversing_player, regret_net,
                            memory_buffer, strategy_buffer, iteration,
                            inference_cache=None):
    if inference_cache is None:
        inference_cache = {}

    if env.is_game_over():
        reward = env.get_reward()
        normalized_reward = reward / (MAX_PLAYER_BET * 2)
        return normalized_reward if traversing_player == 0 else -normalized_reward

    obs = env.get_obs()
    current_player = obs["acting_agent"]
    valid_actions_mask = obs["valid_actions"]
    if valid_actions_mask == 0:
        return 0.0

    is_discard_round = (obs["my_cards"][2] != -1)
    if not is_discard_round:
        valid_actions_mask &= 31
    if valid_actions_mask == 0:
        return 0.0

    raw_state_np = vectorize_observation(obs)
    state_np, perm = get_canonical_state_and_perm(raw_state_np)
    is_discard_round = (obs["my_cards"][2] != -1) and ((valid_actions_mask & 32736) > 0)

    real_mask_bool = MASK_LUT[valid_actions_mask]
    state_hash = state_np.tobytes()
    if state_hash in inference_cache:
        canonical_regrets = inference_cache[state_hash]
    else:
        canonical_regrets = regret_net(state_np)
        if obs["street"] == 0:
            inference_cache[state_hash] = canonical_regrets

    if is_discard_round:
        real_cards = obs["my_cards"][:5]
        translation_map = get_action_translation_map(real_cards, perm)
        real_regrets = canonical_regrets[translation_map]
    else:
        translation_map = DUMMY_TRANSLATION
        real_regrets = canonical_regrets

    real_action_probs = regret_matching(real_regrets, valid_actions_mask)

    if current_player != traversing_player:
        masked_probs = real_action_probs * real_mask_bool
        prob_sum = np.sum(masked_probs)
        if prob_sum > 0:
            masked_probs = masked_probs / prob_sum
        else:
            masked_probs = real_mask_bool / np.sum(real_mask_bool)
        chosen_action_idx = np.random.choice(15, p=masked_probs)
        env.step(map_index_to_env_action(chosen_action_idx, obs))
        utility = external_sampling_mccfr(
            env, traversing_player, regret_net,
            memory_buffer, strategy_buffer, iteration, inference_cache,
        )
        env.unstep()
        return utility

    action_utilities = np.zeros(15, dtype=np.float16)
    for action_idx in range(15):
        if not real_mask_bool[action_idx]:
            continue
        env.step(map_index_to_env_action(action_idx, obs))
        action_utilities[action_idx] = external_sampling_mccfr(
            env, traversing_player, regret_net,
            memory_buffer, strategy_buffer, iteration, inference_cache,
        )
        env.unstep()

    node_utility = np.dot(real_action_probs, action_utilities)
    real_target_regrets = action_utilities - node_utility
    real_target_regrets[~real_mask_bool] = 0.0

    canonical_target_regrets = np.zeros(15, dtype=np.float16)
    canonical_strategy = np.zeros(15, dtype=np.float16)
    canonical_target_regrets[translation_map] = real_target_regrets
    canonical_strategy[translation_map] = real_action_probs * iteration

    memory_buffer.push(state_np, canonical_target_regrets, valid_actions_mask)
    strategy_buffer.push(state_np, canonical_strategy, valid_actions_mask)

    return node_utility


def train_regret_network(memory_buffer, device):
    """Trains a fresh regret network from the reservoir of collected regrets."""
    regret_net = RegretNetwork(input_size=INPUT_SIZE).to(device)
    optimizer = optim.Adam(regret_net.parameters(), lr=LEARNING_RATE)
    bit_shifts = torch.tensor([1 << i for i in range(15)], device=device)

    total_loss = 0.0
    for _ in range(TRAINING_STEPS):
        states_np, targets_np, masks_np = memory_buffer.sample(BATCH_SIZE)

        states = torch.from_numpy(states_np).float().to(device)
        targets = torch.from_numpy(targets_np).float().to(device)

        masks_tensor = torch.as_tensor(masks_np, dtype=torch.int32, device=device).unsqueeze(1)
        valid_bool_mask = (masks_tensor & bit_shifts) > 0

        predictions = regret_net(states)
        squared_errors = (predictions - targets) ** 2
        masked_errors = squared_errors * valid_bool_mask
        loss = masked_errors.sum() / (valid_bool_mask.sum() + 1e-8)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(regret_net.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()

    return regret_net, optimizer, total_loss / TRAINING_STEPS


def train_strategy_network(strategy_buffer, device, batch_size=2048, epochs=30,
                           lr=0.001, save_path="strategy_weights.pt"):
    """Distills the time-averaged strategy buffer into the final strategy network."""
    strategy_net = StrategyNetwork(input_size=INPUT_SIZE).to(device)
    strategy_net.train()
    optimizer = optim.Adam(strategy_net.parameters(), lr=lr)
    criterion = nn.KLDivLoss(reduction='batchmean')
    bit_shifts = torch.tensor([1 << i for i in range(15)], device=device)

    if strategy_buffer.size < batch_size:
        print(f"Not enough strategy data to distill (size {strategy_buffer.size}).")
        return strategy_net

    batches_per_epoch = strategy_buffer.size // batch_size
    for epoch in range(epochs):
        epoch_loss = 0.0
        for _ in range(batches_per_epoch):
            states_np, targets_np, masks_np = strategy_buffer.sample(batch_size)

            row_sums = targets_np.sum(axis=1, keepdims=True)
            valid_rows = (row_sums > 0).flatten()
            if not np.any(valid_rows):
                continue

            states_np = states_np[valid_rows]
            targets_np = targets_np[valid_rows] / row_sums[valid_rows]
            masks_np = masks_np[valid_rows]

            states = torch.as_tensor(states_np, dtype=torch.float32, device=device)
            targets = torch.as_tensor(targets_np, dtype=torch.float32, device=device)
            masks_tensor = torch.as_tensor(masks_np, dtype=torch.int32, device=device).unsqueeze(1)
            valid_bool_mask = (masks_tensor & bit_shifts) > 0

            probs = strategy_net(states, valid_actions_mask=valid_bool_mask)
            log_probs = torch.log(probs.clamp_min(1e-9))
            loss = criterion(log_probs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        print(f"Distill epoch {epoch + 1:02d}/{epochs} | KL: {epoch_loss / batches_per_epoch:.6f}")

    torch.save(strategy_net.state_dict(), save_path)
    print(f"Saved strategy weights to {save_path}")
    return strategy_net


def save_checkpoint(iteration, regret_net, optimizer, memory_buffer, strategy_buffer):
    ckpt_dir = os.path.join(CHECKPOINT_DIR, f"iter_{iteration}")
    os.makedirs(ckpt_dir, exist_ok=True)
    torch.save(regret_net.state_dict(), os.path.join(ckpt_dir, "regret_net.pt"))
    torch.save(optimizer.state_dict(), os.path.join(ckpt_dir, "regret_opt.pt"))
    memory_buffer.save(ckpt_dir, "regret_buffer")
    strategy_buffer.save(ckpt_dir, "strategy_buffer")
    print(f"Checkpoint saved to {ckpt_dir}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(1)
    print(f"Training Deep CFR on device: {device}")

    regret_net = RegretNetwork(input_size=INPUT_SIZE).to(device)
    memory_buffer = FastMemoryBuffer(capacity=BUFFER_CAPACITY, state_dim=INPUT_SIZE)
    strategy_buffer = FastMemoryBuffer(capacity=BUFFER_CAPACITY * 2, state_dim=INPUT_SIZE)
    env = FastPokerState()

    for iteration in tqdm(range(1, TOTAL_ITERATIONS + 1)):
        regret_net.eval()
        numpy_regret_net = FastNumpyMLP(regret_net)
        inference_cache = {}

        for _ in tqdm(range(GAMES_PER_ITERATION), leave=False):
            env.reset()
            traversing_player = np.random.choice([0, 1])
            external_sampling_mccfr(
                env, traversing_player, numpy_regret_net,
                memory_buffer, strategy_buffer, iteration, inference_cache,
            )

        regret_net, optimizer, avg_loss = train_regret_network(memory_buffer, device)
        print(f"Iteration {iteration} | Avg masked MSE: {avg_loss:.4f} | Buffer: {memory_buffer.size}")

        if iteration % CHECKPOINT_EVERY == 0:
            save_checkpoint(iteration, regret_net, optimizer, memory_buffer, strategy_buffer)

    train_strategy_network(strategy_buffer, device)


if __name__ == "__main__":
    main()
