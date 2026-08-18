import torch
import torch.nn as nn
import numpy as np

VERBOSE = False

def set_verbose(verbose_mode):
    global VERBOSE
    VERBOSE = verbose_mode
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        # Two linear layers per block is the industry standard
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.relu1 = nn.LeakyReLU(0.05)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.LeakyReLU(0.05)

    def forward(self, x):
        identity = x  # Save the original input!
        
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        
        # 🚨 THE SKIP CONNECTION 🚨
        # We add the original input back to the mutated output before the final activation
        out = out + identity 
        
        out = self.relu2(out)
        return out

KEEP_PAIRS = [(i, j) for i in range(5) for j in range(i + 1, 5)]
NUM_DISCARD_CLASSES = len(KEEP_PAIRS)  
class PokerNetwork(nn.Module):
    # 🔴 UPGRADED: input_size is now 123, num_actions is now 5 (Fold, Raise, Check, Call, Discard)
    def __init__(self, input_size=123, hidden_size=256, num_actions=11):
        super().__init__()
        self.card_stream = nn.Sequential(
            nn.Linear(108, hidden_size),
            nn.LeakyReLU(0.05)
        )
        
        # Stream 2: Process the continuous betting/street features
        self.bet_stream = nn.Sequential(
            nn.Linear(input_size-108, 64),
            nn.LeakyReLU(0.05)
        )
        self.merger = nn.Sequential(
            nn.Linear(hidden_size + 64, hidden_size),
            nn.LeakyReLU(0.05)
        )
        # Merge them
        self.core = nn.Sequential(
            ResidualBlock(hidden_size),
            ResidualBlock(hidden_size),
            ResidualBlock(hidden_size)
        )
        self.action_head = nn.Linear(hidden_size, num_actions)
    
    def forward(self, x):
        card_features = x[:, :108]
        state_features = x[:, 108:]

        card_out = self.card_stream(card_features)
        state_out = self.bet_stream(state_features)

        merged_features = torch.cat([card_out, state_out], dim=1)
        merged_features = self.merger(merged_features)
        features = self.core(merged_features)
        action_logits = self.action_head(features)
        
        # Output bet size between 0.0 and 3.0 (representing pot multiplier)
        
        return action_logits

def get_canonical_suit_mapping(obs_dict, num_ranks=9, num_suits=3):
    """
    Creates a deterministic mapping of original suits to canonical suits
    based on the exact locations and ranks of cards in each suit.
    """
    # Signature format: [my_cards, comm_cards, my_discarded, opp_discarded]
    suit_signatures = {s: [[], [], [], []] for s in range(num_suits)}
    
    def add_cards_to_signature(cards, category_index):
        for c in cards:
            if c != -1:
                suit = c // num_ranks
                rank = c % num_ranks
                suit_signatures[suit][category_index].append(rank)

    # 1. Map cards to their specific location buckets
    add_cards_to_signature(obs_dict["my_cards"], 0)
    add_cards_to_signature(obs_dict["community_cards"], 1)
    add_cards_to_signature(obs_dict["my_discarded_cards"], 2)
    add_cards_to_signature(obs_dict["opp_discarded_cards"], 3)

    signatures = []
    for suit, loc_lists in suit_signatures.items():
        # 2. Create a unique, order-independent tuple for this suit
        sig_tuple = (
            tuple(sorted(loc_lists[0], reverse=True)),
            tuple(sorted(loc_lists[1], reverse=True)),
            tuple(sorted(loc_lists[2], reverse=True)),
            tuple(sorted(loc_lists[3], reverse=True))
        )
        signatures.append((sig_tuple, suit))

    # 3. Sort signatures descending. 
    signatures.sort(reverse=True)

    suit_mapping = {}
    for canonical_id, (sig, original_suit) in enumerate(signatures):
        suit_mapping[original_suit] = canonical_id

    return suit_mapping

def encode_state_cards_only(obs_dict):
    """
    A simplified version of your encoder focusing just on the card canonicalization.
    """
    NUM_RANKS = 9
    NUM_SUITS = 3  # Based on your DECK_SIZE of 27

    # Pass the entire obs_dict so the canonicalizer knows where cards came from
    suit_mapping = get_canonical_suit_mapping(obs_dict, NUM_RANKS, NUM_SUITS)

    def canonicalize(cards):
        canon_cards = []
        for c in cards:
            if c == -1:
                continue
            suit = c // NUM_RANKS
            rank = c % NUM_RANKS
            canonical_suit = suit_mapping[suit]
            new_card = (canonical_suit * NUM_RANKS) + rank
            canon_cards.append(new_card)
        # Sort to ensure order-independence within the arrays themselves
        return sorted(canon_cards)

    return {
        "canon_my_cards": canonicalize(obs_dict["my_cards"]),
        "canon_comm": canonicalize(obs_dict["community_cards"]),
        "canon_my_discard": canonicalize(obs_dict["my_discarded_cards"]),
        "canon_opp_discard": canonicalize(obs_dict["opp_discarded_cards"])
    }

def encode_state(obs_dict):
    """
    Converts the Gym PokerEnv observation dictionary into a flat 1D NumPy array.
    Canonicalizes suits dynamically to drastically reduce the state space.
    Total dimensions: (4 * 27) + 4 + 1 + 5 + 5 = 123
    """
    DECK_SIZE = 27
    NUM_RANKS = 9
    MAX_PLAYER_BET = 100.0

    card_states = encode_state_cards_only(obs_dict)
    canon_my_cards = card_states['canon_my_cards']
    canon_comm = card_states['canon_comm']
    canon_my_discard = card_states['canon_my_discard']
    canon_opp_discard = card_states['canon_opp_discard']
    # --- 2. Build One-Hot Card Vectors ---
    my_cards_vec = np.zeros(DECK_SIZE, dtype=np.float32)
    comm_cards_vec = np.zeros(DECK_SIZE, dtype=np.float32)
    my_discard_vec = np.zeros(DECK_SIZE, dtype=np.float32)
    opp_discard_vec = np.zeros(DECK_SIZE, dtype=np.float32)

    if canon_my_cards: my_cards_vec[canon_my_cards] = 1.0
    if canon_comm: comm_cards_vec[canon_comm] = 1.0
    if canon_my_discard: my_discard_vec[canon_my_discard] = 1.0
    if canon_opp_discard: opp_discard_vec[canon_opp_discard] = 1.0
    
    # --- 3. Encode Street (One-hot, 4 elements) ---
    street_vec = np.zeros(4, dtype=np.float32)
    street_vec[obs_dict["street"]] = 1.0

    # --- 4. Encode Positional Advantage (1 element) ---
    blind_pos = obs_dict.get("blind_position", obs_dict["acting_agent"])
    is_blind = 1.0 if blind_pos == obs_dict["acting_agent"] else -1.0
    position_vec = np.array([is_blind], dtype=np.float32)

    # --- 5. Normalize Bets and Total Pot (5 elements) ---
    my_bet = obs_dict["my_bet"]
    opp_bet = obs_dict["opp_bet"]
    pot_size = obs_dict.get("pot_size", my_bet + opp_bet)

    bets_vec = np.array([
        my_bet / MAX_PLAYER_BET,
        opp_bet / MAX_PLAYER_BET,
        pot_size / (MAX_PLAYER_BET * 2), 
        obs_dict["min_raise"] / MAX_PLAYER_BET,
        obs_dict["max_raise"] / MAX_PLAYER_BET
    ], dtype=np.float32)
    
    # --- 6. Valid Actions Mask (5 elements) ---
    valid_actions_vec = np.array(obs_dict["valid_actions"], dtype=np.float32)

    # Concatenate everything into a single flat array of size 123
    return np.concatenate([
        my_cards_vec, comm_cards_vec, my_discard_vec, opp_discard_vec,
        street_vec, position_vec, bets_vec, valid_actions_vec
    ])