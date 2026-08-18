import random
from bit_utils import evaluate_hand_27
FOLD_BIT = 1 << 0
CALL_BIT = 1 << 1
MIN_BIT  = 1 << 2
HALF_BIT = 1 << 3
MAX_BIT  = 1 << 4
DISCARD_BITS = 0x7FE0
VALID_DISCARDS_CONSTANT = [False, False, False, False, False] + [True] * 10
class FastPokerState:
    # UPDATED: Added p0_folded and p1_folded
    __slots__ = [
        'p0_cards', 'p1_cards', 'comm_cards', 'deck',
        'p0_discarded', 'p1_discarded',
        'street', 'acting_player', 
        'p0_bet', 'p1_bet',
        'p0_folded', 'p1_folded', 'history', 'raises_this_street'
    ]

    def __init__(self):
        self.p0_cards = 0
        self.p1_cards = 0
        self.comm_cards = 0
        self.deck = 0x7FFFFFF 
        self.p0_discarded = 0
        self.p1_discarded = 0
        
        self.street = 0
        self.acting_player = 0
        self.p0_bet = 1 
        self.p1_bet = 2

        # Initialize the flags
        self.p0_folded = False
        self.p1_folded = False

        self.history = []

        self.raises_this_street = 0
        
    def clone(self):
        new_state = FastPokerState.__new__(FastPokerState)
        new_state.p0_cards = self.p0_cards
        new_state.p1_cards = self.p1_cards
        new_state.comm_cards = self.comm_cards
        new_state.deck = self.deck
        new_state.p0_discarded = self.p0_discarded
        new_state.p1_discarded = self.p1_discarded
        new_state.street = self.street
        new_state.acting_player = self.acting_player
        new_state.p0_bet = self.p0_bet
        new_state.p1_bet = self.p1_bet
        
        # UPDATED: Copy the folded flags
        new_state.p0_folded = self.p0_folded
        new_state.p1_folded = self.p1_folded
        return new_state
    def get_obs(self):
        # Determine perspective
        if self.acting_player == 0:
            my_cards_raw = self._bitboard_to_list(self.p0_cards)
            my_discard_raw = self._bitboard_to_list(self.p0_discarded)
            opp_discard_raw = self._bitboard_to_list(self.p1_discarded)
            my_bet, opp_bet = self.p0_bet, self.p1_bet
        else:
            my_cards_raw = self._bitboard_to_list(self.p1_cards)
            my_discard_raw = self._bitboard_to_list(self.p1_discarded)
            opp_discard_raw = self._bitboard_to_list(self.p0_discarded)
            my_bet, opp_bet = self.p1_bet, self.p0_bet

        comm_raw = self._bitboard_to_list(self.comm_cards)

        def pad(l, length):
            return tuple(l + [-1] * (length - len(l)))

        mask_int = self.get_valid_actions_mask_int()
        #valid_actions_list = [(mask_int & (1 << i)) > 0 for i in range(15)]

        return {
            "street": int(self.street),
            "acting_agent": int(self.acting_player),
            "my_cards": pad(my_cards_raw, 5),
            "community_cards": pad(comm_raw, 5),
            "my_bet": int(my_bet),
            "opp_bet": int(opp_bet),
            "my_discarded_cards": pad(my_discard_raw, 3),
            "opp_discarded_cards": pad(opp_discard_raw, 3), # Now revealed immediately
            "min_raise": int(self.calculate_min_raise()),
            "max_raise": 100,
            "valid_actions": mask_int
        }
    def _bitboard_to_list(self, bitboard):
        """Helper to convert a bitmask integer back to a list of card indices."""
        return [i for i in range(27) if (bitboard & (1 << i))]
    
    def calculate_min_raise(self):
        """
        Calculates the minimum total bet required to raise.
        Formula: Current_Max_Bet + (Current_Max_Bet - Previous_Max_Bet)
        """
        b0, b1 = self.p0_bet, self.p1_bet
        
        # If the pot is unraised (e.g. both players at 2 chips)
        if b0 == b1:
            return min(b0 + 2, 100) # BB is 2
            
        # Standard raise increment
        current_max = max(b0, b1)
        current_min = min(b0, b1)
        raise_increment = max(current_max - current_min, 2)
        
        return min(current_max + raise_increment, 100)
    def get_valid_actions_mask_int(self):
        # 1. DISCARD PHASE (Only on Street 1)
        if self.street == 1:
            my_discarded = self.p0_discarded if self.acting_player == 0 else self.p1_discarded
            # If I haven't discarded yet, I MUST discard. Betting is locked.
            if my_discarded == 0:
                return DISCARD_BITS 
            
            # If I HAVE discarded, but the opponent hasn't, the game should 
            # have already passed the turn to them. If it's my turn and I've 
            # discarded, we must be in the betting phase.
            # However, betting ONLY starts once BOTH have discarded.
            if self.p0_discarded == 0 or self.p1_discarded == 0:
                return 0 # Failsafe: No valid actions if waiting for other's discard

        # 2. BETTING PHASE (All streets, but only after discards on Street 1)
        mask = FOLD_BIT | CALL_BIT
        
        b_me = self.p0_bet if self.acting_player == 0 else self.p1_bet
        
        # Check raise cap (4 raises) and total bet limit (100)
        if self.raises_this_street < 4 and b_me < 100:
            min_r = self.calculate_min_raise()
            if min_r <= 100:
                mask |= (MIN_BIT | HALF_BIT | MAX_BIT)
                
        return mask

    def is_game_over(self):
        """Returns True if someone folded or we reached the end of the River."""
        if getattr(self, 'p0_folded', False) or getattr(self, 'p1_folded', False):
            return True
        return self.street > 3

    def get_reward(self):
        """
        Evaluates the winner and returns the chip delta for Player 0.
        """
        # If someone folded, the other player wins the pot
        if getattr(self, 'p0_folded', False):
            return -self.p0_bet
        if getattr(self, 'p1_folded', False):
            return self.p1_bet
            
        # At Showdown: Evaluate both bitboards
        # (Assumes evaluate_hand_27 is imported/defined in your file)
        score_0 = evaluate_hand_27(self.p0_cards | self.comm_cards)
        score_1 = evaluate_hand_27(self.p1_cards | self.comm_cards)
        
        if score_0 > score_1:
            return self.p1_bet  # Player 0 wins Player 1's bet
        elif score_1 > score_0:
            return -self.p0_bet # Player 0 loses their bet
        else:
            return 0            # Tie, no chips exchanged

    def step(self, env_action, debug=False):
        # 1. Take Snapshot and push to history
        snapshot = (
            self.p0_cards, self.p1_cards, self.comm_cards, self.deck,
            self.p0_discarded, self.p1_discarded,
            self.street, self.acting_player,
            self.p0_bet, self.p1_bet,
            self.p0_folded, self.p1_folded, self.raises_this_street
        )
        self.history.append(snapshot)
        
        action_type, amt, k1, k2 = env_action
        old_player = self.acting_player
        old_street = self.street
        # 2. Handle FOLD
        if action_type == 0:
            if self.acting_player == 0: self.p0_folded = True
            else: self.p1_folded = True
            if debug:
                print(f"DEBUG: P{old_player} FOLDED. Game Over.")
            return

        # 3. Handle RAISE / CALL / CHECK
        prev_bet_equal = (self.p0_bet == self.p1_bet)
        
        if action_type == 1: # RAISE
            self.raises_this_street += 1
            if self.acting_player == 0: self.p0_bet = amt
            else: self.p1_bet = amt
            if debug:
                print(f"DEBUG: P{old_player} RAISED to {amt}")
        elif action_type in [2, 3]: # CHECK / CALL
            if self.acting_player == 0: self.p0_bet = self.p1_bet
            else: self.p1_bet = self.p0_bet
            if debug:
                verb = "CHECKED" if prev_bet_equal else "CALLED"
                print(f"DEBUG: P{old_player} {verb}")
        
        # 4. Handle DISCARD
        elif action_type == 4:
            if self.acting_player == 0:
                cards_list = self._bitboard_to_list(self.p0_cards)
                keep_mask = (1 << cards_list[k1]) | (1 << cards_list[k2])
                self.p0_discarded = self.p0_cards ^ keep_mask
                self.p0_cards = keep_mask
            else:
                cards_list = self._bitboard_to_list(self.p1_cards)
                if k1 >= len(cards_list) or k2 >= len(cards_list):
                    print(f"\n--- STATE ERROR ---")
                    print(f"k1: {k1}, k2: {k2}")
                    print(f"cards_list: {cards_list} (length {len(cards_list)})")
                    print(f"p1_cards bitboard: {self.p1_cards}")
                    print(f"env_action passed: {env_action}")
                    raise IndexError("Caught invalid card indices before crash.")
                keep_mask = (1 << cards_list[k1]) | (1 << cards_list[k2])
                self.p1_discarded = self.p1_cards ^ keep_mask
                self.p1_cards = keep_mask
            if debug:
                print(f"DEBUG: P{old_player} DISCARDED")

        # 5. STREET TRANSITION LOGIC
        bets_equal = (self.p0_bet == self.p1_bet)

        # Discard phase logic (Street 1)
        if self.street == 1 and action_type == 4:
            # If both have discarded, betting round on the Flop starts
            if self.p0_discarded != 0 and self.p1_discarded != 0:
                self.acting_player = 1 # BB acts first for betting
                if debug:
                    print(f"DEBUG: Both discarded. Starting Flop Betting. P1 acts first.")
            else:
                # Wait for the next player to discard
                self.acting_player = 1 - self.acting_player
                if debug:
                    print(f"DEBUG: Waiting for P{self.acting_player} to discard.")
            return

        # Betting round closure
        if bets_equal:
            # SPECIAL CASE: Pre-flop (Street 0), P0 calls. 
            # Both bets are 2, but P1 (BB) MUST have their option.
            if self.street == 0 and old_player == 0 and self.p0_bet == 2:
                self.acting_player = 1
                if debug: print("DEBUG: P0 called BB. P1 (BB) gets their option.")
            
            # CASE: Standard Call-around or Check-around
            elif not prev_bet_equal or action_type == 2:
                self._advance_street()
                if debug: print(f"DEBUG: Street Advanced! {old_street} -> {self.street}.")
            
            # CASE: Failsafe (shouldn't hit often)
            else:
                self.acting_player = 1 - self.acting_player
        else:
            # Bets are not equal, betting MUST continue
            self.acting_player = 1 - self.acting_player
            if debug: print(f"DEBUG: Betting continues. P{self.acting_player} acts.")

    def _advance_street(self):
        self.street += 1
        self.raises_this_street = 0
        
        # 1. Fast bitboard extraction (Brian Kernighan's Algorithm)
        remaining_indices = []
        n = self.deck
        while n:
            lsb = n & -n
            remaining_indices.append(lsb.bit_length() - 1)
            n &= n - 1
            
        # 2. Draw cards and create a single bitmask
        if self.street == 1: # Deal Flop (3 cards)
            c1, c2, c3 = random.sample(remaining_indices, 3)
            drawn_mask = (1 << c1) | (1 << c2) | (1 << c3)
            
        elif self.street in [2, 3]: # Deal Turn/River (1 card)
            card = random.choice(remaining_indices)
            drawn_mask = 1 << card
            
        else:
            drawn_mask = 0
            
        # 3. Apply the mask to the board and deck in one atomic step
        if drawn_mask:
            self.comm_cards |= drawn_mask
            self.deck &= ~drawn_mask
            
        # 4. 🚨 Rule: Post-flop/discard ALWAYS starts with Big Blind (Player 1)
        self.acting_player = 1
    
    def reset(self):
        """
        Instantly resets the hand, shuffling and dealing 5 cards to each player.
        Returns the initial observation dictionary.
        """
        # 1. Reset betting and game state
        self.street = 0
        self.acting_player = 0  # Player 0 (Small Blind) acts first pre-flop
        self.p0_bet = 1
        self.p1_bet = 2
        
        self.p0_folded = False
        self.p1_folded = False
        
        self.p0_discarded = 0
        self.p1_discarded = 0
        self.comm_cards = 0
        
        # 2. Reset the full deck (27 bits set to 1 = 0x7FFFFFF)
        self.deck = 0x7FFFFFF
        self.p0_cards = 0
        self.p1_cards = 0
        self.raises_this_street = 0
        self.history.clear()
        # 3. Shuffle and Deal (10 unique indices from 0 to 26)
        dealt_cards = random.sample(range(27), 10)
        
        # Deal first 5 cards to Player 0
        self.p0_cards = sum(1 << c for c in dealt_cards[:5])
            
        self.p1_cards = sum(1 << c for c in dealt_cards[5:])
        self.deck &= ~(self.p0_cards | self.p1_cards)

        return self.get_obs()
    def unstep(self):
        """
        Restores the environment to the exact state it was in before 
        the last call to step().
        """
        if not self.history:
            raise IndexError("Cannot unstep: History is empty.")
            
        # Unpack the last snapshot back into the state variables
        (
            self.p0_cards, self.p1_cards, self.comm_cards, self.deck,
            self.p0_discarded, self.p1_discarded,
            self.street, self.acting_player,
            self.p0_bet, self.p1_bet,
            self.p0_folded, self.p1_folded, self.raises_this_street
        ) = self.history.pop()
    