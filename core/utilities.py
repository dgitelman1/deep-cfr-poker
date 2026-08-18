def get_top_k_keep_indices(obs_dict, k=3):
    my_cards = [c for c in obs_dict["my_cards"] if c != -1]
    comm_cards = [c for c in obs_dict["community_cards"] if c != -1]
    
    if len(my_cards) == 5 and len(comm_cards) >= 3:
        board = comm_cards[:3] 
        pot_size = max(1, obs_dict.get("pot_size", obs_dict["my_bet"] + obs_dict["opp_bet"]))
        
        # List to store (EV, index_1, index_2)
        evaluated_hands = []
        
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
                    if unique_ranks[4] - unique_ranks[0] == 4:
                        is_straight = True
                    elif unique_ranks == [0, 1, 2, 3, 8]:
                        is_straight = True
                        
                # --- 2. ASSIGN BASE VALUES ---
                if is_flush and is_straight: base_hand_value = 8000
                elif freqs == [3, 2]: base_hand_value = 6000
                elif is_flush: base_hand_value = 5000
                elif is_straight: base_hand_value = 4000
                elif freqs == [3, 1, 1]: base_hand_value = 3000
                elif freqs == [2, 2, 1]: base_hand_value = 2000
                elif freqs == [2, 1, 1, 1]: base_hand_value = 1000
                else: base_hand_value = 0
                
                # --- 3. EVALUATE DRAWS ---
                hit_prob = 0.0
                if base_hand_value >= 2000:
                    hit_prob = 1.0
                if base_hand_value < 4000:
                    if max_suit == 4:
                        hit_prob = max(hit_prob, 0.468) 
                    
                    temp_ranks = unique_ranks.copy()
                    if 8 in temp_ranks:
                        temp_ranks.insert(0, -1)
                        
                    if len(temp_ranks) >= 4:
                        for r_idx in range(len(temp_ranks) - 3):
                            span = temp_ranks[r_idx+3] - temp_ranks[r_idx]
                            if span == 3:
                                if temp_ranks[r_idx] == -1 or temp_ranks[r_idx+3] == 8:
                                    hit_prob = max(hit_prob, 0.299)
                                else:
                                    hit_prob = max(hit_prob, 0.544)
                            elif span == 4:
                                hit_prob = max(hit_prob, 0.299)
                
                # --- 4. CALCULATE IMMEDIATE POT EV ---
                combination_ev = (hit_prob * pot_size * 100) + base_hand_value + (top_pair_rank * 10) + sum(ranks)
                
                # Save the combination to our list
                evaluated_hands.append((combination_ev, i, j))
                
        # Sort by EV in descending order (highest EV first)
        evaluated_hands.sort(key=lambda x: x[0], reverse=True)
        
        # Extract just the (keep_1, keep_2) tuples for the top K hands
        top_k_indices = [(hand[1], hand[2]) for hand in evaluated_hands[:k]]
        return top_k_indices
        
    else:
        # Fallback if pre-flop or invalid: just return the first K simple combinations
        return [(0, 1), (0, 2), (1, 2)][:k]

import numpy as np
def get_betting_stats(strategy_buffer):
    # strategy_buffer contains (obs, features, strategy_vector, ...)
    strategies = np.array([item[2] for item in strategy_buffer])
    
    # Average probability assigned to each raise tier
    avg_raise_probs = np.mean(strategies[:, 3:7], axis=0) 
    
    print(f"Min: {avg_raise_probs[0]:.2f}, Half: {avg_raise_probs[1]:.2f}, "
          f"Pot: {avg_raise_probs[2]:.2f}, All-In: {avg_raise_probs[3]:.2f}")
    
