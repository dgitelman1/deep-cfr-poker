
# Pre-computed bitmasks for the 6 possible straights in a 9-rank deck (2-9, A)
# Ranks are 0-8: 2, 3, 4, 5, 6, 7, 8, 9, A
STRAIGHT_MASKS = [
    0x1F0,  # 6-7-8-9-A (Decimal: 496)
    0x0F8,  # 5-6-7-8-9 (Decimal: 248)
    0x07C,  # 4-5-6-7-8 (Decimal: 124)
    0x03E,  # 3-4-5-6-7 (Decimal: 62)
    0x01F,  # 2-3-4-5-6 (Decimal: 31)
    0x10F   # A-2-3-4-5 (Decimal: 271) - The Ace plays low
]

def get_highest_n_bits(mask, n):
    """Returns the integer value of keeping only the highest 'n' bits of a mask."""
    result = 0
    count = 0
    for i in range(8, -1, -1): # Scan ranks from A (8) down to 2 (0)
        if (mask & (1 << i)) != 0:
            result |= (1 << i)
            count += 1
            if count == n:
                break
    return result

def bit_count(n):
    """Kernighan's algorithm to count set bits (popcount)."""
    return n.bit_count()


def evaluate_hand_27(cards_mask):
    # 1. Isolate the suits
    diamonds = cards_mask & 0x1FF
    hearts = (cards_mask >> 9) & 0x1FF
    spades = (cards_mask >> 18) & 0x1FF

    # 2. Frequencies
    ranks_mask = diamonds | hearts | spades
    pairs_mask = (diamonds & hearts) | (diamonds & spades) | (hearts & spades)
    trips_mask = diamonds & hearts & spades
    
    # Pre-sort all ranks present for kicker calculation
    all_ranks = sorted([r for r in range(9) if (ranks_mask & (1 << r))], reverse=True)

    # 3. Check for Flushes
    flush_mask = 0
    if bin(diamonds).count('1') >= 5: flush_mask = diamonds
    elif bin(hearts).count('1') >= 5: flush_mask = hearts
    elif bin(spades).count('1') >= 5: flush_mask = spades

    # 4. Straight Flush (Category 8)
    if flush_mask:
        for mask in STRAIGHT_MASKS:
            if (flush_mask & mask) == mask:
                return (8 << 24) | get_straight_high_card(mask)

    # 5. Full House (Category 7)
    if trips_mask > 0:
        t_rank = get_highest_rank(trips_mask)
        p_mask = pairs_mask & ~(1 << t_rank)
        if p_mask > 0:
            p_rank = get_highest_rank(p_mask)
            return (7 << 24) | (t_rank << 4) | p_rank

    # 6. Flush (Category 6)
    if flush_mask:
        f_ranks = sorted([r for r in range(9) if (flush_mask & (1 << r))], reverse=True)
        return (6 << 24) | (f_ranks[0] << 16) | (f_ranks[1] << 12) | (f_ranks[2] << 8) | (f_ranks[3] << 4) | f_ranks[4]

    # 7. Straight (Category 5)
    for mask in STRAIGHT_MASKS:
        if (ranks_mask & mask) == mask:
            return (5 << 24) | get_straight_high_card(mask)

    # 8. Three of a Kind (Category 4)
    if trips_mask > 0:
        t_rank = get_highest_rank(trips_mask)
        k = [r for r in all_ranks if r != t_rank]
        return (4 << 24) | (t_rank << 8) | (k[0] << 4) | k[1]

    # 9. Two Pair (Category 3)
    if bin(pairs_mask).count('1') >= 2:
        p_list = sorted([r for r in range(9) if (pairs_mask & (1 << r))], reverse=True)
        h_p, l_p = p_list[0], p_list[1]
        k = [r for r in all_ranks if r != h_p and r != l_p][0]
        return (3 << 24) | (h_p << 8) | (l_p << 4) | k

    # 10. Pair (Category 2)
    if pairs_mask > 0:
        p_rank = get_highest_rank(pairs_mask)
        k = [r for r in all_ranks if r != p_rank]
        #print(f"Evaluating Pair: {p_rank}, Available Kickers: {k}")
        return (2 << 24) | (p_rank << 12) | (k[0] << 8) | (k[1] << 4) | k[2]

    # 11. High Card (Category 1)
    return (1 << 24) | (all_ranks[0] << 16) | (all_ranks[1] << 12) | (all_ranks[2] << 8) | (all_ranks[3] << 4) | all_ranks[4]

# --- Helper Functions ---
def get_highest_rank(mask):
    if mask == 0: return -1
    return mask.bit_length() - 1

def get_straight_high_card(mask):
    # Special case for A-2-3-4-5 straight (Ace is Rank 8)
    if mask == 0b100001111: # A, 2, 3, 4, 5
        return 3 # 5 is the high card
    return get_highest_rank(mask)