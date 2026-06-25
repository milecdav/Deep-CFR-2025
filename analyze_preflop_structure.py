#!/usr/bin/env python3
"""
Analyze preflop structure theoretically.

With MAX_N_RAISES_PER_ROUND[PREFLOP] = 2, and 2 players:
- Each player can act multiple times
- Actions: FOLD (f), CHECK/CALL (c), RAISE (r)
- Max 2 raises per round means max 2 raises total in preflop

Possible endings:
1. Immediate fold: f
2. Check/call to end: cc
3. Raise then fold: rf
4. Raise then call: rc
5. Check/call, raise, fold: crf
6. Check/call, raise, call: crc
7. Check/call, raise, raise, fold: crrf (if allowed)
8. Check/call, raise, raise, call: crrc (if allowed)
... and more complex sequences

Let's enumerate all possible sequences programmatically.
"""

def enumerate_preflop_endings(max_raises=1, first_action_no_call=True):
    """
    Enumerate all possible preflop action sequences that end the preflop round.
    
    Rules:
    - 2 players (SB acts first, then BB)
    - Actions: FOLD (f), CHECK/CALL (c), RAISE (r)
    - Max raises: max_raises per round (1 in this case, since BB counts as a raise)
    - FIRST_ACTION_NO_CALL: SB cannot call BB as first action (must fold or raise)
    - Sequence ends when: someone folds, or betting is complete (no more raises possible)
    """
    
    endings = set()
    
    def is_valid_sequence(seq):
        """Check if sequence is valid."""
        raises_count = seq.count('r')
        if raises_count > max_raises:
            return False
        # First action cannot be call if FIRST_ACTION_NO_CALL is True
        if first_action_no_call and len(seq) > 0 and seq[0] == 'c':
            return False
        return True
    
    def generate_sequences(max_length=10, current="", player=0):
        """Generate all valid sequences."""
        if len(current) >= max_length:
            return
        
        # If someone folded, sequence ends
        if 'f' in current:
            if is_valid_sequence(current):
                endings.add(current)
            return
        
        # Check if betting is complete (both players acted and no more raises possible)
        if len(current) >= 2:
            raises_count = current.count('r')
            # If we've had max raises and last action was call, betting is complete
            if raises_count == max_raises and current[-1] == 'c':
                if is_valid_sequence(current):
                    endings.add(current)
                return
            # If last two actions are both calls (cc), betting is complete
            # But wait - if FIRST_ACTION_NO_CALL, first action can't be 'c'
            # So 'cc' means: raise, then call (rc) or similar
            # Actually, 'cc' after a raise means: raise, call, call - but that's rc, c
            # Let me think: if SB raises, BB can call (rc), then betting is complete
            # If SB raises, BB raises (rr), then SB must call (rrc) - but that's 2 raises, not allowed
            # So after a raise, BB can only call or fold
            # After raise+call, betting is complete
            if len(current) >= 2 and current[-1] == 'c':
                # Check if this completes betting
                # If there was a raise and then a call, betting is complete
                if 'r' in current[:-1]:  # There was a raise before the call
                    if is_valid_sequence(current):
                        endings.add(current)
                    return
        
        # Generate next actions
        raises_count = current.count('r')
        
        # Fold ends immediately
        generate_sequences(max_length, current + 'f', 1 - player)
        
        # Call (but not as first action if FIRST_ACTION_NO_CALL)
        if not (first_action_no_call and len(current) == 0):
            generate_sequences(max_length, current + 'c', 1 - player)
        
        # Raise (if allowed)
        if raises_count < max_raises:
            generate_sequences(max_length, current + 'r', 1 - player)
    
    # Start with SB acting first
    generate_sequences(max_length=8)
    
    return sorted(endings, key=lambda x: (len(x), x))

if __name__ == "__main__":
    print("=" * 80)
    print("Theoretical Preflop Endings Analysis")
    print("=" * 80)
    print("\nWith MAX_N_RAISES_PER_ROUND[PREFLOP] = 2")
    print("(Note: Comment says 'is actually 1, but BB counts as a raise')")
    print("So effectively only 1 raise is allowed after the BB.")
    print("\nFIRST_ACTION_NO_CALL = True")
    print("(SB cannot call BB as first action - must fold or raise)")
    print("\nEnumerating all possible preflop action sequences...\n")
    
    # Only 1 raise allowed (BB counts as the first raise)
    # FIRST_ACTION_NO_CALL = True means SB can't call BB
    endings = enumerate_preflop_endings(max_raises=1, first_action_no_call=True)
    
    print(f"Found {len(endings)} unique preflop endings:\n")
    for i, ending in enumerate(endings, 1):
        print(f"  {i:2d}. {ending}")
    
    print("\n" + "=" * 80)
    print("Note: This is a theoretical enumeration.")
    print("The actual tree may have more endings due to:")
    print("  - Different betting patterns")
    print("  - All-in situations")
    print("  - Edge cases in the implementation")
    print("=" * 80)
