#!/usr/bin/env python3
"""
Calculate the size of the public tree for Flop5Holdem.

The public tree includes:
- All possible action sequences (FOLD, CHECK/CALL, RAISE)
- All possible board card combinations
- Only public information (no private cards)

Usage:
    ml Python/3.11.5-GCCcore-13.2.0
    source venv/bin/activate
    python calculate_flop5_tree_size.py
"""

import sys
import os

# Add PokerRL to path
sys.path.insert(0, 'PokerRL-2025')

from PokerRL.game.games import Flop5Holdem
from PokerRL.game.Poker import Poker
from PokerRL.game.wrappers import FlatLimitPokerEnvBuilder
from PokerRL.game._.tree.PublicTree import PublicTree
from PokerRL.game._.look_up_table import LutHolderHoldem

def calculate_tree_size():
    """Build the public tree and count nodes."""
    
    # Create environment arguments
    env_args = Flop5Holdem.ARGS_CLS(
        n_seats=2,
        starting_stack_sizes_list=[Flop5Holdem.DEFAULT_STACK_SIZE for _ in range(2)],
        use_simplified_headsup_obs=True,
    )
    
    # Create LUT holder (needs env_cls, not rules)
    lut_holder = LutHolderHoldem(Flop5Holdem)
    
    # Create environment builder
    env_bldr = FlatLimitPokerEnvBuilder(
        env_cls=Flop5Holdem,
        env_args=env_args,
    )
    env_bldr.lut_holder = lut_holder
    
    # Calculate board combinations
    n_cards_in_deck = 52
    n_hole_cards = 4  # 2 per player
    n_flop_cards = 5
    
    # Cards available for board: 52 - 4 (hole cards) = 48
    n_available = n_cards_in_deck - n_hole_cards
    
    # Number of board combinations: C(48, 5)
    from scipy.special import comb
    n_board_combinations = int(comb(n_available, n_flop_cards, exact=True))
    
    print("=" * 80)
    print("Flop5Holdem Public Tree Analysis")
    print("=" * 80)
    print(f"\nGame Structure:")
    print(f"  - Rounds: PREFLOP, FLOP")
    print(f"  - Board cards: {n_flop_cards} cards dealt at once")
    print(f"  - Max raises per round: PREFLOP=2, FLOP=2")
    print(f"  - Actions: FOLD (0), CHECK/CALL (1), RAISE (2)")
    print(f"  - Players: 2 (heads-up)")
    
    print(f"\nBoard Combinations:")
    print(f"  - Cards in deck: {n_cards_in_deck}")
    print(f"  - Hole cards dealt: {n_hole_cards} (2 per player)")
    print(f"  - Cards available for board: {n_available}")
    print(f"  - Number of possible boards: C({n_available}, {n_flop_cards}) = {n_board_combinations:,}")
    
    # Try to build the actual tree (this may be slow/memory intensive)
    print(f"\nAttempting to build public tree...")
    try:
        stack_size = Flop5Holdem.DEFAULT_STACK_SIZE
        tree = PublicTree(
            env_bldr=env_bldr,
            stack_size=stack_size,
            stop_at_street=None,  # Build full tree
            put_out_new_round_after_limit=False,
            is_debugging=False,
        )
        
        tree.build_tree()
        
        print(f"\nTree Statistics:")
        print(f"  - Total nodes: {tree.n_nodes:,}")
        print(f"  - Non-terminal nodes: {tree.n_nonterm:,}")
        print(f"  - Terminal nodes: {tree.n_nodes - tree.n_nonterm:,}")
        
        # Estimate action nodes vs chance nodes
        # Each board combination creates a chance node
        # Action nodes are created for each action sequence
        print(f"\nNode Breakdown (estimated):")
        print(f"  - Chance nodes (board combinations): ~{n_board_combinations:,}")
        print(f"  - Action nodes: ~{tree.n_nodes - n_board_combinations:,}")
        
        # Calculate average branching factor
        if tree.n_nonterm > 0:
            avg_branching = tree.n_nodes / tree.n_nonterm if tree.n_nonterm > 0 else 0
            print(f"  - Average branching factor: ~{avg_branching:.2f}")
        
    except Exception as e:
        print(f"\nError building tree: {e}")
        print(f"This is expected if the tree is too large to build in memory.")
        print(f"\nEstimated tree size based on structure:")
        print(f"  - The tree has {n_board_combinations:,} board combinations")
        print(f"  - Each board can have many action sequences")
        print(f"  - With max 2 raises per round and 2 rounds, action sequences can be quite long")
        print(f"  - Estimated total nodes: Very large (potentially millions)")
    
    print("\n" + "=" * 80)
    print("Note: The public tree size depends on:")
    print("  1. Number of board combinations (fixed: 1,712,304)")
    print("  2. Action sequences (varies by betting pattern)")
    print("  3. Early terminations (folds)")
    print("=" * 80)

if __name__ == "__main__":
    calculate_tree_size()
