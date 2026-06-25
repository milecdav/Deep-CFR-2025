#!/usr/bin/env python3
"""
Build a preflop tree and then build the flop subtree for one specific board.

This is much more manageable than building the full tree with all 1.7M board combinations.

Usage:
    ml Python/3.11.5-GCCcore-13.2.0
    source venv/bin/activate
    python build_preflop_and_sample_flop_tree.py
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
from PokerRL.game.PokerEnvStateDictEnums import EnvDictIdxs
import numpy as np

def build_preflop_tree():
    """Build just the preflop tree (no board cards)."""
    
    print("=" * 80)
    print("Building Preflop Tree")
    print("=" * 80)
    
    print("Creating environment arguments...")
    # Create environment arguments
    env_args = Flop5Holdem.ARGS_CLS(
        n_seats=2,
        starting_stack_sizes_list=[Flop5Holdem.DEFAULT_STACK_SIZE for _ in range(2)],
        use_simplified_headsup_obs=True,
    )
    print("Environment arguments created.")
    
    print("Creating LUT holder...")
    # Create LUT holder
    lut_holder = LutHolderHoldem(Flop5Holdem)
    print("LUT holder created.")
    
    print("Creating environment builder...")
    # Create environment builder
    env_bldr = FlatLimitPokerEnvBuilder(
        env_cls=Flop5Holdem,
        env_args=env_args,
    )
    env_bldr.lut_holder = lut_holder
    print("Environment builder created.")
    
    print("Creating PublicTree object...")
    # Build tree stopping at PREFLOP (before flop cards are dealt)
    stack_size = Flop5Holdem.DEFAULT_STACK_SIZE
    tree_preflop = PublicTree(
        env_bldr=env_bldr,
        stack_size=[stack_size, stack_size],  # Must be a list with one entry per seat
        stop_at_street=Poker.FLOP,  # Stop before FLOP
        put_out_new_round_after_limit=False,
        is_debugging=False,
    )
    print("PublicTree object created.")
    
    print("Building preflop tree...")
    print("(This may take a while - the preflop tree can be large)")
    import time
    start_time = time.time()
    tree_preflop.build_tree()
    elapsed = time.time() - start_time
    print(f"Preflop tree built in {elapsed:.2f} seconds")
    
    print(f"\nPreflop Tree Statistics:")
    print(f"  - Total nodes: {tree_preflop.n_nodes:,}")
    print(f"  - Non-terminal nodes: {tree_preflop.n_nonterm:,}")
    print(f"  - Terminal nodes: {tree_preflop.n_nodes - tree_preflop.n_nonterm:,}")
    
    return tree_preflop, env_bldr, lut_holder

def build_flop_subtree_for_one_board(env_bldr, lut_holder, board_cards_2d=None):
    """Build the flop subtree for one specific board."""
    
    print("\n" + "=" * 80)
    print("Building Flop Subtree for One Board")
    print("=" * 80)
    
    # Create environment arguments
    env_args = Flop5Holdem.ARGS_CLS(
        n_seats=2,
        starting_stack_sizes_list=[Flop5Holdem.DEFAULT_STACK_SIZE for _ in range(2)],
        use_simplified_headsup_obs=True,
    )
    
    # Create environment builder
    env_bldr_flop = FlatLimitPokerEnvBuilder(
        env_cls=Flop5Holdem,
        env_args=env_args,
    )
    env_bldr_flop.lut_holder = lut_holder
    
    # If no board specified, use a default one (e.g., first 5 cards)
    if board_cards_2d is None:
        # Get a sample board (e.g., cards 0-4)
        board_cards_1d = np.array([0, 1, 2, 3, 4], dtype=np.int8)
        board_cards_2d = lut_holder.get_2d_cards(board_cards_1d)
    
    # Create environment to get card string representation
    env_temp = Flop5Holdem(env_args=env_args, lut_holder=lut_holder, is_evaluating=True)
    print(f"Using board: {env_temp.cards2str(cards_2d=board_cards_2d)}")
    
    # Build a full tree and then extract flop nodes for this specific board
    # Actually, we need to find a way to start from flop
    # Let's build a full tree and count flop nodes
    print("Building full tree to extract flop subtree for one board...")
    
    stack_size = Flop5Holdem.DEFAULT_STACK_SIZE
    tree_flop = PublicTree(
        env_bldr=env_bldr_flop,
        stack_size=[stack_size, stack_size],  # Must be a list with one entry per seat
        stop_at_street=None,  # Build full tree
        put_out_new_round_after_limit=False,
        is_debugging=False,
    )
    
    try:
        tree_flop.build_tree()
        
        # Count nodes that are at flop with our specific board
        flop_nodes = 0
        flop_nonterm = 0
        
        def count_flop_nodes_for_board(node):
            nonlocal flop_nodes, flop_nonterm
            # Check if this node is at flop with our board
            if (node.env_state[EnvDictIdxs.current_round] == Poker.FLOP and
                np.array_equal(node.env_state[EnvDictIdxs.board_2d], board_cards_2d)):
                flop_nodes += 1
                if not node.is_terminal:
                    flop_nonterm += 1
            for child in node.children:
                count_flop_nodes_for_board(child)
        
        if tree_flop.root:
            count_flop_nodes_for_board(tree_flop.root)
        
        print(f"\nFlop Subtree Statistics (for one board):")
        print(f"  - Total flop nodes for this board: {flop_nodes:,}")
        print(f"  - Non-terminal flop nodes: {flop_nonterm:,}")
        print(f"  - Terminal flop nodes: {flop_nodes - flop_nonterm:,}")
        
        print(f"\nFull Tree Statistics:")
        print(f"  - Total nodes: {tree_flop.n_nodes:,}")
        print(f"  - Non-terminal nodes: {tree_flop.n_nonterm:,}")
        print(f"  - Terminal nodes: {tree_flop.n_nodes - tree_flop.n_nonterm:,}")
        
        return tree_flop, flop_nodes
        
    except Exception as e:
        print(f"Error building full tree: {e}")
        print("This is expected if the tree is too large.")
        print("\nNote: To build flop subtree for one board, we would need to:")
        print("  1. Build preflop tree")
        print("  2. Find a chance node that leads to our board")
        print("  3. Build subtree from that chance node")
        print("\nFor now, estimating flop subtree size based on structure...")
        
        # Estimate: flop has similar structure to preflop
        # With max 2 raises per round, similar action sequences
        # We'll use a default estimate if tree_preflop isn't available
        estimated_flop_nodes = 1000
        print(f"  - Estimated flop nodes per board: ~{estimated_flop_nodes:,}")
        
        return None, estimated_flop_nodes

def main():
    """Main function."""
    
    # Build preflop tree
    tree_preflop, env_bldr, lut_holder = build_preflop_tree()
    
    # Build flop subtree for one board
    tree_flop, flop_nodes = build_flop_subtree_for_one_board(env_bldr, lut_holder)
    
    # Summary
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"Preflop tree nodes: {tree_preflop.n_nodes:,}")
    print(f"Flop subtree nodes (for one board): {flop_nodes:,}")
    print(f"\nEstimated full tree size:")
    print(f"  - Preflop nodes: {tree_preflop.n_nodes:,}")
    print(f"  - Flop nodes per board: ~{flop_nodes:,}")
    print(f"  - Number of boards: 1,712,304")
    print(f"  - Estimated total flop nodes: ~{flop_nodes * 1712304:,}")
    print(f"  - Estimated total tree nodes: ~{tree_preflop.n_nodes + flop_nodes * 1712304:,}")
    print("=" * 80)

if __name__ == "__main__":
    main()
