#!/usr/bin/env python3
"""
Count the preflop tree endings/terminal nodes.

The preflop action sequences should be:
- f = fold
- cc = check/call (both players)
- crf = check/call, raise, fold
- crc = check/call, raise, call
- rf = raise, fold
- rc = raise, call
etc.

Usage:
    ml Python/3.11.5-GCCcore-13.2.0
    source venv/bin/activate
    python count_preflop_endings.py
"""

import sys
sys.path.insert(0, 'PokerRL-2025')

from PokerRL.game.games import Flop5Holdem
from PokerRL.game.Poker import Poker
from PokerRL.game.wrappers import FlatLimitPokerEnvBuilder
from PokerRL.game._.tree.PublicTree import PublicTree
from PokerRL.game._.look_up_table import LutHolderHoldem
from PokerRL.game._.tree._.nodes import PlayerActionNode, ChanceNode

def get_action_string(action):
    """Convert action to string."""
    if action == Poker.FOLD:
        return "f"
    elif action == Poker.CHECK_CALL:
        return "c"
    elif action == Poker.BET_RAISE:
        return "r"
    return "?"

def get_path_string(node):
    """Get the action sequence path to this node."""
    path = []
    current = node
    while current.parent is not None:
        if isinstance(current, PlayerActionNode):
            path.insert(0, get_action_string(current.action))
        elif isinstance(current, ChanceNode):
            path.insert(0, "chance")
        current = current.parent
    return "".join(path)

def count_preflop_endings():
    """Build preflop tree and count terminal nodes."""
    
    print("=" * 80)
    print("Counting Preflop Tree Endings")
    print("=" * 80)
    
    # Create environment arguments
    env_args = Flop5Holdem.ARGS_CLS(
        n_seats=2,
        starting_stack_sizes_list=[Flop5Holdem.DEFAULT_STACK_SIZE for _ in range(2)],
        use_simplified_headsup_obs=True,
    )
    
    # Create LUT holder
    lut_holder = LutHolderHoldem(Flop5Holdem)
    
    # Create environment builder
    env_bldr = FlatLimitPokerEnvBuilder(
        env_cls=Flop5Holdem,
        env_args=env_args,
    )
    env_bldr.lut_holder = lut_holder
    
    # Build tree stopping at PREFLOP (before flop cards are dealt)
    stack_size = Flop5Holdem.DEFAULT_STACK_SIZE
    tree = PublicTree(
        env_bldr=env_bldr,
        stack_size=[stack_size, stack_size],
        stop_at_street=Poker.FLOP,  # Stop before FLOP
        put_out_new_round_after_limit=False,
        is_debugging=False,
    )
    
    print("Building preflop tree...")
    print("(This may take a while - building full preflop tree)")
    import time
    start_time = time.time()
    
    # Monkey-patch to show progress
    original_build = tree._build_tree
    node_count = [0]
    
    def progress_build(current_node):
        node_count[0] += len(current_node.children) if hasattr(current_node, 'children') else 0
        if node_count[0] % 1000 == 0:
            print(f"  Built {node_count[0]:,} nodes so far...", end='\r', flush=True)
        return original_build(current_node)
    
    tree._build_tree = progress_build
    tree.build_tree()
    print()  # New line after progress
    elapsed = time.time() - start_time
    print(f"Tree built in {elapsed:.2f} seconds\n")
    
    # Count terminal nodes
    terminal_nodes = []
    
    def collect_terminal_nodes(node):
        if node.is_terminal:
            path = get_path_string(node)
            terminal_nodes.append(path)
        for child in node.children:
            collect_terminal_nodes(child)
    
    if tree.root:
        collect_terminal_nodes(tree.root)
    
    print(f"Preflop Tree Statistics:")
    print(f"  - Total nodes: {tree.n_nodes:,}")
    print(f"  - Non-terminal nodes: {tree.n_nonterm:,}")
    print(f"  - Terminal nodes: {len(terminal_nodes):,}")
    
    # Count unique endings
    unique_endings = {}
    for path in terminal_nodes:
        unique_endings[path] = unique_endings.get(path, 0) + 1
    
    print(f"\nUnique Preflop Endings ({len(unique_endings)} unique sequences):")
    print("-" * 80)
    
    # Sort by path length, then alphabetically
    sorted_endings = sorted(unique_endings.items(), key=lambda x: (len(x[0]), x[0]))
    
    for path, count in sorted_endings:
        print(f"  {path:20s} : {count:6d} occurrences")
    
    print("\n" + "=" * 80)
    print("Summary:")
    print(f"  Total terminal nodes: {len(terminal_nodes):,}")
    print(f"  Unique action sequences: {len(unique_endings)}")
    print("=" * 80)
    
    return tree, terminal_nodes, unique_endings

if __name__ == "__main__":
    tree, terminals, unique = count_preflop_endings()
