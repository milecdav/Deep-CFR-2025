#!/usr/bin/env python3
"""
Precompute LBR preflop equity cache for faster LBR evaluation.

This script computes the preflop equity for all possible LBR hands when the agent
has a full range (minus LBR's cards). The results are cached to disk for use during
LBR evaluation.
"""

import argparse
import os
import pickle
import time
import numpy as np
try:
    from tqdm import tqdm
except ImportError:
    # Fallback if tqdm is not available
    def tqdm(iterable, desc=""):
        print(desc)
        return iterable

from PokerRL.game.Poker import Poker
from PokerRL.game.PokerRange import PokerRange
from PokerRL.eval.lbr import _util
from PokerRL.eval.lbr.LBRArgs import LBRArgs
from DeepCFR.TrainingProfile import TrainingProfile
from PokerRL.game.games import Flop5Holdem


def precompute_preflop_cache(t_prof, cache_path, game_cls, log_every=1, lbr_subloop_log_every=0):
    """Precompute preflop equity for all possible LBR hands."""
    eval_env_bldr = _util.get_env_builder_lbr(t_prof=t_prof)
    rules = eval_env_bldr.rules
    lut_holder = eval_env_bldr.lut_holder
    
    cache = {}
    range_size = rules.RANGE_SIZE
    
    print(f"Precomputing preflop LBR equity cache for {range_size} hands...")
    print(f"Game: {game_cls.__name__}")
    print(f"Cache will be saved to: {cache_path}")
    
    # Import here to avoid circular dependencies
    from PokerRL.eval.lbr.LocalLBRWorker import _LBRRolloutManager
    
    # Iterate over all possible LBR hands
    overall_start = time.time()
    for lbr_range_idx in tqdm(range(range_size), desc="Computing equity"):
        try:
            hand_start = time.time()
            completed = lbr_range_idx + 1
            # Create a fresh environment for each hand (ensures clean preflop state)
            env = eval_env_bldr.get_new_env(is_evaluating=True, stack_size=t_prof.eval_stack_sizes[0])
            env.reset()
            assert env.current_round == Poker.PREFLOP, f"Expected PREFLOP, got {env.current_round}"
            
            # Get LBR's hole cards from range index
            lbr_hand_2d = lut_holder.get_2d_hole_cards_from_range_idx(range_idx=lbr_range_idx)
            
            # Create rollout manager for this LBR hand
            rollout_mngr = _LBRRolloutManager(
                t_prof=t_prof,
                env_bldr=eval_env_bldr,
                env=env,
                lbr_hand_2d=lbr_hand_2d,
                progress_label=f"LBR hand {completed}/{range_size} (idx={lbr_range_idx})",
                progress_every=lbr_subloop_log_every,
            )
            
            # Create a fresh agent range (full range minus LBR's cards)
            agent_range_cache = PokerRange(env_bldr=eval_env_bldr)
            agent_range_cache.reset()
            agent_range_cache.set_cards_to_zero_prob(cards_2d=lbr_hand_2d)
            
            # Compute equity
            equity = rollout_mngr.get_lbr_checkdown_equity(agent_range=agent_range_cache)
            
            # Store in cache
            cache[lbr_range_idx] = float(equity)  # Ensure it's a Python float, not numpy type

            hand_time_s = time.time() - hand_start
            if (completed % max(1, log_every) == 0) or completed == 1 or completed == range_size:
                elapsed_s = time.time() - overall_start
                avg_s = elapsed_s / completed
                remaining = range_size - completed
                eta_s = remaining * avg_s
                print(
                    f"[{completed}/{range_size}] idx={lbr_range_idx} "
                    f"hand_time={hand_time_s:.3f}s avg={avg_s:.3f}s ETA={eta_s/60.0:.1f}m",
                    flush=True,
                )
        except Exception as e:
            print(f"\nError computing equity for range_idx {lbr_range_idx}: {e}", flush=True)
            import traceback
            traceback.print_exc()
            raise
    
    # Save cache to disk
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, 'wb') as f:
        pickle.dump(cache, f)
    
    print(f"\nCache saved successfully! {len(cache)} entries.")
    print(f"Cache file: {cache_path}")
    return cache


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Precompute LBR preflop equity cache')
    parser.add_argument('--game', type=str, default='Flop5Holdem',
                       help='Game class name (default: Flop5Holdem)')
    parser.add_argument('--cache-dir', type=str, default=None,
                       help='Directory to save cache (default: ~/poker_ai_data/lbr_cache)')
    parser.add_argument('--log-every', type=int, default=1,
                       help='Print timing/progress every N hands (default: 1 for debug)')
    parser.add_argument('--lbr-subloop-log-every', type=int, default=0,
                       help='Log every N terminal boards inside LBR recursion (0 disables)')
    args = parser.parse_args()
    
    # Import game class
    if args.game == 'Flop5Holdem':
        from PokerRL.game.games import Flop5Holdem
        game_cls = Flop5Holdem
    else:
        raise ValueError(f"Unknown game: {args.game}")
    
    # Create a minimal training profile for the game
    # Need minimal LBRArgs for get_env_builder_lbr to work
    t_prof = TrainingProfile(
        name="LBR_CACHE_PRECOMPUTE",
        game_cls=game_cls,
        eval_stack_sizes=([game_cls.DEFAULT_STACK_SIZE, game_cls.DEFAULT_STACK_SIZE],),
        lbr_args=LBRArgs(
            n_lbr_hands_per_seat=1,  # Not used for cache computation
            lbr_check_to_round=None,
            n_parallel_lbr_workers=1,
            use_gpu_for_batch_eval=False,
            DISTRIBUTED=False,
        ),
    )
    
    # Determine cache path
    if args.cache_dir:
        cache_dir = args.cache_dir
    else:
        cache_dir = os.path.join(os.path.expanduser("~"), "poker_ai_data", "lbr_cache")
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"preflop_cache_{game_cls.__name__}.pkl")
    
    # Precompute cache
    cache = precompute_preflop_cache(
        t_prof,
        cache_path,
        game_cls,
        log_every=args.log_every,
        lbr_subloop_log_every=args.lbr_subloop_log_every,
    )
    
    print(f"\nDone! Cache contains {len(cache)} preflop equity values.")
    print(f"Sample values: {list(cache.items())[:5]}")
