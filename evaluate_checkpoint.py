#!/usr/bin/env python3
"""
Script to evaluate checkpoints from Deep-CFR training runs.

Usage:
    # Evaluate single agent vs uniform
    python evaluate_checkpoint.py --experiment EXPERIMENT_SD-CFR_vs_Deep-CFR_FHP_LightGBM_run0 --iteration 50 --mode vs_uniform

    # Evaluate single agent with LBR
    python evaluate_checkpoint.py --experiment EXPERIMENT_SD-CFR_vs_Deep-CFR_FHP_LightGBM_run0 --iteration 50 --mode lbr

    # Evaluate two agents head-to-head
    python evaluate_checkpoint.py --experiment1 EXPERIMENT_SD-CFR_vs_Deep-CFR_FHP_LightGBM_run0 --experiment2 EXPERIMENT_SD-CFR_vs_Deep-CFR_FHP_NN_run0 --iteration1 50 --iteration2 50 --mode h2h
"""

import argparse
import multiprocessing as mp
import os
from os.path import join as ospj

import numpy as np

from DeepCFR.EvalAgentDeepCFR import EvalAgentDeepCFR
from PokerRL.eval.lbr.LBRArgs import LBRArgs
from PokerRL.eval.vs_uniform.VsUniformArgs import VsUniformArgs
from PokerRL.eval.head_to_head.H2HArgs import H2HArgs
from PokerRL.game import AgentTournament

# Global variables for multiprocessing (set by _set_h2h_local_state)
_H2H_LOCAL_EVAL_AGENT_PATH_1 = None
_H2H_LOCAL_EVAL_AGENT_PATH_2 = None
_H2H_LOCAL_EVAL_AGENT_1 = None
_H2H_LOCAL_EVAL_AGENT_2 = None
_H2H_LOCAL_EVAL_ENV_BLDR = None


def _set_h2h_local_state(arg1, arg2, eval_env_bldr):
    """Set global state for h2h multiprocessing workers.
    
    For fork method: arg1 and arg2 are eval_agent objects
    For spawn method: arg1 and arg2 are eval_agent paths (strings)
    """
    global _H2H_LOCAL_EVAL_AGENT_PATH_1
    global _H2H_LOCAL_EVAL_AGENT_PATH_2
    global _H2H_LOCAL_EVAL_AGENT_1
    global _H2H_LOCAL_EVAL_AGENT_2
    global _H2H_LOCAL_EVAL_ENV_BLDR
    _H2H_LOCAL_EVAL_ENV_BLDR = eval_env_bldr
    
    # Check if arguments are paths (strings) or agents (objects)
    if isinstance(arg1, str) and isinstance(arg2, str):
        # Spawn method: arguments are paths
        _H2H_LOCAL_EVAL_AGENT_PATH_1 = arg1
        _H2H_LOCAL_EVAL_AGENT_PATH_2 = arg2
        # Load agents in each worker process
        _H2H_LOCAL_EVAL_AGENT_1 = EvalAgentDeepCFR.load_from_disk(path_to_eval_agent=arg1)
        _H2H_LOCAL_EVAL_AGENT_2 = EvalAgentDeepCFR.load_from_disk(path_to_eval_agent=arg2)
    else:
        # Fork method: arguments are agent objects
        _H2H_LOCAL_EVAL_AGENT_1 = arg1
        _H2H_LOCAL_EVAL_AGENT_2 = arg2
        _H2H_LOCAL_EVAL_AGENT_PATH_1 = None
        _H2H_LOCAL_EVAL_AGENT_PATH_2 = None
    np.random.seed()


def _run_h2h_task(task):
    """Run a single h2h evaluation task in a worker process."""
    seat_p0, n_hands, stack_size_list = task
    
    # For fork method, use global agents; for spawn, they're loaded in _set_h2h_local_state
    if _H2H_LOCAL_EVAL_AGENT_1 is None or _H2H_LOCAL_EVAL_AGENT_2 is None:
        # Fallback: load from paths if not set (for spawn method)
        if _H2H_LOCAL_EVAL_AGENT_PATH_1 and _H2H_LOCAL_EVAL_AGENT_PATH_2:
            eval_agent_1 = EvalAgentDeepCFR.load_from_disk(path_to_eval_agent=_H2H_LOCAL_EVAL_AGENT_PATH_1)
            eval_agent_2 = EvalAgentDeepCFR.load_from_disk(path_to_eval_agent=_H2H_LOCAL_EVAL_AGENT_PATH_2)
        else:
            raise RuntimeError("Eval agents not initialized in worker process")
    else:
        eval_agent_1 = _H2H_LOCAL_EVAL_AGENT_1
        eval_agent_2 = _H2H_LOCAL_EVAL_AGENT_2
    
    eval_env_bldr = _H2H_LOCAL_EVAL_ENV_BLDR
    assert eval_env_bldr is not None
    
    # Set modes
    mode1 = eval_agent_1.t_prof.eval_modes_of_algo[0]
    mode2 = eval_agent_2.t_prof.eval_modes_of_algo[0]
    eval_agent_1.set_mode(mode1)
    eval_agent_2.set_mode(mode2)
    
    # Both set_stack_size and get_new_env expect a list
    eval_agent_1.set_stack_size(stack_size=stack_size_list)
    eval_agent_2.set_stack_size(stack_size=stack_size_list)
    env = eval_env_bldr.get_new_env(is_evaluating=True, stack_size=stack_size_list)
    seat_p1 = 1 - seat_p0
    
    winnings = np.empty(shape=(n_hands,), dtype=np.float32)
    for hand_id in range(n_hands):
        _, r_for_all, done, _ = env.reset()
        eval_agent_1.reset(deck_state_dict=env.cards_state_dict())
        eval_agent_2.reset(deck_state_dict=env.cards_state_dict())
        
        while not done:
            p_id_acting = env.current_player.seat_id
            
            if p_id_acting == seat_p0:
                action_int, _ = eval_agent_1.get_action(step_env=True, need_probs=False)
                eval_agent_2.notify_of_action(p_id_acted=p_id_acting, action_he_did=action_int)
            elif p_id_acting == seat_p1:
                action_int, _ = eval_agent_2.get_action(step_env=True, need_probs=False)
                eval_agent_1.notify_of_action(p_id_acted=p_id_acting, action_he_did=action_int)
            else:
                raise ValueError("Only HU supported!")
            
            _, r_for_all, done, _ = env.step(action_int)
        
        winnings[hand_id] = r_for_all[seat_p0] * env.REWARD_SCALAR * env.EV_NORMALIZER
    
    return winnings


def find_eval_agent_path(experiment_name, iteration, data_path=None):
    """Find the path to an exported eval agent."""
    if data_path is None:
        data_path = os.path.join(os.path.expanduser("~"), "poker_ai_data")
    
    eval_agent_dir = ospj(data_path, "eval_agent", experiment_name, str(iteration))
    
    # Try SINGLE mode first (most common for SD-CFR)
    eval_agent_path = ospj(eval_agent_dir, "eval_agentSINGLE.pkl")
    if os.path.exists(eval_agent_path):
        return eval_agent_path
    
    # Try AVRG_NET mode
    eval_agent_path = ospj(eval_agent_dir, "eval_agentAVRG_NET.pkl")
    if os.path.exists(eval_agent_path):
        return eval_agent_path
    
    raise FileNotFoundError(
        f"Could not find eval agent for experiment '{experiment_name}' at iteration {iteration}.\n"
        f"Looked in: {eval_agent_dir}"
    )


def evaluate_vs_uniform(eval_agent, n_hands=300000, n_workers=1):
    """Evaluate agent against uniform random opponent."""
    from PokerRL.eval.vs_uniform.LocalVsUniformMaster import LocalVsUniformMaster
    
    # Create a dummy training profile for the evaluator
    t_prof = eval_agent.t_prof
    t_prof.module_args["vs_uniform"] = VsUniformArgs(n_hands=n_hands, n_workers=n_workers)
    
    # Create evaluator (we'll use it manually since we already have the eval agent)
    print(f"Evaluating vs uniform-random opponent ({n_hands} hands)...")
    
    env_bldr = eval_agent.env_bldr
    env = env_bldr.get_new_env(is_evaluating=True)
    
    eval_agent.set_mode(eval_agent.t_prof.eval_modes_of_algo[0])
    # set_stack_size expects a list
    stack_size_list = [env.DEFAULT_STACK_SIZE] * env_bldr.N_SEATS
    eval_agent.set_stack_size(stack_size=stack_size_list)
    
    winnings = []
    for seat_eval in range(2):
        seat_uniform = 1 - seat_eval
        for hand_id in range(n_hands // 2):
            _, r_for_all, done, _ = env.reset()
            eval_agent.reset(deck_state_dict=env.cards_state_dict())
            
            while not done:
                p_id_acting = env.current_player.seat_id
                if p_id_acting == seat_eval:
                    action_int, _ = eval_agent.get_action(step_env=True, need_probs=False)
                elif p_id_acting == seat_uniform:
                    legal_actions = env.get_legal_actions()
                    action_int = legal_actions[np.random.randint(len(legal_actions))]
                    eval_agent.notify_of_action(p_id_acted=p_id_acting, action_he_did=action_int)
                else:
                    raise ValueError("Only HU supported!")
                
                _, r_for_all, done, _ = env.step(action_int)
            
            winnings.append(r_for_all[seat_eval] * env.REWARD_SCALAR * env.EV_NORMALIZER)
    
    mean = np.mean(winnings)
    std = np.std(winnings)
    n = len(winnings)
    conf_95 = 1.96 * std / np.sqrt(n)
    
    print(f"\nResults:")
    print(f"  Mean: {mean:.6f} MBB_per_G")
    print(f"  95% CI: [{mean - conf_95:.6f}, {mean + conf_95:.6f}]")
    print(f"  N hands: {n}")
    
    return mean, conf_95


def evaluate_lbr(eval_agent, n_lbr_hands=30000, n_workers=10):
    """Evaluate agent using Local Best Response (LBR)."""
    from PokerRL.eval.lbr.LocalLBRMaster import LocalLBRMaster
    from PokerRL.eval.lbr.LocalLBRWorker import LocalLBRWorker
    from DeepCFR.workers.chief.local import Chief
    
    t_prof = eval_agent.t_prof
    t_prof.module_args["lbr"] = LBRArgs(
        n_lbr_hands_per_seat=n_lbr_hands,
        n_parallel_lbr_workers=n_workers,
        use_gpu_for_batch_eval=False,
    )
    
    print(f"Evaluating with LBR ({n_lbr_hands} hands per seat, {n_workers} workers)...")
    print("Note: LBR evaluation requires a running Chief. This is a simplified version.")
    print("For full LBR evaluation, consider using the training script with LBR evaluator enabled.")
    
    # Create chief and set strategy buffers from eval agent
    chief = Chief(t_prof=t_prof)
    
    # Manually set strategy buffers from eval agent
    if hasattr(eval_agent, '_strategy_buffers'):
        chief._strategy_buffers = eval_agent._strategy_buffers
    
    # Create LBR workers
    lbr_workers = [
        LocalLBRWorker(t_prof=t_prof, chief_handle=chief, eval_agent_cls=EvalAgentDeepCFR)
        for _ in range(n_workers)
    ]
    
    # Create LBR master
    lbr_master = LocalLBRMaster(t_prof=t_prof, chief_handle=chief)
    lbr_master.set_worker_handles(*lbr_workers)
    
    # Set weights for evaluation
    if hasattr(eval_agent, '_strategy_buffers'):
        lbr_master.weights_for_eval_agent = {
            EvalAgentDeepCFR.EVAL_MODE_SINGLE: [
                eval_agent._strategy_buffers[p].state_dict() for p in range(t_prof.n_seats)
            ]
        }
    else:
        # Fallback for AVRG_NET mode
        lbr_master.weights_for_eval_agent = {
            EvalAgentDeepCFR.EVAL_MODE_AVRG_NET: [
                eval_agent.avrg_net_policies[p].net_state_dict() for p in range(t_prof.n_seats)
            ]
        }
    
    # Run evaluation
    lbr_master.evaluate(iter_nr=0)
    
    print("\nLBR evaluation complete. Check TensorBoard logs for results.")


def evaluate_h2h(eval_agent1, eval_agent2, n_hands=100000, n_workers=1, eval_agent_path_1=None, eval_agent_path_2=None):
    """Evaluate two agents head-to-head with optional parallelization."""
    print(f"Evaluating head-to-head ({n_hands} hands per seat, {n_workers} workers)...")
    
    env_bldr = eval_agent1.env_bldr
    
    # Ensure both agents use the same mode
    mode1 = eval_agent1.t_prof.eval_modes_of_algo[0]
    mode2 = eval_agent2.t_prof.eval_modes_of_algo[0]
    
    eval_agent1.set_mode(mode1)
    eval_agent2.set_mode(mode2)
    
    # Get default stack size from the game class (as a list for get_new_env)
    try:
        default_stack = env_bldr.env_cls.DEFAULT_STACK_SIZE
    except AttributeError:
        # Fallback: use a reasonable default or get from env_args
        stack_list = getattr(env_bldr.env_args, 'starting_stack_sizes_list', None)
        if stack_list is None or not isinstance(stack_list, list):
            default_stack = 20000
        else:
            default_stack = stack_list[0] if len(stack_list) > 0 else 20000
    
    # stack_size must be a list for get_new_env
    stack_size = [default_stack] * env_bldr.N_SEATS
    
    if n_workers == 1:
        # Single-threaded: use AgentTournament directly
        env_args = env_bldr.env_args
        tournament = AgentTournament(
            env_cls=env_bldr.env_cls,
            env_args=env_args,
            eval_agent_1=eval_agent1,
            eval_agent_2=eval_agent2,
        )
        mean, upper_conf95, lower_conf95 = tournament.run(n_games_per_seat=n_hands // 2)
    else:
        # Multi-threaded: use multiprocessing
        n_hands_per_seat = n_hands // 2
        n_hands_per_worker = max(1, n_hands_per_seat // n_workers)
        
        # Create tasks: (seat_p0, n_hands, stack_size_list)
        tasks = []
        for seat_p0 in range(2):  # Both seats
            remaining = n_hands_per_seat
            worker_id = 0
            while remaining > 0:
                hands_this_worker = min(n_hands_per_worker, remaining)
                tasks.append((seat_p0, hands_this_worker, stack_size))
                remaining -= hands_this_worker
                worker_id += 1
        
        # Use multiprocessing
        try:
            # Check if CUDA is initialized (affects fork vs spawn)
            try:
                import torch
                cuda_initialized = torch.cuda.is_initialized()
            except Exception:
                cuda_initialized = False
            
            mp_start_method = "spawn" if cuda_initialized else "fork"
            ctx = mp.get_context(mp_start_method)
            
            if mp_start_method == "fork":
                # For fork, we can use the objects directly
                _set_h2h_local_state(eval_agent1, eval_agent2, env_bldr)
                pool = ctx.Pool(processes=n_workers)
            else:
                # For spawn, we need paths to reload agents in each worker
                if eval_agent_path_1 is None or eval_agent_path_2 is None:
                    raise ValueError(
                        "eval_agent_path_1 and eval_agent_path_2 must be provided for spawn method. "
                        "This should be set automatically when calling from main()."
                    )
                # For spawn, pass paths and let workers load agents
                pool = ctx.Pool(
                    processes=n_workers,
                    initializer=_set_h2h_local_state,
                    initargs=(eval_agent_path_1, eval_agent_path_2, env_bldr),
                )
            
            # Run tasks in parallel
            results = pool.map(_run_h2h_task, tasks)
            pool.close()
            pool.join()
            
            # Aggregate results
            all_winnings = np.concatenate(results)
            
        except Exception as e:
            import traceback
            print(f"Warning: Multiprocessing failed ({e}), falling back to single-threaded")
            if __debug__:  # Only print traceback in debug mode
                traceback.print_exc()
            # Fallback to single-threaded
            env_args = env_bldr.env_args
            tournament = AgentTournament(
                env_cls=env_bldr.env_cls,
                env_args=env_args,
                eval_agent_1=eval_agent1,
                eval_agent_2=eval_agent2,
            )
            mean, upper_conf95, lower_conf95 = tournament.run(n_games_per_seat=n_hands // 2)
            return mean, upper_conf95, lower_conf95
        
        # Compute statistics
        mean = np.mean(all_winnings).item()
        std = np.std(all_winnings).item()
        n = len(all_winnings)
        conf_95 = 1.96 * std / np.sqrt(n)
        upper_conf95 = mean + conf_95
        lower_conf95 = mean - conf_95
    
    print(f"\nHead-to-head results:")
    print(f"  Agent 1 ({eval_agent1.t_prof.name}): {mean:.6f} MBB_per_G")
    print(f"  95% CI: [{lower_conf95:.6f}, {upper_conf95:.6f}]")
    print(f"  Agent 2 ({eval_agent2.t_prof.name}): {-mean:.6f} MBB_per_G")
    print(f"  Total hands: {n_hands}")
    
    return mean, upper_conf95, lower_conf95


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Deep-CFR checkpoints",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument("--experiment", type=str, help="Experiment name (for single agent evaluation)")
    parser.add_argument("--experiment1", type=str, help="First experiment name (for h2h)")
    parser.add_argument("--experiment2", type=str, help="Second experiment name (for h2h)")
    parser.add_argument("--iteration", type=int, help="Iteration number (for single agent)")
    parser.add_argument("--iteration1", type=int, help="Iteration number for first agent (for h2h)")
    parser.add_argument("--iteration2", type=int, help="Iteration number for second agent (for h2h)")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["lbr", "vs_uniform", "h2h"],
        required=True,
        help="Evaluation mode"
    )
    parser.add_argument("--n-hands", type=int, default=300000, help="Number of hands for vs_uniform or h2h")
    parser.add_argument("--n-lbr-hands", type=int, default=30000, help="Number of LBR hands per seat")
    parser.add_argument("--n-workers", type=int, default=10, help="Number of parallel workers")
    parser.add_argument("--data-path", type=str, default=None, help="Path to poker_ai_data (default: ~/poker_ai_data)")
    
    args = parser.parse_args()
    
    # Validate arguments based on mode
    if args.mode == "h2h":
        if not args.experiment1 or not args.experiment2:
            parser.error("--experiment1 and --experiment2 are required for h2h mode")
        if args.iteration1 is None or args.iteration2 is None:
            parser.error("--iteration1 and --iteration2 are required for h2h mode")
    else:
        if not args.experiment:
            parser.error("--experiment is required for lbr and vs_uniform modes")
        if args.iteration is None:
            parser.error("--iteration is required for lbr and vs_uniform modes")
    
    # Load eval agents
    if args.mode == "h2h":
        print(f"Loading eval agent 1: {args.experiment1} at iteration {args.iteration1}")
        eval_agent_path1 = find_eval_agent_path(args.experiment1, args.iteration1, args.data_path)
        eval_agent1 = EvalAgentDeepCFR.load_from_disk(path_to_eval_agent=eval_agent_path1)
        
        print(f"Loading eval agent 2: {args.experiment2} at iteration {args.iteration2}")
        eval_agent_path2 = find_eval_agent_path(args.experiment2, args.iteration2, args.data_path)
        eval_agent2 = EvalAgentDeepCFR.load_from_disk(path_to_eval_agent=eval_agent_path2)
        
        evaluate_h2h(
            eval_agent1, 
            eval_agent2, 
            n_hands=args.n_hands, 
            n_workers=args.n_workers,
            eval_agent_path_1=eval_agent_path1,
            eval_agent_path_2=eval_agent_path2
        )
    else:
        print(f"Loading eval agent: {args.experiment} at iteration {args.iteration}")
        eval_agent_path = find_eval_agent_path(args.experiment, args.iteration, args.data_path)
        eval_agent = EvalAgentDeepCFR.load_from_disk(path_to_eval_agent=eval_agent_path)
        
        if args.mode == "vs_uniform":
            evaluate_vs_uniform(eval_agent, n_hands=args.n_hands, n_workers=args.n_workers)
        elif args.mode == "lbr":
            evaluate_lbr(eval_agent, n_lbr_hands=args.n_lbr_hands, n_workers=args.n_workers)


if __name__ == "__main__":
    main()

