import argparse
from PokerRL.eval.vs_uniform.VsUniformArgs import VsUniformArgs
from PokerRL.game.games import Flop5Holdem

from DeepCFR.EvalAgentDeepCFR import EvalAgentDeepCFR
from DeepCFR.TrainingProfile import TrainingProfile
from DeepCFR.workers.driver.Driver import Driver

if __name__ == '__main__':
    """
    Runs the experiment from The paper "Single Deep Counterfactual Regret Minimization" (Steinberger 2019).

    Uses 24 cores.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--device-training", default="auto")
    parser.add_argument("--device-parameter-server", default="auto")
    parser.add_argument("--device-inference", default="auto")
    parser.add_argument(
        "--adv-model-type",
        default="nn",
        choices=["nn", "lightgbm"],
        help="Advantage approximator backend for SD-CFR SINGLE mode.",
    )
    parser.add_argument(
        "--adv-lgbm-device-type",
        default="auto",
        choices=["cpu", "gpu", "auto"],
        help="Device type for LightGBM training: 'cpu', 'gpu', or 'auto' (auto uses gpu if CUDA available).",
    )
    import os
    parser.add_argument("--n-workers", type=int,
                        default=max(1, (len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count() or 1) - 2),
                        help="Number of parallel LearnerActor subprocesses")
    parser.add_argument("--run-id", type=int, default=None,
                        help="Run ID to append to experiment name for unique checkpoints (e.g., 0-4)")
    parser.add_argument(
        "--nn-size",
        default="medium",
        choices=["small", "medium", "large"],
        help="Neural network size: 'small' (~2x smaller), 'medium' (original), or 'large' (~2x larger). Only applies when --adv-model-type=nn.",
    )
    args = parser.parse_args()

    # Network size presets (only used for NN, not LightGBM)
    # Original (medium): 192, 64, 64
    # Small: 96, 32, 32 (~2x smaller)
    # Large: 384, 128, 128 (~2x larger)
    nn_size_presets = {
        "small": {
            "n_cards_state_units_adv": 96,
            "n_merge_and_table_layer_units_adv": 32,
            "n_units_final_adv": 32,
            "n_cards_state_units_avrg": 96,
            "n_merge_and_table_layer_units_avrg": 32,
            "n_units_final_avrg": 32,
        },
        "medium": {
            "n_cards_state_units_adv": 192,
            "n_merge_and_table_layer_units_adv": 64,
            "n_units_final_adv": 64,
            "n_cards_state_units_avrg": 192,
            "n_merge_and_table_layer_units_avrg": 64,
            "n_units_final_avrg": 64,
        },
        "large": {
            "n_cards_state_units_adv": 384,
            "n_merge_and_table_layer_units_adv": 128,
            "n_units_final_adv": 128,
            "n_cards_state_units_avrg": 384,
            "n_merge_and_table_layer_units_avrg": 128,
            "n_units_final_avrg": 128,
        },
    }
    nn_config = nn_size_presets[args.nn_size]

    # Build experiment name with run-id if provided
    base_name = "EXPERIMENT_SD-CFR_vs_Deep-CFR_FHP"
    if args.run_id is not None:
        model_type_suffix = "LightGBM" if args.adv_model_type == "lightgbm" else "NN"
        size_suffix = args.nn_size.capitalize() if args.adv_model_type == "nn" else ""
        if size_suffix:
            experiment_name = f"{base_name}_{model_type_suffix}_{size_suffix}_run{args.run_id}"
        else:
            experiment_name = f"{base_name}_{model_type_suffix}_run{args.run_id}"
    else:
        experiment_name = base_name

    ctrl = Driver(t_prof=TrainingProfile(name=experiment_name,

                                         nn_type="feedforward",  # We also support RNNs, but the paper uses FF

                                         n_workers=args.n_workers,
                                         print_progress=False,

                                         # regulate exports
                                         export_each_net=False,
                                         checkpoint_freq=99999999,
                                         eval_agent_export_freq=1,  # produces around 15GB over 150 iterations!

                                         n_actions_traverser_samples=3,  # = external sampling in FHP
                                         n_traversals_per_iter=300000,
                                         n_batches_adv_training=4000,
                                         mini_batch_size_adv=10240,
                                         init_adv_model="random",
                                         adv_model_type=args.adv_model_type,
                                         # LightGBM parameters for Flop5Holdem
                                         adv_lgbm_num_boost_round=100,  # Keep at 200 (don't slow CFR iterations)
                                         adv_lgbm_learning_rate=0.1,
                                         adv_lgbm_num_leaves=64,  # Moderate increase from 64 (96 < 2^7, preserves leaf-wise growth)
                                         adv_lgbm_min_data_in_leaf=20,
                                         adv_lgbm_bagging_freq=1,
                                         adv_lgbm_feature_fraction=0.8,
                                         adv_lgbm_bagging_fraction=0.8,
                                         adv_lgbm_lambda_l1=0.1,  # Keep at 0.1 (0.15 was negligible change)
                                         adv_lgbm_lambda_l2=0.1,  # Keep at 0.1
                                         adv_lgbm_max_depth=6,  # Keep at 7 (sufficient for 96 leaves)
                                         adv_lgbm_max_train_samples=1000000,
                                         adv_lgbm_device_type=args.adv_lgbm_device_type,  # "cpu", "gpu", or "auto" (auto uses gpu if CUDA available)
                                         adv_lgbm_num_threads=args.n_workers,  # Use n_workers threads for LightGBM parallel training

                                         use_pre_layers_adv=True,
                                         n_cards_state_units_adv=nn_config["n_cards_state_units_adv"],
                                         n_merge_and_table_layer_units_adv=nn_config["n_merge_and_table_layer_units_adv"],
                                         n_units_final_adv=nn_config["n_units_final_adv"],

                                         max_buffer_size_adv=2e6,  # *20 LAs = 40M
                                         lr_adv=0.001,
                                         lr_patience_adv=99999999,  # No lr decay

                                         n_batches_avrg_training=20000,
                                         mini_batch_size_avrg=1024,  # *20=20480
                                         init_avrg_model="random",

                                         use_pre_layers_avrg=True,
                                         n_cards_state_units_avrg=nn_config["n_cards_state_units_avrg"],
                                         n_merge_and_table_layer_units_avrg=nn_config["n_merge_and_table_layer_units_avrg"],
                                         n_units_final_avrg=nn_config["n_units_final_avrg"],

                                         max_buffer_size_avrg=2e6,
                                         lr_avrg=0.001,
                                         lr_patience_avrg=99999999,  # No lr decay

                                         # With the H2H evaluator, these two are evaluated against eachother.
                                         eval_modes_of_algo=(
                                             EvalAgentDeepCFR.EVAL_MODE_SINGLE,
                                         ),

                                         log_verbose=True,
                                         game_cls=Flop5Holdem,

                                         # enables simplified obs. Default works also for 3+ players
                                         use_simplified_headsup_obs=True,                                         
                                         vs_uniform_args=VsUniformArgs(
                                             n_hands=300000,
                                             n_workers=args.n_workers,
                                         ),
                                         device_training=args.device_training,
                                         device_parameter_server=args.device_parameter_server,
                                         device_inference=args.device_inference,
                                         ),
                 # Evaluate vs uniform-random every 15 iterations.
                 eval_methods={},

                  # 150 = 300 when 2 viewing alternating iterations as 2 (as usually done).
                  # This repo implements alternating iters as a single iter, which is why this says 150.
                  n_iterations=150,
                  )
    ctrl.run()
