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
    import os
    parser.add_argument("--n-workers", type=int,
                        default=max(1, (len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count() or 1) - 2),
                        help="Number of parallel LearnerActor subprocesses")
    args = parser.parse_args()

    ctrl = Driver(t_prof=TrainingProfile(name="EXPERIMENT_SD-CFR_vs_Deep-CFR_FHP",

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
                                         mini_batch_size_adv=10240,  # *20=10240
                                         init_adv_model="random",

                                         use_pre_layers_adv=True,
                                         n_cards_state_units_adv=192,
                                         n_merge_and_table_layer_units_adv=64,
                                         n_units_final_adv=64,

                                         max_buffer_size_adv=2e6,  # *20 LAs = 40M
                                         lr_adv=0.001,
                                         lr_patience_adv=99999999,  # No lr decay

                                         n_batches_avrg_training=20000,
                                         mini_batch_size_avrg=1024,  # *20=20480
                                         init_avrg_model="random",

                                         use_pre_layers_avrg=True,
                                         n_cards_state_units_avrg=192,
                                         n_merge_and_table_layer_units_avrg=64,
                                         n_units_final_avrg=64,

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
                 eval_methods={"vs_uniform": 15},

                  # 150 = 300 when 2 viewing alternating iterations as 2 (as usually done).
                  # This repo implements alternating iters as a single iter, which is why this says 150.
                  n_iterations=150,
                  )
    ctrl.run()
