# Copyright (c) 2019 Eric Steinberger


import copy

import psutil
import torch
from PokerRL.game import bet_sets


def _resolve_device(device_str):
    """Resolve 'auto' to 'cuda' if available, otherwise 'cpu'."""
    if device_str == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_str
from PokerRL.game.games import DiscretizedNLLeduc
from PokerRL.game.wrappers import HistoryEnvBuilder, FlatLimitPokerEnvBuilder
from PokerRL.rl.MaybeRay import MaybeRay
from PokerRL.rl.base_cls.TrainingProfileBase import TrainingProfileBase
from PokerRL.rl.neural.AvrgStrategyNet import AvrgNetArgs
from PokerRL.rl.neural.DuelingQNet import DuelingQArgs

from DeepCFR.EvalAgentDeepCFR import EvalAgentDeepCFR
from DeepCFR.workers.la.AdvWrapper import AdvTrainingArgs
from DeepCFR.workers.la.AvrgWrapper import AvrgTrainingArgs
from utils.memory import estimate_batch_size


class TrainingProfile(TrainingProfileBase):

    def __init__(self,

                 # ------ General
                 name="",
                 log_verbose=True,
                 print_progress=True,  # Whether to print progress messages like "Generating Data...", "Training Advantage Net...", etc.
                 log_export_freq=1,
                 checkpoint_freq=99999999,
                 eval_agent_export_freq=999999999,
                 n_learner_actor_workers=8,
                 n_workers=None,  # Alias for n_learner_actor_workers; takes precedence if set
                 max_n_las_sync_simultaneously=10,
                 nn_type="feedforward",  # "recurrent" or "feedforward"

                 # ------ Computing
                 path_data=None,
                 device_inference="cpu",
                 device_training="cpu",
                 device_parameter_server="cpu",
                 DISTRIBUTED=False,
                 CLUSTER=False,
                 DEBUGGING=False,
                 memory_per_worker=None,  # bytes per Ray worker; 0 disables limit
                 memory_per_worker_multiplier=1.0,  # scale auto or explicit memory for large models
                 object_store_memory=None,

                 # ------ Env
                 game_cls=DiscretizedNLLeduc,
                 n_seats=2,
                 agent_bet_set=bet_sets.B_2,
                 start_chips=None,
                 chip_randomness=(0, 0),
                 uniform_action_interpolation=False,
                 use_simplified_headsup_obs=True,

                 # ------ Evaluation
                 eval_modes_of_algo=(EvalAgentDeepCFR.EVAL_MODE_SINGLE,),
                 eval_stack_sizes=None,

                 # ------ General Deep CFR params
                 n_traversals_per_iter=30000,
                 online=False,
                 iter_weighting_exponent=1.0,
                 n_actions_traverser_samples=3,

                 sampler="mo",

                 # --- Adv Hyperparameters
                 n_batches_adv_training=5000,
                 init_adv_model="random",

                 rnn_cls_str_adv="lstm",
                 rnn_units_adv=128,
                 rnn_stack_adv=1,
                 dropout_adv=0.0,
                 use_pre_layers_adv=False,
                 n_cards_state_units_adv=96,
                 n_merge_and_table_layer_units_adv=32,
                 n_units_final_adv=64,
                 mini_batch_size_adv=None,
                 n_mini_batches_per_la_per_update_adv=1,
                 optimizer_adv="adam",
                 loss_adv="weighted_mse",
                 lr_adv=0.001,
                 grad_norm_clipping_adv=10.0,
                 lr_patience_adv=999999999,
                 normalize_last_layer_FLAT_adv=True,
                 adv_model_type="nn",  # "nn" or "lightgbm"
                 adv_lgbm_num_boost_round=200,  # Match other repo
                 adv_lgbm_learning_rate=0.1,  # Match other repo
                 adv_lgbm_num_leaves=96,  # Moderate increase from 64 (96 < 2^7, preserves leaf-wise growth)
                 adv_lgbm_min_data_in_leaf=20,  # Match other repo (reduced from 50)
                 adv_lgbm_feature_fraction=0.8,  # Match other repo
                 adv_lgbm_bagging_fraction=0.8,  # Match other repo
                 adv_lgbm_bagging_freq=1,  # Match other repo
                 adv_lgbm_lambda_l1=0.1,  # Match other repo (added regularization)
                 adv_lgbm_lambda_l2=0.1,  # Match other repo (reduced from 1.0)
                 adv_lgbm_max_depth=7,  # Match other repo (set limit instead of unlimited)
                 adv_lgbm_max_train_samples=300000,
                 adv_lgbm_verbose=-1,
                 adv_lgbm_device_type="cpu",  # "cpu", "gpu", or "auto" (auto uses gpu if available)
                 adv_lgbm_num_threads=None,  # None = use all available cores, or specify number

                 max_buffer_size_adv=3e6,

                 # ------ SPECIFIC TO AVRG NET
                 n_batches_avrg_training=15000,
                 init_avrg_model="random",

                 rnn_cls_str_avrg="lstm",
                 rnn_units_avrg=128,
                 rnn_stack_avrg=1,
                 dropout_avrg=0.0,
                 use_pre_layers_avrg=False,
                 n_cards_state_units_avrg=96,
                 n_merge_and_table_layer_units_avrg=32,
                 n_units_final_avrg=64,
                 mini_batch_size_avrg=None,
                 n_mini_batches_per_la_per_update_avrg=1,
                 loss_avrg="weighted_mse",
                 optimizer_avrg="adam",
                 lr_avrg=0.001,
                 grad_norm_clipping_avrg=10.0,
                 lr_patience_avrg=999999999,
                 normalize_last_layer_FLAT_avrg=True,

                 max_buffer_size_avrg=3e6,

                 # ------ SPECIFIC TO SINGLE
                 export_each_net=False,
                 eval_agent_max_strat_buf_size=None,

                 # ------ Optional
                 lbr_args=None,
                 rl_br_args=None,
                 h2h_args=None,
                 vs_uniform_args=None,

                 ):
        if n_workers is not None:
            n_learner_actor_workers = n_workers
        adv_model_type = str(adv_model_type).lower()
        if adv_model_type not in ("nn", "lightgbm"):
            raise ValueError(f"Unknown adv_model_type: {adv_model_type}. Expected 'nn' or 'lightgbm'.")

        # Resolve "auto" device strings to actual PyTorch device names
        device_training = _resolve_device(device_training)
        device_inference = _resolve_device(device_inference)
        device_parameter_server = _resolve_device(device_parameter_server)

        print(" ************************** Initing args for: ", name, "  **************************")
        if object_store_memory is None:
            total_mem = psutil.virtual_memory().total
            object_store_memory = int(total_mem * 0.4)
        elif isinstance(object_store_memory, float) and 0 < object_store_memory < 1:
            total_mem = psutil.virtual_memory().total
            object_store_memory = int(total_mem * object_store_memory)
        self.object_store_memory = object_store_memory
        MaybeRay._object_store_memory = object_store_memory

        if mini_batch_size_adv is None or mini_batch_size_avrg is None:
            est = estimate_batch_size()
            if mini_batch_size_adv is None:
                mini_batch_size_adv = est
            if mini_batch_size_avrg is None:
                mini_batch_size_avrg = est

        if nn_type == "recurrent":
            from PokerRL.rl.neural.MainPokerModuleRNN import MPMArgsRNN

            env_bldr_cls = HistoryEnvBuilder

            mpm_args_adv = MPMArgsRNN(rnn_cls_str=rnn_cls_str_adv,
                                      rnn_units=rnn_units_adv,
                                      rnn_stack=rnn_stack_adv,
                                      rnn_dropout=dropout_adv,
                                      use_pre_layers=use_pre_layers_adv,
                                      n_cards_state_units=n_cards_state_units_adv,
                                      n_merge_and_table_layer_units=n_merge_and_table_layer_units_adv)
            mpm_args_avrg = MPMArgsRNN(rnn_cls_str=rnn_cls_str_avrg,
                                       rnn_units=rnn_units_avrg,
                                       rnn_stack=rnn_stack_avrg,
                                       rnn_dropout=dropout_avrg,
                                       use_pre_layers=use_pre_layers_avrg,
                                       n_cards_state_units=n_cards_state_units_avrg,
                                       n_merge_and_table_layer_units=n_merge_and_table_layer_units_avrg)

        elif nn_type == "feedforward":
            from PokerRL.rl.neural.MainPokerModuleFLAT import MPMArgsFLAT

            env_bldr_cls = FlatLimitPokerEnvBuilder

            mpm_args_adv = MPMArgsFLAT(use_pre_layers=use_pre_layers_adv,
                                       card_block_units=n_cards_state_units_adv,
                                       other_units=n_merge_and_table_layer_units_adv,
                                       normalize=normalize_last_layer_FLAT_adv)
            mpm_args_avrg = MPMArgsFLAT(use_pre_layers=use_pre_layers_avrg,
                                        card_block_units=n_cards_state_units_avrg,
                                        other_units=n_merge_and_table_layer_units_avrg,
                                        normalize=normalize_last_layer_FLAT_avrg)

        else:
            raise ValueError(nn_type)

        if adv_model_type == "lightgbm" and nn_type != "feedforward":
            raise ValueError("LightGBM ADV currently supports only nn_type='feedforward'.")

        super().__init__(
            name=name,
            log_verbose=log_verbose,
            log_export_freq=log_export_freq,
            checkpoint_freq=checkpoint_freq,
            eval_agent_export_freq=eval_agent_export_freq,
            path_data=path_data,
            game_cls=game_cls,
            env_bldr_cls=env_bldr_cls,
            start_chips=start_chips,
            eval_modes_of_algo=eval_modes_of_algo,
            eval_stack_sizes=eval_stack_sizes,

            DEBUGGING=DEBUGGING,
            DISTRIBUTED=False,
            CLUSTER=False,
            device_inference=device_inference,

            module_args={
                "adv_training": AdvTrainingArgs(
                    adv_net_args=DuelingQArgs(
                        mpm_args=mpm_args_adv,
                        n_units_final=n_units_final_adv,
                    ),
                    n_batches_adv_training=n_batches_adv_training,
                    init_adv_model=init_adv_model,
                    batch_size=mini_batch_size_adv,
                    n_mini_batches_per_update=n_mini_batches_per_la_per_update_adv,
                    optim_str=optimizer_adv,
                    loss_str=loss_adv,
                    lr=lr_adv,
                    grad_norm_clipping=grad_norm_clipping_adv,
                    device_training=device_training,
                    max_buffer_size=max_buffer_size_adv,
                    lr_patience=lr_patience_adv,
                    adv_model_type=adv_model_type,
                    lgbm_num_boost_round=adv_lgbm_num_boost_round,
                    lgbm_learning_rate=adv_lgbm_learning_rate,
                    lgbm_num_leaves=adv_lgbm_num_leaves,
                    lgbm_min_data_in_leaf=adv_lgbm_min_data_in_leaf,
                    lgbm_feature_fraction=adv_lgbm_feature_fraction,
                    lgbm_bagging_fraction=adv_lgbm_bagging_fraction,
                    lgbm_bagging_freq=adv_lgbm_bagging_freq,
                    lgbm_lambda_l1=adv_lgbm_lambda_l1,
                    lgbm_lambda_l2=adv_lgbm_lambda_l2,
                    lgbm_max_depth=adv_lgbm_max_depth,
                    lgbm_max_train_samples=adv_lgbm_max_train_samples,
                    lgbm_verbose=adv_lgbm_verbose,
                    lgbm_device_type=adv_lgbm_device_type,
                    lgbm_num_threads=adv_lgbm_num_threads,
                ),
                "avrg_training": AvrgTrainingArgs(
                    avrg_net_args=AvrgNetArgs(
                        mpm_args=mpm_args_avrg,
                        n_units_final=n_units_final_avrg,
                    ),
                    n_batches_avrg_training=n_batches_avrg_training,
                    init_avrg_model=init_avrg_model,
                    batch_size=mini_batch_size_avrg,
                    n_mini_batches_per_update=n_mini_batches_per_la_per_update_avrg,
                    loss_str=loss_avrg,
                    optim_str=optimizer_avrg,
                    lr=lr_avrg,
                    grad_norm_clipping=grad_norm_clipping_avrg,
                    device_training=device_training,
                    max_buffer_size=max_buffer_size_avrg,
                    lr_patience=lr_patience_avrg,
                ),
                "env": game_cls.ARGS_CLS(
                    n_seats=n_seats,
                    starting_stack_sizes_list=[start_chips for _ in range(n_seats)],
                    bet_sizes_list_as_frac_of_pot=copy.deepcopy(agent_bet_set),
                    stack_randomization_range=chip_randomness,
                    use_simplified_headsup_obs=use_simplified_headsup_obs,
                    uniform_action_interpolation=uniform_action_interpolation
                ),
                "lbr": lbr_args,
                "rlbr": rl_br_args,
                "h2h": h2h_args,
                "vs_uniform": vs_uniform_args,
            }
        )
        self.path_log_storage = None

        self.nn_type = nn_type
        self.online = online
        self.n_traversals_per_iter = n_traversals_per_iter
        self.iter_weighting_exponent = iter_weighting_exponent
        self.sampler = sampler
        self.n_actions_traverser_samples = n_actions_traverser_samples

        self.mini_batch_size_adv = mini_batch_size_adv
        self.mini_batch_size_avrg = mini_batch_size_avrg

        # SINGLE
        self.export_each_net = export_each_net
        self.eval_agent_max_strat_buf_size = eval_agent_max_strat_buf_size

        # Progress printing
        self.print_progress = print_progress

        # Always use the requested number of LearnerActor workers.
        # DISTRIBUTED/CLUSTER are ignored; parallelism is handled via
        # torch.multiprocessing (LAProxy subprocesses).
        print("Running with ", n_learner_actor_workers, "LearnerActor Workers.")
        self.n_learner_actors = n_learner_actor_workers
        self.max_n_las_sync_simultaneously = max_n_las_sync_simultaneously

        self.device_parameter_server = device_parameter_server
        self.memory_per_worker = memory_per_worker
        self.memory_per_worker_multiplier = memory_per_worker_multiplier

    def __getstate__(self):
        state = self.__dict__.copy()
        state["tb_writer"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.tb_writer = None
