import torch

from PokerRL.rl.neural.DuelingQNet import DuelingQNet
from PokerRL.rl.neural.NetWrapperBase import NetWrapperArgsBase as _NetWrapperArgsBase
from PokerRL.rl.neural.NetWrapperBase import NetWrapperBase as _NetWrapperBase
from DeepCFR.utils.device import resolve_device


class AdvWrapper(_NetWrapperBase):

    def __init__(self, env_bldr, adv_training_args, owner, device):
        device = resolve_device(device)
        super().__init__(
            net=DuelingQNet(env_bldr=env_bldr, q_args=adv_training_args.adv_net_args, device=device),
            env_bldr=env_bldr,
            args=adv_training_args,
            owner=owner,
            device=device
        )
        self._batch_size = adv_training_args.batch_size

    def get_advantages(self, pub_obses, range_idxs, legal_action_mask):
        self._net.eval()
        with torch.no_grad():
            return self._net(pub_obses=pub_obses, range_idxs=range_idxs, legal_action_masks=legal_action_mask)

    def _mini_batch_loop(self, buffer, grad_mngr):
        sample = buffer.sample(device=self.device, batch_size=self._batch_size)
        if sample is None:
            return
        batch_pub_obs, \
        batch_range_idxs, \
        batch_legal_action_masks, \
        batch_adv, \
        batch_loss_weight, \
            = sample

        # [batch_size, n_actions]
        adv_pred = self._net(pub_obses=batch_pub_obs,
                             range_idxs=batch_range_idxs,
                             legal_action_masks=batch_legal_action_masks)

        grad_mngr.backprop(pred=adv_pred, target=batch_adv,
                           loss_weights=batch_loss_weight.unsqueeze(-1).expand_as(batch_adv))


class LightGBMAdvWrapper:

    def __init__(self, env_bldr, adv_training_args, owner, device):
        self._env_bldr = env_bldr
        self._adv_training_args = adv_training_args
        self.owner = owner
        self.device = resolve_device(device)
        self.loss_last_batch = None
        self._model_state = None

    def net_state_dict(self):
        return self._model_state

    def load_net_state_dict(self, state_dict):
        self._model_state = state_dict

    def state_dict(self):
        return {
            "model_state": self._model_state,
        }

    def load_state_dict(self, state):
        self._model_state = state.get("model_state", None)

    def eval(self):
        return


class AdvTrainingArgs(_NetWrapperArgsBase):

    def __init__(self,
                 adv_net_args,
                 n_batches_adv_training=1000,
                 batch_size=4096,
                 n_mini_batches_per_update=1,
                 optim_str="adam",
                 loss_str="weighted_mse",
                 lr=0.001,
                 grad_norm_clipping=10.0,
                 device_training="cpu",
                 max_buffer_size=2e6,
                 lr_patience=100,
                 init_adv_model="last",
                 adv_model_type="nn",
                 lgbm_num_boost_round=200,
                 lgbm_learning_rate=0.05,
                 lgbm_num_leaves=96,
                 lgbm_min_data_in_leaf=50,
                 lgbm_feature_fraction=0.8,
                 lgbm_bagging_fraction=0.8,
                 lgbm_bagging_freq=1,
                 lgbm_lambda_l1=0.0,
                 lgbm_lambda_l2=1.0,
                 lgbm_max_depth=-1,
                 lgbm_max_train_samples=300000,
                 lgbm_verbose=-1,
                 lgbm_device_type="cpu",
                 lgbm_num_threads=None,
                 ):
        super().__init__(batch_size=batch_size, n_mini_batches_per_update=n_mini_batches_per_update,
                         optim_str=optim_str, loss_str=loss_str, lr=lr, grad_norm_clipping=grad_norm_clipping,
                         device_training=device_training)
        self.adv_net_args = adv_net_args
        self.n_batches_adv_training = n_batches_adv_training
        self.lr_patience = lr_patience
        self.max_buffer_size = int(max_buffer_size)
        self.init_adv_model = init_adv_model
        self.adv_model_type = str(adv_model_type).lower()
        self.lgbm_num_boost_round = int(lgbm_num_boost_round)
        self.lgbm_learning_rate = float(lgbm_learning_rate)
        self.lgbm_num_leaves = int(lgbm_num_leaves)
        self.lgbm_min_data_in_leaf = int(lgbm_min_data_in_leaf)
        self.lgbm_feature_fraction = float(lgbm_feature_fraction)
        self.lgbm_bagging_fraction = float(lgbm_bagging_fraction)
        self.lgbm_bagging_freq = int(lgbm_bagging_freq)
        self.lgbm_lambda_l1 = float(lgbm_lambda_l1)
        self.lgbm_lambda_l2 = float(lgbm_lambda_l2)
        self.lgbm_max_depth = int(lgbm_max_depth) if lgbm_max_depth != -1 else -1
        self.lgbm_max_train_samples = int(lgbm_max_train_samples)
        self.lgbm_verbose = int(lgbm_verbose)
        self.lgbm_device_type = str(lgbm_device_type).lower()
        self.lgbm_num_threads = lgbm_num_threads  # None = use all cores, or int for specific number