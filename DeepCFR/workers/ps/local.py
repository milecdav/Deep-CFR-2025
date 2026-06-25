import os
import pickle

import psutil
import torch
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter

from DeepCFR.EvalAgentDeepCFR import EvalAgentDeepCFR
from PokerRL.rl import rl_util
from PokerRL.rl.base_cls.workers.ParameterServerBase import ParameterServerBase
from PokerRL.rl.neural.AvrgStrategyNet import AvrgStrategyNet
from PokerRL.rl.neural.DuelingQNet import DuelingQNet
from DeepCFR.utils.device import resolve_device
from DeepCFR.lightgbm_adv import LightGBMAdvModel


class ParameterServer(ParameterServerBase):

    def __init__(self, t_prof, owner, chief_ref):
        """Parameter server that may reconstruct the chief actor by name."""

        # chief_ref is passed directly as an object.
        chief_handle = chief_ref

        super().__init__(t_prof=t_prof, chief_handle=chief_handle)

        self.owner = owner
        self._device = resolve_device(t_prof.device_parameter_server)
        self._adv_args = t_prof.module_args["adv_training"]
        self._adv_model_type = self._adv_args.adv_model_type

        self._adv_net = self._get_new_adv_net()
        if self._adv_model_type == "nn":
            self._adv_optim, self._adv_lr_scheduler = self._get_new_adv_optim()
            self._adv_criterion = rl_util.str_to_loss_cls(self._adv_args.loss_str)
        else:
            self._adv_optim = None
            self._adv_lr_scheduler = None
            self._adv_criterion = None
            self._lgbm_cache = []
            self._lgbm_samples_cached = 0

        if self._t_prof.log_verbose:
            log_dir = os.path.join(self._t_prof.path_log_storage, f"PS{owner}")
            os.makedirs(log_dir, exist_ok=True)
            self._tb_writer = SummaryWriter(log_dir=log_dir, flush_secs=5, max_queue=10)
        else:
            self._tb_writer = None

        self._AVRG = EvalAgentDeepCFR.EVAL_MODE_AVRG_NET in self._t_prof.eval_modes_of_algo
        self._SINGLE = EvalAgentDeepCFR.EVAL_MODE_SINGLE in self._t_prof.eval_modes_of_algo

        # """"""""""""""""""""""""""""
        # Deep CFR
        # """"""""""""""""""""""""""""
        if self._AVRG:
            self._avrg_args = t_prof.module_args["avrg_training"]
            self._avrg_net = self._get_new_avrg_net()
            self._avrg_optim, self._avrg_lr_scheduler = self._get_new_avrg_optim()
            self._avrg_criterion = rl_util.str_to_loss_cls(self._avrg_args.loss_str)

    # ______________________________________________ API to pull from PS _______________________________________________

    def get_adv_weights(self):
        if self._adv_model_type == "nn":
            self._adv_net.zero_grad()
            return self._ray.state_dict_to_numpy(self._adv_net.state_dict())
        # For LightGBM, ensure model is trained and all boosters are present
        if not self._adv_net.is_trained:
            # Model not trained yet - return None or empty state to indicate no model available
            # This should not happen in normal flow, but handle gracefully
            return {
                "model_type": "lightgbm_adv",
                "n_actions": self._adv_net._n_actions,
                "range_size": self._adv_net._range_size,
                "lgbm_params": self._adv_net._lgbm_params,
                "num_boost_round": self._adv_net._num_boost_round,
                "is_trained": False,
                "boosters": [None] * self._adv_net._n_actions,
                "range_idx_to_priv_obs": self._adv_net._range_idx_to_priv_obs,
            }
        # Verify all action-specific boosters are present (at least some should be trained)
        state_dict = self._adv_net.state_dict()
        # Ensure boosters list has correct length
        if len(state_dict.get("boosters", [])) != self._adv_net._n_actions:
            # This should not happen, but fix it if it does
            boosters = state_dict.get("boosters", [])
            while len(boosters) < self._adv_net._n_actions:
                boosters.append(None)
            state_dict["boosters"] = boosters
        return state_dict

    def get_avrg_weights(self):
        self._avrg_net.zero_grad()
        return self._ray.state_dict_to_numpy(self._avrg_net.state_dict())

    # ____________________________________________ API to make PS compute ______________________________________________
    def apply_grads_adv(self, list_of_grads):
        self._apply_grads(list_of_grads=list_of_grads, optimizer=self._adv_optim, net=self._adv_net,
                          grad_norm_clip=self._adv_args.grad_norm_clipping)

    def apply_grads_avrg(self, list_of_grads):
        self._apply_grads(list_of_grads=list_of_grads, optimizer=self._avrg_optim, net=self._avrg_net,
                          grad_norm_clip=self._avrg_args.grad_norm_clipping)

    def reset_adv_net(self, cfr_iter):
        if self._adv_model_type != "nn":
            self._lgbm_cache = []
            self._lgbm_samples_cached = 0
            if self._adv_args.init_adv_model == "random":
                self._adv_net = self._get_new_adv_net()
            elif self._adv_args.init_adv_model != "last":
                raise ValueError(self._adv_args.init_adv_model)
            if self._t_prof.log_verbose and (cfr_iter % 3 == 0) and self._tb_writer is not None:
                process = psutil.Process(os.getpid())
                self._tb_writer.add_scalar("Debug/MemoryUsage/PS", process.memory_info().rss, cfr_iter)
            return

        if self._adv_args.init_adv_model == "last":
            self._adv_net.zero_grad()
            if not self._t_prof.online:
                self._adv_optim, self._adv_lr_scheduler = self._get_new_adv_optim()
        elif self._adv_args.init_adv_model == "random":
            self._adv_net = self._get_new_adv_net()
            self._adv_optim, self._adv_lr_scheduler = self._get_new_adv_optim()
        else:
            raise ValueError(self._adv_args.init_adv_model)

        if self._t_prof.log_verbose and (cfr_iter % 3 == 0) and self._tb_writer is not None:
            # Logs
            process = psutil.Process(os.getpid())
            self._tb_writer.add_scalar("Debug/MemoryUsage/PS", process.memory_info().rss, cfr_iter)

    def reset_avrg_net(self):
        if self._avrg_args.init_avrg_model == "last":
            self._avrg_net.zero_grad()
            if not self._t_prof.online:
                self._avrg_optim, self._avrg_lr_scheduler = self._get_new_avrg_optim()

        elif self._avrg_args.init_avrg_model == "random":
            self._avrg_net = self._get_new_avrg_net()
            self._avrg_optim, self._avrg_lr_scheduler = self._get_new_avrg_optim()

        else:
            raise ValueError(self._avrg_args.init_avrg_model)

    def step_scheduler_adv(self, loss):
        if self._adv_model_type == "nn":
            self._adv_lr_scheduler.step(loss)

    def step_scheduler_avrg(self, loss):
        self._avrg_lr_scheduler.step(loss)

    def train_adv_step(self, batch):
        """Forward + backward + optimizer step on one raw batch. Returns loss, or None if batch is None."""
        if self._adv_model_type != "nn":
            return self._train_adv_step_lightgbm(batch)

        if batch is None:
            return None
        pub_obs, range_idxs, legal_masks, adv, loss_weights = batch
        pub_obs = pub_obs.to(self._device)
        range_idxs = range_idxs.to(self._device)
        legal_masks = legal_masks.to(self._device)
        adv = adv.to(self._device)
        loss_weights = loss_weights.to(self._device)

        self._adv_net.train()
        self._adv_optim.zero_grad()
        pred = self._adv_net(pub_obses=pub_obs, range_idxs=range_idxs, legal_action_masks=legal_masks)
        loss = self._adv_criterion(pred, adv, loss_weights.unsqueeze(-1).expand_as(adv))
        loss.backward()
        if self._adv_args.grad_norm_clipping is not None:
            torch.nn.utils.clip_grad_norm_(self._adv_net.parameters(), max_norm=self._adv_args.grad_norm_clipping)
        self._adv_optim.step()
        loss_val = loss.item()
        self._adv_lr_scheduler.step(loss_val)
        return loss_val

    def _train_adv_step_lightgbm(self, batch):
        # Just cache the batch - don't train yet
        # Training will happen at the end of the iteration via train_adv_finalize
        self._cache_lgbm_batch(batch)
        return None
    
    def train_adv_finalize(self):
        """Train LightGBM on all accumulated batches. Called at the end of iteration."""
        if self._adv_model_type != "nn":
            if self._lgbm_samples_cached == 0:
                print(f"LightGBM: train_adv_finalize called but no samples cached (cache size: {len(self._lgbm_cache)})", flush=True)
                return None

            valid_batches = [b for b in self._lgbm_cache if b is not None]
            if len(valid_batches) == 0:
                print(f"LightGBM: train_adv_finalize called but no valid batches (cache size: {len(self._lgbm_cache)}, samples: {self._lgbm_samples_cached})", flush=True)
                self._lgbm_cache = []
                self._lgbm_samples_cached = 0
                return None
            
            pub_obs = torch.cat([b[0] for b in valid_batches], dim=0).cpu().numpy()
            range_idxs = torch.cat([b[1] for b in valid_batches], dim=0).cpu().numpy()
            legal_masks = torch.cat([b[2] for b in valid_batches], dim=0).cpu().numpy()
            adv = torch.cat([b[3] for b in valid_batches], dim=0).cpu().numpy()
            loss_weights = torch.cat([b[4] for b in valid_batches], dim=0).cpu().numpy()

            self._lgbm_cache = []
            self._lgbm_samples_cached = 0

            loss = self._adv_net.fit(
                pub_obses=pub_obs,
                range_idxs=range_idxs,
                legal_action_masks=legal_masks,
                adv_targets=adv,
                sample_weights=loss_weights,
            )
            return loss
        return None

    def train_adv_direct(self, pub_obs, range_idxs, legal_masks, adv, loss_weights):
        """Train LightGBM directly with provided data (bypasses batch caching)."""
        if self._adv_model_type != "nn":
            # Convert to numpy if needed
            if isinstance(pub_obs, torch.Tensor):
                pub_obs = pub_obs.cpu().numpy()
            if isinstance(range_idxs, torch.Tensor):
                range_idxs = range_idxs.cpu().numpy()
            if isinstance(legal_masks, torch.Tensor):
                legal_masks = legal_masks.cpu().numpy()
            if isinstance(adv, torch.Tensor):
                adv = adv.cpu().numpy()
            if isinstance(loss_weights, torch.Tensor):
                loss_weights = loss_weights.cpu().numpy()
            
            loss = self._adv_net.fit(
                pub_obses=pub_obs,
                range_idxs=range_idxs,
                legal_action_masks=legal_masks,
                adv_targets=adv,
                sample_weights=loss_weights,
            )
            return loss
        return None

    def _cache_lgbm_batch(self, batch):
        if batch is None:
            self._lgbm_cache.append(None)
            return

        pub_obs, range_idxs, legal_masks, adv, loss_weights = batch
        if pub_obs is None:
            self._lgbm_cache.append(None)
            return

        n = int(pub_obs.shape[0])
        max_samples = self._adv_args.lgbm_max_train_samples
        if (self._lgbm_samples_cached + n) > max_samples:
            keep = max(0, max_samples - self._lgbm_samples_cached)
            if keep == 0:
                self._lgbm_cache.append(None)
                return
            idx = torch.randperm(n, device=pub_obs.device)[:keep]
            pub_obs = pub_obs[idx]
            range_idxs = range_idxs[idx]
            legal_masks = legal_masks[idx]
            adv = adv[idx]
            loss_weights = loss_weights[idx]
            n = keep

        self._lgbm_cache.append(
            (
                pub_obs.detach().cpu(),
                range_idxs.detach().cpu(),
                legal_masks.detach().cpu(),
                adv.detach().cpu(),
                loss_weights.detach().cpu(),
            )
        )
        self._lgbm_samples_cached += n

    def train_avrg_step(self, batch):
        """Forward + backward + optimizer step on one raw avrg batch. Returns loss, or None if batch is None."""
        if batch is None or not self._AVRG:
            return None
        pub_obs, range_idxs, legal_masks, avrg, loss_weights = batch
        pub_obs = pub_obs.to(self._device)
        range_idxs = range_idxs.to(self._device)
        legal_masks = legal_masks.to(self._device)
        avrg = avrg.to(self._device)
        loss_weights = loss_weights.to(self._device)

        self._avrg_net.train()
        self._avrg_optim.zero_grad()
        pred = self._avrg_net(pub_obses=pub_obs, range_idxs=range_idxs, legal_action_masks=legal_masks)
        loss = self._avrg_criterion(pred, avrg, loss_weights.unsqueeze(-1).expand_as(avrg))
        loss.backward()
        if self._avrg_args.grad_norm_clipping is not None:
            torch.nn.utils.clip_grad_norm_(self._avrg_net.parameters(), max_norm=self._avrg_args.grad_norm_clipping)
        self._avrg_optim.step()
        loss_val = loss.item()
        self._avrg_lr_scheduler.step(loss_val)
        return loss_val

    # ______________________________________________ API for checkpointing _____________________________________________
    def checkpoint(self, curr_step):
        state = {
            "adv_net": self._adv_net.state_dict(),
            "adv_optim": None if self._adv_optim is None else self._adv_optim.state_dict(),
            "adv_lr_sched": None if self._adv_lr_scheduler is None else self._adv_lr_scheduler.state_dict(),
            "seat_id": self.owner,
        }
        if self._AVRG:
            state["avrg_net"] = self._avrg_net.state_dict()
            state["avrg_optim"] = self._avrg_optim.state_dict()
            state["avrg_lr_sched"] = self._avrg_lr_scheduler.state_dict()

        with open(self._get_checkpoint_file_path(name=self._t_prof.name, step=curr_step,
                                                 cls=self.__class__, worker_id="P" + str(self.owner)),
                  "wb") as pkl_file:
            pickle.dump(obj=state, file=pkl_file, protocol=pickle.HIGHEST_PROTOCOL)

    def load_checkpoint(self, name_to_load, step):
        with open(self._get_checkpoint_file_path(name=name_to_load, step=step,
                                                 cls=self.__class__, worker_id="P" + str(self.owner)),
                  "rb") as pkl_file:
            state = pickle.load(pkl_file)

            assert self.owner == state["seat_id"]

        if self._adv_model_type == "nn":
            self._adv_net.load_state_dict(state["adv_net"])
            self._adv_optim.load_state_dict(state["adv_optim"])
            self._adv_lr_scheduler.load_state_dict(state["adv_lr_sched"])
        else:
            self._adv_net = LightGBMAdvModel.from_state_dict(state["adv_net"])
        if self._AVRG:
            self._avrg_net.load_state_dict(state["avrg_net"])
            self._avrg_optim.load_state_dict(state["avrg_optim"])
            self._avrg_lr_scheduler.load_state_dict(state["avrg_lr_sched"])

    # __________________________________________________________________________________________________________________
    def _get_new_adv_net(self):
        if self._adv_model_type != "nn":
            params = {
                "objective": "regression",
                "metric": "l2",
                "learning_rate": self._adv_args.lgbm_learning_rate,
                "num_leaves": self._adv_args.lgbm_num_leaves,
                "min_data_in_leaf": self._adv_args.lgbm_min_data_in_leaf,
                "feature_fraction": self._adv_args.lgbm_feature_fraction,
                "bagging_fraction": self._adv_args.lgbm_bagging_fraction,
                "bagging_freq": self._adv_args.lgbm_bagging_freq,
                "lambda_l1": self._adv_args.lgbm_lambda_l1,
                "lambda_l2": self._adv_args.lgbm_lambda_l2,
                "max_depth": self._adv_args.lgbm_max_depth,
                "verbosity": self._adv_args.lgbm_verbose,
            }
            # Add num_threads if specified (None means use all available cores)
            if self._adv_args.lgbm_num_threads is not None:
                params["num_threads"] = int(self._adv_args.lgbm_num_threads)
            # Resolve device_type: "auto" -> "gpu" if available, else "cpu"
            device_type = self._adv_args.lgbm_device_type
            if device_type == "auto":
                try:
                    import torch
                    device_type = "gpu" if torch.cuda.is_available() else "cpu"
                except Exception:
                    device_type = "cpu"
            
            return LightGBMAdvModel(
                n_actions=self._env_bldr.N_ACTIONS,
                range_size=self._env_bldr.rules.RANGE_SIZE,
                lgbm_params=params,
                num_boost_round=self._adv_args.lgbm_num_boost_round,
                device_type=device_type,
                range_idx_to_priv_obs=self._env_bldr.lut_holder.LUT_RANGE_IDX_TO_PRIVATE_OBS,
            )
        return DuelingQNet(q_args=self._adv_args.adv_net_args, env_bldr=self._env_bldr, device=self._device)

    def _get_new_avrg_net(self):
        return AvrgStrategyNet(avrg_net_args=self._avrg_args.avrg_net_args, env_bldr=self._env_bldr,
                               device=self._device)

    def _get_new_adv_optim(self):
        opt = rl_util.str_to_optim_cls(self._adv_args.optim_str)(self._adv_net.parameters(), lr=self._adv_args.lr)
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer=opt,
                                                   threshold=0.001,
                                                   factor=0.5,
                                                   patience=self._adv_args.lr_patience,
                                                   min_lr=0.00002)
        return opt, scheduler

    def _get_new_avrg_optim(self):
        opt = rl_util.str_to_optim_cls(self._avrg_args.optim_str)(self._avrg_net.parameters(), lr=self._avrg_args.lr)
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer=opt,
                                                   threshold=0.0001,
                                                   factor=0.5,
                                                   patience=self._avrg_args.lr_patience,
                                                   min_lr=0.00002)
        return opt, scheduler
