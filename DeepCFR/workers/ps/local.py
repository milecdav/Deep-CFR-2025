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


class ParameterServer(ParameterServerBase):

    def __init__(self, t_prof, owner, chief_ref):
        """Parameter server that may reconstruct the chief actor by name."""

        # chief_ref is passed directly as an object.
        chief_handle = chief_ref

        super().__init__(t_prof=t_prof, chief_handle=chief_handle)

        self.owner = owner
        self._device = resolve_device(t_prof.device_parameter_server)
        self._adv_args = t_prof.module_args["adv_training"]

        self._adv_net = self._get_new_adv_net()
        self._adv_optim, self._adv_lr_scheduler = self._get_new_adv_optim()
        self._adv_criterion = rl_util.str_to_loss_cls(self._adv_args.loss_str)

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
        self._adv_net.zero_grad()
        return self._ray.state_dict_to_numpy(self._adv_net.state_dict())

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
        self._adv_lr_scheduler.step(loss)

    def step_scheduler_avrg(self, loss):
        self._avrg_lr_scheduler.step(loss)

    def train_adv_step(self, batch):
        """Forward + backward + optimizer step on one raw batch. Returns loss, or None if batch is None."""
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
            "adv_optim": self._adv_optim.state_dict(),
            "adv_lr_sched": self._adv_lr_scheduler.state_dict(),
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

        self._adv_net.load_state_dict(state["adv_net"])
        self._adv_optim.load_state_dict(state["adv_optim"])
        self._adv_lr_scheduler.load_state_dict(state["adv_lr_sched"])
        if self._AVRG:
            self._avrg_net.load_state_dict(state["avrg_net"])
            self._avrg_optim.load_state_dict(state["avrg_optim"])
            self._avrg_lr_scheduler.load_state_dict(state["avrg_lr_sched"])

    # __________________________________________________________________________________________________________________
    def _get_new_adv_net(self):
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
