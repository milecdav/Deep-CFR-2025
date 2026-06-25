import torch

from DeepCFR.workers.la.buffers._ReservoirBufferBase import ReservoirBufferBase as _ResBufBase


class AdvReservoirBuffer(_ResBufBase):

    def __init__(self, owner, nn_type, max_size, env_bldr, iter_weighting_exponent):
        super().__init__(owner=owner, max_size=max_size, env_bldr=env_bldr, nn_type=nn_type,
                         iter_weighting_exponent=iter_weighting_exponent)

        self._adv_buffer = torch.zeros((max_size, env_bldr.N_ACTIONS), dtype=torch.float32, device=self.device)

    def add(self, pub_obs, range_idx, legal_action_mask, adv, iteration):
        if self.size < self._max_size:
            self._add(idx=self.size,
                      pub_obs=pub_obs,
                      range_idx=range_idx,
                      legal_action_mask=legal_action_mask,
                      adv=adv,
                      iteration=iteration)
            self.size += 1

        elif self._should_add():
            self._add(idx=self._random_idx(),
                      pub_obs=pub_obs,
                      range_idx=range_idx,
                      legal_action_mask=legal_action_mask,
                      adv=adv,
                      iteration=iteration)

        self.n_entries_seen += 1

    def sample(self, batch_size, device):
        if self.size == 0 or self.size < batch_size:
            return None

        indices = torch.randint(0, self.size, (batch_size,), dtype=torch.long, device=self.device)

        if self._nn_type == "recurrent":
            obses = self._pub_obs_buffer[indices.cpu().numpy()]
        elif self._nn_type == "feedforward":
            obses = self._pub_obs_buffer[indices].to(device)
        else:
            raise NotImplementedError

        return \
            obses, \
            self._range_idx_buffer[indices].to(device), \
            self._legal_action_mask_buffer[indices].to(device), \
            self._adv_buffer[indices].to(device), \
            self._iteration_buffer[indices].to(device) / self._last_cfr_iteration_seen

    def get_all_data(self, max_samples=None, device=None):
        """Get all data from the buffer (or up to max_samples) for LightGBM training.
        
        Returns:
            tuple: (pub_obs, range_idxs, legal_action_masks, adv, loss_weights) or None if buffer is empty
        """
        if self.size == 0:
            return None
        
        # Use all data or up to max_samples
        n_samples = self.size if max_samples is None else min(self.size, max_samples)
        if n_samples == 0:
            return None
        
        # Get all indices (or random sample if max_samples is specified)
        if max_samples is not None and max_samples < self.size:
            indices = torch.randperm(self.size, device=self.device)[:max_samples]
        else:
            indices = torch.arange(self.size, dtype=torch.long, device=self.device)
        
        if self._nn_type == "recurrent":
            obses = self._pub_obs_buffer[indices.cpu().numpy()]
        elif self._nn_type == "feedforward":
            obses = self._pub_obs_buffer[indices]
            if device is not None:
                obses = obses.to(device)
        else:
            raise NotImplementedError

        return (
            obses,
            self._range_idx_buffer[indices],
            self._legal_action_mask_buffer[indices],
            self._adv_buffer[indices],
            self._iteration_buffer[indices] / self._last_cfr_iteration_seen
        )

    def _add(self, idx, pub_obs, range_idx, legal_action_mask, adv, iteration):
        if self._nn_type == "feedforward":
            pub_obs = torch.from_numpy(pub_obs)

        self._pub_obs_buffer[idx] = pub_obs
        self._range_idx_buffer[idx] = range_idx
        self._legal_action_mask_buffer[idx] = legal_action_mask
        self._adv_buffer[idx] = adv

        self._iteration_buffer[idx] = float(iteration) ** self._iter_weighting_exponent

        self._last_cfr_iteration_seen = iteration

    def state_dict(self):
        return {
            "base": super().state_dict(),
            "adv": self._adv_buffer,
        }

    def load_state_dict(self, state):
        super().load_state_dict(state["base"])
        self._adv_buffer = state["adv"]
