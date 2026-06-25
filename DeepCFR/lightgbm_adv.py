import numpy as np
import threading

def _to_numpy_2d(pub_obses):
    import torch

    if isinstance(pub_obses, torch.Tensor):
        arr = pub_obses.detach().cpu().numpy()
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        return arr.astype(np.float32, copy=False)

    if isinstance(pub_obses, np.ndarray):
        arr = pub_obses
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        return arr.astype(np.float32, copy=False)

    # List/tuple of observations.
    arr = np.asarray(pub_obses, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr


def _range_idxs_to_numpy(range_idxs):
    import torch

    if isinstance(range_idxs, torch.Tensor):
        return range_idxs.detach().cpu().numpy().astype(np.int64, copy=False)
    return np.asarray(range_idxs, dtype=np.int64)


def build_adv_features(pub_obses, range_idxs, range_idx_to_priv_obs):
    """
    Build features for LightGBM advantage model.

    Uses the same one-hot card encoding as the NN (rank + suit per hole card),
    giving LightGBM semantically meaningful features it can split on directly
    (e.g. "has an Ace", "has a heart") instead of arbitrary binary bits.

    Args:
        pub_obses: Public observations
        range_idxs: Range indices (private card representations)
        range_idx_to_priv_obs: LUT mapping range_idx -> one-hot private observation
                               shape (RANGE_SIZE, N_HOLE_CARDS * (N_RANKS + N_SUITS))

    Returns:
        Feature matrix: [pub_obs_features, one_hot_private_cards]
    """
    obs_2d = _to_numpy_2d(pub_obses)
    r = _range_idxs_to_numpy(range_idxs).reshape(-1)

    # Look up one-hot private card encoding (same representation the NN uses)
    priv_obs = range_idx_to_priv_obs[r]  # (batch, priv_obs_dim)

    return np.concatenate([obs_2d, priv_obs], axis=1)


class LightGBMAdvModel:
    MODEL_TYPE = "lightgbm_adv"
    # Lock to serialize GPU initialization (OpenCL context creation is not thread-safe)
    # ParameterServers run in the same process, so threading.Lock is sufficient
    _gpu_init_lock = threading.Lock()
    _gpu_initialized = False

    def __init__(self, n_actions, range_size, lgbm_params, num_boost_round, device_type="cpu",
                 range_idx_to_priv_obs=None):
        self._n_actions = int(n_actions)
        self._range_size = int(range_size)
        self._lgbm_params = dict(lgbm_params)
        # LUT mapping range_idx -> one-hot private observation (same as NN uses)
        if range_idx_to_priv_obs is not None:
            self._range_idx_to_priv_obs = np.asarray(range_idx_to_priv_obs, dtype=np.float32)
        else:
            self._range_idx_to_priv_obs = None
        
        # Set device type: "cpu", "gpu", or "auto" (auto uses gpu if CUDA available)
        device_type = str(device_type).lower()
        if device_type == "auto":
            # Check if CUDA is available (LightGBM GPU requires CUDA)
            try:
                import torch
                device_type = "gpu" if torch.cuda.is_available() else "cpu"
            except Exception:
                device_type = "cpu"
        
        self._lgbm_params['device_type'] = device_type
        # Only print once when GPU is enabled (not for every action/model)
        if device_type == "gpu" and not hasattr(LightGBMAdvModel, '_gpu_warning_printed'):
            print(f"LightGBM: Using GPU acceleration (device_type='gpu')")
            LightGBMAdvModel._gpu_warning_printed = True
        self._num_boost_round = int(num_boost_round)
        self._boosters = [None for _ in range(self._n_actions)]
        self._is_trained = False
        self._device_type = device_type

    @property
    def is_trained(self):
        return self._is_trained

    def fit(self, pub_obses, range_idxs, legal_action_masks, adv_targets, sample_weights):
        try:
            import lightgbm as lgb
        except ImportError as e:
            raise ImportError(
                "LightGBM ADV mode requested but 'lightgbm' is not installed. "
                "Install it with `pip install lightgbm`."
            ) from e

        X = build_adv_features(pub_obses=pub_obses, range_idxs=range_idxs,
                               range_idx_to_priv_obs=self._range_idx_to_priv_obs)
        y = np.asarray(adv_targets, dtype=np.float32)
        masks = np.asarray(legal_action_masks, dtype=np.float32)
        w = np.asarray(sample_weights, dtype=np.float32).reshape(-1)

        if X.shape[0] == 0:
            return None

        # Build LightGBM params dict with improved hyperparameters
        params = {
            "objective": "regression",
            "metric": "l2",
            "learning_rate": self._lgbm_params.get('learning_rate', 0.1),
            "num_leaves": self._lgbm_params.get('num_leaves', 64),  # Increased from 31 (more capacity)
            "max_depth": self._lgbm_params.get('max_depth', 7),  # Set to 7 like other repo
            "min_data_in_leaf": self._lgbm_params.get('min_data_in_leaf', 20),  # Reduced from 100 (allows finer splits)
            "bagging_fraction": self._lgbm_params.get('bagging_fraction', 0.8),  # Added subsampling like other repo
            "bagging_freq": self._lgbm_params.get('bagging_freq', 1),  # Bag every iteration like other repo
            "feature_fraction": self._lgbm_params.get('feature_fraction', 0.8),  # Added feature subsampling like other repo
            "lambda_l1": self._lgbm_params.get('lambda_l1', 0.1),  # Added L1 regularization like other repo
            "lambda_l2": self._lgbm_params.get('lambda_l2', 0.1),  # Added L2 regularization like other repo
            "verbosity": self._lgbm_params.get('verbosity', -1),
            "device_type": self._device_type,
        }
        # Add num_threads if specified
        if self._lgbm_params.get('num_threads') is not None:
            params["num_threads"] = int(self._lgbm_params['num_threads'])

        total_loss = 0.0
        total_weight = 0.0

        # Train separate booster for each action (faster than MultiOutputRegressor)
        for a in range(self._n_actions):
            wa = w * masks[:, a]
            if float(np.sum(wa)) <= 0:
                self._boosters[a] = None
                continue

            # Copy sliced array to avoid LightGBM memory warning about doubled peak memory
            y_a = np.asarray(y[:, a], dtype=np.float32, copy=True)
            ds = lgb.Dataset(X, label=y_a, weight=wa, free_raw_data=True)
            
            # Serialize GPU initialization to avoid deadlocks when multiple ParameterServers initialize simultaneously
            # Only the very first GPU training call needs locking
            if self._device_type == 'gpu' and not LightGBMAdvModel._gpu_initialized:
                with LightGBMAdvModel._gpu_init_lock:
                    # Double-check after acquiring lock (another process might have initialized)
                    if not LightGBMAdvModel._gpu_initialized:
                        # Debug: Print before first GPU training call
                        if a == 0 and self._boosters[0] is None:
                            print(f"LightGBM: Initializing GPU (first training call) for action {a} with {X.shape[0]} samples...", flush=True)
                        
                        # First GPU training call - initialize GPU context
                        booster = lgb.train(
                            params=params,
                            train_set=ds,
                            num_boost_round=self._num_boost_round,
                        )
                        self._boosters[a] = booster
                        LightGBMAdvModel._gpu_initialized = True
                        
                        # Debug: Print after successful initialization
                        if a == 0:
                            print(f"LightGBM: GPU initialized successfully for action {a}", flush=True)
                    else:
                        # Another process already initialized GPU, proceed with normal training
                        booster = lgb.train(
                            params=params,
                            train_set=ds,
                            num_boost_round=self._num_boost_round,
                        )
                        self._boosters[a] = booster
            else:
                # Normal training (CPU or GPU already initialized)
                booster = lgb.train(
                    params=params,
                    train_set=ds,
                    num_boost_round=self._num_boost_round,
                )
                self._boosters[a] = booster

            pred = booster.predict(X)
            err = (pred - y[:, a]) ** 2
            total_loss += float(np.sum(err * wa))
            total_weight += float(np.sum(wa))

        self._is_trained = any(b is not None for b in self._boosters)
        if total_weight <= 0:
            return None
        return total_loss / total_weight

    def predict(self, pub_obses, range_idxs):
        X = build_adv_features(pub_obses=pub_obses, range_idxs=range_idxs,
                               range_idx_to_priv_obs=self._range_idx_to_priv_obs)
        out = np.zeros((X.shape[0], self._n_actions), dtype=np.float32)
        if not self._is_trained:
            return out
        for a, booster in enumerate(self._boosters):
            if booster is not None:
                out[:, a] = booster.predict(X).astype(np.float32, copy=False)
        return out

    def state_dict(self):
        # Serialize boosters to strings (this can be slow for large models)
        # We serialize all boosters even though it's expensive, as this is only called during export/checkpointing
        booster_strings = []
        for i, b in enumerate(self._boosters):
            if b is None:
                booster_strings.append(None)
            else:
                # model_to_string() can be slow for large models
                booster_strings.append(b.model_to_string())
        return {
            "model_type": self.MODEL_TYPE,
            "n_actions": self._n_actions,
            "range_size": self._range_size,
            "lgbm_params": self._lgbm_params,
            "num_boost_round": self._num_boost_round,
            "is_trained": self._is_trained,
            "boosters": booster_strings,
            "range_idx_to_priv_obs": self._range_idx_to_priv_obs,
        }

    @classmethod
    def from_state_dict(cls, state):
        # Extract device_type from saved params or default to cpu
        device_type = state.get("lgbm_params", {}).get("device_type", "cpu")
        obj = cls(
            n_actions=state["n_actions"],
            range_size=state["range_size"],
            lgbm_params=state["lgbm_params"],
            num_boost_round=state["num_boost_round"],
            device_type=device_type,
            range_idx_to_priv_obs=state.get("range_idx_to_priv_obs"),
        )
        obj._is_trained = bool(state.get("is_trained", False))

        boosters = state.get("boosters", [])
        # Ensure boosters list has correct length (should match n_actions)
        if len(boosters) != obj._n_actions:
            # Pad with None if too short, truncate if too long
            if len(boosters) < obj._n_actions:
                boosters = list(boosters) + [None] * (obj._n_actions - len(boosters))
            else:
                boosters = boosters[:obj._n_actions]
        
        # Load boosters if any are present
        if any(s is not None for s in boosters):
            try:
                import lightgbm as lgb
            except ImportError as e:
                raise ImportError(
                    "Loading LightGBM ADV weights requires 'lightgbm'. "
                    "Install it with `pip install lightgbm`."
                ) from e
            obj._boosters = [
                None if s is None else lgb.Booster(model_str=s)
                for s in boosters
            ]
        else:
            # All boosters are None - ensure list is correct length
            obj._boosters = [None] * obj._n_actions
        return obj
