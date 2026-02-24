"""Deep Counterfactual Regret Minimization algorithms."""

__all__ = []
__version__ = "0.1.0"

# ---------------------------------------------------------------------------
# Multiprocessing start method
# ---------------------------------------------------------------------------
# Force spawn start method for CUDA safety (must be before any subprocess
# is created).
import multiprocessing as _mp

try:
    _mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass

# ---------------------------------------------------------------------------
# Patch MaybeRay to handle MPFuture objects
# ---------------------------------------------------------------------------
# PokerRL's MaybeRay already has passthrough mode when runs_distributed=False
# (all ray.get/wait/remote calls become no-ops or direct calls).  The only
# missing piece is that wait() and get() need to block on MPFuture objects
# that represent pending results from our LAProxy subprocesses.
try:
    from PokerRL.rl.MaybeRay import MaybeRay
    from DeepCFR.workers.mp_backend import MPFuture

    def _wait(self, _list, num_returns=None, timeout=None, return_not_ready=False):
        if self.runs_distributed:
            try:
                import ray
                num_returns = len(_list) if num_returns is None else num_returns
                rdy, not_rdy = ray.wait(_list, num_returns=num_returns, timeout=timeout)
                if return_not_ready:
                    return rdy, not_rdy
                return rdy
            except Exception:
                pass

        # Block on any pending MPFuture items.
        for item in _list:
            if isinstance(item, MPFuture):
                item.wait()

        if return_not_ready:
            return _list, []
        return _list

    def _get(self, obj):
        if self.runs_distributed:
            try:
                import ray
                return ray.get(obj)
            except Exception:
                pass

        if isinstance(obj, MPFuture):
            return obj.result()
        if isinstance(obj, list):
            return [_get(self, item) for item in obj]
        return obj

    MaybeRay.wait = _wait
    MaybeRay.get = _get

except Exception:
    pass
