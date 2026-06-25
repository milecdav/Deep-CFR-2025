"""Multiprocessing backend for Deep-CFR LearnerActors.

Provides MPFuture (async result handle), LAProxy (proxy to a subprocess
LearnerActor), and _la_worker_loop (the subprocess entry point).
"""

import multiprocessing as mp
import traceback


class MPFuture:
    """Represents a pending result from a worker subprocess.

    Commands are sent to the subprocess via a cmd_queue and results come
    back on a result_queue.  Because each subprocess processes commands
    sequentially and we never issue a second command before consuming the
    first result, the result_queue acts as a 1-slot channel.

    Calling `wait()` or `result()` blocks until the subprocess finishes and
    caches the value so subsequent calls are cheap.

    If the subprocess is killed by SIGKILL (e.g., OOM killer), _consume()
    detects process death via polling and raises a descriptive error instead
    of hanging forever.
    """

    def __init__(self, result_queue, process=None):
        self._result_queue = result_queue
        self._process = process  # optional; used to detect unexpected death
        self._done = False
        self._value = None
        self._error = None

    def _consume(self):
        if not self._done:
            import queue as _queue
            while True:
                try:
                    status, value = self._result_queue.get(timeout=1.0)
                    if status == "ok":
                        self._value = value
                    else:
                        self._error = value
                    self._done = True
                    return
                except _queue.Empty:
                    if self._process is not None and not self._process.is_alive():
                        exit_code = self._process.exitcode
                        self._error = (
                            f"Worker subprocess died unexpectedly "
                            f"(exitcode={exit_code}). "
                            f"If exitcode=-9, this is likely an OOM kill — "
                            f"reduce max_buffer_size_adv/avrg or request more "
                            f"memory for the job."
                        )
                        self._done = True
                        return
                    # Process still alive — keep waiting.

    def wait(self):
        """Block until the result is available. Raises if the worker errored."""
        self._consume()
        if self._error is not None:
            raise RuntimeError(f"Worker subprocess error:\n{self._error}")

    def result(self):
        """Return the result, blocking if necessary. Raises on worker error."""
        self._consume()
        if self._error is not None:
            raise RuntimeError(f"Worker subprocess error:\n{self._error}")
        return self._value


def _la_worker_loop(t_prof, worker_id, cmd_queue, result_queue):
    """Entry point for each LearnerActor subprocess.

    Creates a LearnerActor, then runs an event loop reading
    (method_name, args) tuples from cmd_queue and writing results to
    result_queue.  The special sentinel "SHUTDOWN" exits the loop.
    """
    # Limit each worker to 1 PyTorch intra-op thread.  Without this, every
    # subprocess spawns as many threads as there are CPU cores, causing
    # massive thread contention when N workers run concurrently on N cores.
    import torch as _torch
    _torch.set_num_threads(1)
    _torch.set_num_interop_threads(1)

    try:
        from DeepCFR.workers.la.local import LearnerActor
        la = LearnerActor(t_prof=t_prof, worker_id=worker_id, chief_ref=None)
    except Exception:
        result_queue.put(("error", traceback.format_exc()))
        return

    while True:
        try:
            cmd = cmd_queue.get()
        except Exception:
            break

        if cmd == "SHUTDOWN":
            break

        method_name, args = cmd
        try:
            result = getattr(la, method_name)(*args)
            result_queue.put(("ok", result))
        except Exception:
            result_queue.put(("error", traceback.format_exc()))


class LAProxy:
    """Proxy for a LearnerActor running in a subprocess.

    Each method queues a command to the subprocess and returns an MPFuture
    immediately, enabling the main process to dispatch to all LAs in
    parallel and wait for them collectively.
    """

    def __init__(self, t_prof, worker_id):
        self._cmd_queue = mp.Queue()
        self._result_queue = mp.Queue()
        self._process = mp.Process(
            target=_la_worker_loop,
            args=(t_prof, worker_id, self._cmd_queue, self._result_queue),
            daemon=True,
        )
        self._process.start()

    def _call(self, method_name, *args):
        self._cmd_queue.put((method_name, args))
        return MPFuture(self._result_queue, process=self._process)

    def generate_data(self, traverser, cfr_iter, n_traversals=None):
        return self._call("generate_data", traverser, cfr_iter, n_traversals)

    def get_adv_batch(self, p_id):
        return self._call("get_adv_batch", p_id)

    def get_adv_all_data(self, p_id, max_samples=None):
        """Return all data from the adv buffer (or up to max_samples) for LightGBM training."""
        return self._call("get_adv_all_data", p_id, max_samples)

    def get_avrg_batch(self, p_id):
        return self._call("get_avrg_batch", p_id)

    def get_adv_grads(self, p_id):
        return self._call("get_adv_grads", p_id)

    def get_avrg_grads(self, p_id):
        return self._call("get_avrg_grads", p_id)

    def get_loss_last_batch_adv(self, p_id):
        return self._call("get_loss_last_batch_adv", p_id)

    def get_loss_last_batch_avrg(self, p_id):
        return self._call("get_loss_last_batch_avrg", p_id)

    def update(self, adv_state_dicts=None, avrg_state_dicts=None):
        return self._call("update", adv_state_dicts, avrg_state_dicts)

    def checkpoint(self, curr_step):
        return self._call("checkpoint", curr_step)

    def load_checkpoint(self, name_to_load, step):
        return self._call("load_checkpoint", name_to_load, step)

    def shutdown(self):
        """Send shutdown signal and wait for the subprocess to exit."""
        try:
            self._cmd_queue.put("SHUTDOWN")
            self._process.join(timeout=10)
        except Exception:
            pass
        if self._process.is_alive():
            self._process.terminate()
