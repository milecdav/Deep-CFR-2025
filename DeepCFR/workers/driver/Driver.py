import atexit
import os
import re
import socket
import subprocess
import time
import warnings
from datetime import datetime

import psutil
import torch

from DeepCFR.EvalAgentDeepCFR import EvalAgentDeepCFR
from DeepCFR.workers.chief.local import Chief
from DeepCFR.workers.driver._HighLevelAlgo import HighLevelAlgo
from DeepCFR.workers.mp_backend import LAProxy
from DeepCFR.workers.ps.local import ParameterServer
from DeepCFR.utils.device import resolve_device
from PokerRL._.TensorboardLogger import TensorboardLogger
from PokerRL.rl.base_cls.workers.DriverBase import DriverBase


class Driver(DriverBase):

    def __init__(self, t_prof, eval_methods, n_iterations=None, iteration_to_import=None, name_to_import=None):
        # Always use local worker classes (no Ray distribution).
        # Determine the log directory before calling super().__init__ so that
        # any base-class code that checks t_prof.path_log_storage sees the
        # final value.
        if not getattr(t_prof, "path_log_storage", None):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            sanitized_name = re.sub(r"[^\w.-]", "_", str(t_prof.name))
            log_root = os.path.join(
                os.path.expanduser("~"), "poker_ai_data", "logs", sanitized_name
            )
            t_prof.path_log_storage = os.path.join(log_root, timestamp)

        os.makedirs(t_prof.path_log_storage, exist_ok=True)

        # DriverBase creates Chief (via self._ray.create_worker which is now a
        # direct instantiation) and eval workers (BR, H2H, etc.).
        super().__init__(
            t_prof=t_prof,
            eval_methods=eval_methods,
            n_iterations=n_iterations,
            iteration_to_import=iteration_to_import,
            name_to_import=name_to_import,
            chief_cls=Chief,
            eval_agent_cls=EvalAgentDeepCFR,
        )

        # Resolve devices without Ray resource detection.
        def _is_cuda(device):
            return (
                (isinstance(device, torch.device) and device.type == "cuda")
                or (isinstance(device, str) and device.startswith("cuda"))
            )

        adv_spec = t_prof.module_args["adv_training"].device_training
        inf_spec = t_prof.device_inference
        ps_spec = t_prof.device_parameter_server

        any_cuda_requested = any(_is_cuda(d) for d in (adv_spec, inf_spec, ps_spec))

        if any_cuda_requested:
            try:
                cuda_available = torch.cuda.is_available()
            except Exception:
                cuda_available = False

            def _resolve(spec):
                if isinstance(spec, str) and spec.lower() == "auto":
                    spec = "cuda" if cuda_available else "cpu"
                return resolve_device(spec)

            def _ensure_device(dev, name):
                if _is_cuda(dev) and not cuda_available:
                    warnings.warn(
                        f"CUDA device requested for {name} but GPUs are unavailable; falling back to CPU.",
                        RuntimeWarning,
                    )
                    return torch.device("cpu")
                return dev
        else:
            cuda_available = False

            def _resolve(_):
                return torch.device("cpu")

            def _ensure_device(dev, _name):
                return dev

        adv_device = _ensure_device(_resolve(adv_spec), "adv_training")
        inf_device = _ensure_device(_resolve(inf_spec), "inference")
        ps_device = _ensure_device(_resolve(ps_spec), "parameter_server")  # noqa: F841

        if t_prof.log_verbose:
            print(f"TensorBoard logs will be written to {t_prof.path_log_storage}")

        # Recreate logger with the correct log path.
        self.logger = TensorboardLogger(
            name=t_prof.name,
            chief_handle=self.chief_handle,
            path_log_storage=t_prof.path_log_storage,
            runs_distributed=False,
            runs_cluster=False,
        )

        self._tb_proc = None
        self._tb_port = None

        if "h2h" in list(eval_methods.keys()):
            assert EvalAgentDeepCFR.EVAL_MODE_SINGLE in t_prof.eval_modes_of_algo
            assert EvalAgentDeepCFR.EVAL_MODE_AVRG_NET in t_prof.eval_modes_of_algo
            self._ray.remote(
                self.eval_masters["h2h"][0].set_modes,
                [EvalAgentDeepCFR.EVAL_MODE_SINGLE, EvalAgentDeepCFR.EVAL_MODE_AVRG_NET],
            )

        # Create ParameterServers in the main process (no subprocesses).
        print("Creating Parameter Servers...")
        self.ps_handles = [
            ParameterServer(t_prof, p, self.chief_handle)
            for p in range(t_prof.n_seats)
        ]

        # Create LearnerActors as subprocesses via LAProxy.
        print("Creating LAs...")
        self.la_handles = [
            LAProxy(t_prof=t_prof, worker_id=i)
            for i in range(t_prof.n_learner_actors)
        ]

        # Register atexit to clean up subprocesses on exit.
        la_handles_ref = self.la_handles

        def _shutdown_las():
            for proxy in la_handles_ref:
                try:
                    proxy.shutdown()
                except Exception:
                    pass

        atexit.register(_shutdown_las)

        self._ray.wait([
            self._ray.remote(self.chief_handle.set_ps_handle, *self.ps_handles),
            self._ray.remote(self.chief_handle.set_la_handles, *self.la_handles),
        ])

        print("Created and initialized Workers")

        self.algo = HighLevelAlgo(
            t_prof=t_prof,
            la_handles=self.la_handles,
            ps_handles=self.ps_handles,
            chief_handle=self.chief_handle,
        )

        # Start TensorBoard after worker subprocesses are up.
        self._start_tensorboard()

        self._AVRG = EvalAgentDeepCFR.EVAL_MODE_AVRG_NET in self._t_prof.eval_modes_of_algo
        self._SINGLE = EvalAgentDeepCFR.EVAL_MODE_SINGLE in self._t_prof.eval_modes_of_algo

        self._maybe_load_checkpoint_init()

    def _start_tensorboard(self):
        if not getattr(self._t_prof, "log_verbose", False):
            return

        def _get_free_port():
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("", 0))
                return s.getsockname()[1]

        tb_port = _get_free_port()
        tb_cmd = [
            "tensorboard",
            "--logdir", self._t_prof.path_log_storage,
            "--host", "0.0.0.0",
            "--port", str(tb_port),
        ]
        try:
            self._tb_proc = subprocess.Popen(
                tb_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT,
            )
            self._tb_port = tb_port
            print(f"TensorBoard listening on http://0.0.0.0:{tb_port}/")
        except Exception as ex:
            self._tb_proc = None
            print(f"Failed to start TensorBoard: {ex}")

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("_tb_proc", None)
        state.pop("_tb_port", None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._tb_proc = None
        self._tb_port = None

    def run(self):
        print("Setting stuff up...")
        self.algo.init()

        print("Starting Training...")
        try:
            for _iter_nr in range(10000000 if self.n_iterations is None else self.n_iterations):
                t_iter_start = time.time()
                print("Iteration: ", self._cfr_iter)

                # Maybe train AVRG
                avrg_times = None
                if self._AVRG and self._any_eval_needs_avrg_net():
                    avrg_times = self.algo.train_average_nets(cfr_iter=_iter_nr)

                # Eval
                self.evaluate()

                # Log
                if self._cfr_iter % self._t_prof.log_export_freq == 0:
                    self.save_logs()
                self.periodically_export_eval_agent()

                # Iteration
                iter_times = self.algo.run_one_iter_alternating_update(cfr_iter=self._cfr_iter)

                t_iter_total = time.time() - t_iter_start
                print(
                    "Iter total:", f"{t_iter_total:.3f}s.",
                    "  ||  Generating Data:", str(iter_times["t_generating_data"]) + "s.",
                    "  ||  Batch Fetch ADV", str(iter_times["t_batch_fetch_adv"]) + "s.",
                    "  ||  Trained ADV", str(iter_times["t_training_adv"]) + "s.",
                    "\n"
                )
                if self._AVRG and avrg_times:
                    print(
                        "Batch Fetch AVRG", str(avrg_times["t_batch_fetch_avrg"]) + "s.",
                        "  ||  Trained AVRG", str(avrg_times["t_training_avrg"]) + "s.",
                        "\n"
                    )

                self._cfr_iter += 1

                # Checkpoint
                self.periodically_checkpoint()

        except RuntimeError as e:
            print(f"Training stopped: {e}")
        finally:
            try:
                self._ray.wait([self._ray.remote(self.chief_handle.flush_tb_writers)])
                self._ray.wait([self._ray.remote(self.chief_handle.close_tb_writers)])
            except Exception:
                pass

            # Shut down LA subprocesses.
            for proxy in self.la_handles:
                try:
                    proxy.shutdown()
                except Exception:
                    pass

            if getattr(self, "_tb_proc", None):
                self._tb_proc.terminate()
                try:
                    self._tb_proc.wait(timeout=5)
                except Exception:
                    pass

    def _any_eval_needs_avrg_net(self):
        for e in list(self.eval_masters.values()):
            if self._cfr_iter % e[1] == 0:
                return True
        return False

    def checkpoint(self, **kwargs):
        # Checkpoint all workers sequentially to avoid RAM overload.
        # LAs are subprocesses — consume MPFutures to ensure each checkpoint completes.
        from DeepCFR.workers.mp_backend import MPFuture
        for w in self.la_handles:
            f = self._ray.remote(w.checkpoint, self._cfr_iter)
            if isinstance(f, MPFuture):
                f.result()
        # PS and Chief are in-process — direct calls.
        for w in self.ps_handles + [self.chief_handle]:
            self._ray.wait([self._ray.remote(w.checkpoint, self._cfr_iter)])

        s = [self._cfr_iter]
        if self._cfr_iter > self._t_prof.checkpoint_freq + 1:
            s.append(self._cfr_iter - self._t_prof.checkpoint_freq)
        self._delete_past_checkpoints(steps_not_to_delete=s)

    def load_checkpoint(self, step, name_to_load):
        from DeepCFR.workers.mp_backend import MPFuture
        for w in self.la_handles:
            f = self._ray.remote(w.load_checkpoint, name_to_load, step)
            if isinstance(f, MPFuture):
                f.result()
        for w in self.ps_handles + [self.chief_handle]:
            self._ray.wait([self._ray.remote(w.load_checkpoint, name_to_load, step)])
