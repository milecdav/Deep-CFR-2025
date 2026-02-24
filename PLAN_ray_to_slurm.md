# Plan: Migrate Deep-CFR from Ray to SLURM with Multiprocessing

## Status: IMPLEMENTED ✓

All steps completed on 2026-02-18. See "Implementation Notes" section for deviations from the original plan and debugging information.

---

## Context

The Deep-CFR-2025 repository uses Ray for distributed computing (actor model, remote calls, object store) and TensorBoard on localhost / Ray Dashboard for results viewing. The goal is to replace Ray with `torch.multiprocessing` for single-node SLURM cluster execution, and replace localhost-based result viewing with file-based CSV logging (plus offline TensorBoard files).

**Key insight:** PokerRL's MaybeRay class already has a passthrough mode when `DISTRIBUTED=False` where all `ray.get()`, `ray.wait()`, `ray.remote()` calls become no-ops. We leverage this by keeping `DISTRIBUTED=False` internally, but adding our own multiprocessing layer for LearnerActor parallelism. No PokerRL modifications needed — we monkey-patch only `MaybeRay.wait` and `MaybeRay.get` to handle `MPFuture` objects.

**PokerRL-2025 is now a local directory** at `Deep-CFR-2025/PokerRL-2025/`. It is NOT modified — monkey-patching in `DeepCFR/__init__.py` is the only mechanism used to extend MaybeRay. If you need to modify PokerRL itself, change requirements.txt from the git URL to a local path and reinstall:
```
PokerRL @ file:///home/milecdav/Deep-CFR-2025/PokerRL-2025
```
then `pip install -e PokerRL-2025/`.

**Cluster environment:**
- Module: `ml Python/3.11.5-GCCcore-13.2.0`
- Venv: `source /home/milecdav/Deep-CFR-2025/venv/bin/activate`

---

## Architecture

```
Main Process (SLURM job):
  ├── Driver (orchestrator)
  ├── Chief (logging, strategy buffers)  — in-process, created by DriverBase
  ├── ParameterServer[0..1] (networks)   — in-process, created in Driver.__init__
  └── Evaluators (BR, H2H, etc.)        — in-process, created by DriverBase

Worker Processes (multiprocessing.Process, spawn):
  ├── LearnerActor[0]  — subprocess driven by LAProxy
  ├── LearnerActor[1]  — subprocess driven by LAProxy
  ├── ...
  └── LearnerActor[N-1] — subprocess driven by LAProxy
```

**Communication:** Each `LAProxy` has two `multiprocessing.Queue`s:
- `cmd_queue`: main process → subprocess (sends `(method_name, args)` tuples or `"SHUTDOWN"`)
- `result_queue`: subprocess → main process (sends `("ok", value)` or `("error", traceback_str)`)

`MPFuture` wraps the `result_queue` read. It blocks on `.wait()` / `.result()` and caches the value so repeated calls are free.

**Why this split:** LearnerActors are compute-heavy (data generation, gradient computation). Chief, PS, and Evaluators are lightweight and benefit from being in-process — no serialization overhead for weight/gradient transfer between Driver↔PS.

---

## Data Flow (unchanged algorithm, new transport)

1. `_HighLevelAlgo._generate_traversals` calls `self._ray.remote(la.generate_data, p_id, cfr_iter)` for each LA
   - `self._ray.remote(fn, *args)` → `fn(*args)` (MaybeRay passthrough)
   - `la.generate_data(p_id, cfr_iter)` is an `LAProxy` method → puts command on `cmd_queue`, returns `MPFuture`
   - All LA subprocesses run `generate_data` in parallel
   - `self._ray.wait(futures_list)` → calls `future.wait()` on each `MPFuture`, blocking until all done

2. `_get_adv_gradients` calls `self._ray.remote(la.get_adv_grads, p_id)` for each LA → list of `MPFuture`s
   - `self._ray.wait(grads_refs)` → all futures consumed (results cached in MPFuture)
   - `grads_vals = self._ray.get(grads_refs)` → returns list of cached gradient dicts
   - Filters out None results: `grads_refs = [ref for ref, val in zip(...) if val is not None]`
   - These filtered `MPFuture` objects are passed to `ps.apply_grads_adv(grads_from_all_las)`

3. Inside `ParameterServerBase._apply_grads` (PokerRL code, unmodified):
   - `self._ray.get(batch)` where batch contains `MPFuture`s → returns cached gradient dicts
   - `self._ray.grads_to_torch(g, device)` → moves torch tensors to device (passthrough, no numpy conversion)

4. `ps.get_adv_weights()` → direct call (PS is in-process) → returns state dict (torch tensors)
   - State dict passed to `la.update(w_adv, w_avrg)` → serialized through Queue (torch tensors are picklable)
   - Inside LA subprocess: `self._ray.state_dict_to_torch(self._ray.get(adv_state_dicts[p_id]), device)` → passthrough `.get()` + `.to(device)`

**Critical sequencing invariant:** Each `LAProxy` subprocess processes commands sequentially (one at a time). The main process ALWAYS waits for a command's result before sending the next command to the same LA. This means the `result_queue` is always read in the correct order and `MPFuture` caching is safe.

---

## Files Changed

### Created
| File | Purpose |
|------|---------|
| `DeepCFR/workers/mp_backend.py` | `MPFuture`, `LAProxy`, `_la_worker_loop` |
| `slurm/submit_training.sh` | SLURM template for leduc_example.py |
| `slurm/submit_leduc.sh` | SLURM template for leduc exploitability |
| `slurm/submit_flop5.sh` | SLURM template for Flop5Holdem H2H |
| `PLAN_ray_to_slurm.md` | This file |

### Heavily Modified
| File | Key Changes |
|------|------------|
| `DeepCFR/__init__.py` | Stripped to: (1) `set_start_method("spawn")`, (2) patch `MaybeRay.wait` and `MaybeRay.get` to handle `MPFuture` |
| `DeepCFR/workers/driver/Driver.py` | Removed Ray imports + resource detection; log dir → `~/poker_ai_data/logs/`; PS created directly; LAs created as `LAProxy`; atexit cleanup; removed `_chief_name` / `ray.get_actor` pattern |
| `DeepCFR/workers/driver/_HighLevelAlgo.py` | Removed `import ray`, `_safe_wait`, `_safe_get`, `_handle_actor_error`; replaced with direct `self._ray.wait` / `self._ray.get` calls |
| `DeepCFR/TrainingProfile.py` | Added `n_workers` alias param; `n_learner_actors` always set to `n_learner_actor_workers` (no `DISTRIBUTED` conditional); always passes `DISTRIBUTED=False, CLUSTER=False` to super |

### Updated
| File | Key Changes |
|------|------------|
| `DeepCFR/workers/la/local.py` | Removed `ray.get_actor(chief_ref)` block; `chief_ref` accepted directly (or `None`) |
| `DeepCFR/workers/ps/local.py` | Removed `ray.get_actor(chief_ref)` block; `chief_ref` accepted directly |
| `requirements.txt` | Removed `ray[default]==2.48.0` and `requests>=2.32.4` |
| `leduc_example.py` | Removed `DISTRIBUTED=False`; added `--n-workers` CLI arg; uses `n_workers=args.n_workers` |
| `paper_experiment_*.py` (7 files) | Same pattern: removed `DISTRIBUTED`/`CLUSTER`/`n_learner_actor_workers`; added `--n-workers` |

### Deleted
- `DeepCFR/workers/chief/dist.py`
- `DeepCFR/workers/la/dist.py`
- `DeepCFR/workers/ps/dist.py`

### NOT Changed (intentionally)
- `DeepCFR/workers/chief/local.py` — no modifications needed; in-process, no Ray references
- `PokerRL-2025/` — no modifications; monkey-patching only
- `DeepCFR/utils/csv_logger.py` — not created; CSV logging was planned but deferred (TensorBoard files still work offline)

---

## Implementation Notes & Deviations from Original Plan

### What changed from the plan

**1. Minimal monkey-patching (simpler than planned)**

The original plan proposed patching many MaybeRay methods (`init_local`, `create_worker`, `remote`, `wait`, `get`, `put`, `state_dict_to_numpy`, `state_dict_to_torch`, `grads_to_numpy`, `grads_to_torch`). In practice, only `wait` and `get` needed patching because:
- `init_local()` is already a no-op when `runs_distributed=False`
- `create_worker(cls, *args)` already calls `cls(*args)` directly when not distributed
- `remote(fn, *args)` already calls `fn(*args)` directly when not distributed
- `state_dict_to_numpy/torch` and `grads_to_numpy/torch` all work correctly in passthrough mode with torch tensors

**2. No MPMaybeRay class created**

Instead of creating a separate `MPMaybeRay` class, we patch the existing `MaybeRay` instance methods directly. This is cleaner and requires fewer changes.

**3. No DriverBase patch**

The original plan included a `DriverBase.__init__` patch to zero out `_default_num_gpus`. This was only needed to prevent Ray from reserving GPU resources for actors. With no Ray, it's unnecessary.

**4. CSV logger not implemented**

Deferred. TensorBoard event files work fine for offline viewing (`tensorboard --logdir ~/poker_ai_data/logs/`). Add if needed.

**5. `_safe_wait`/`_safe_get` removed, not replaced with simple wrappers**

Calls like `self._safe_wait(refs)` are replaced with direct `self._ray.wait(refs)` throughout `_HighLevelAlgo.py`. The original methods also caught `ray.exceptions.RayActorError` for fault tolerance — this is gone. If a subprocess crashes, the error propagates as a `RuntimeError` via `MPFuture.result()`.

**6. `n_workers` is an alias, `n_learner_actor_workers` still works**

Both parameters are accepted by `TrainingProfile`. `n_workers` takes precedence if set. This preserves backward compat.

**7. `checkpoint()` signature in Driver.py**

The original `checkpoint(self, **kwargs)` is kept but internally now makes sequential calls to LA proxies (via queue, async) followed by direct calls to PS and Chief.

---

## Debugging Guide

### How to trace a failure in the subprocess

If a `LearnerActor` subprocess crashes, `MPFuture.result()` raises:
```
RuntimeError: Worker subprocess error:
Traceback (most recent call last):
  ...
```
The full traceback from the subprocess is included. Look at `_la_worker_loop` in `mp_backend.py` — it wraps every method call in try/except and sends the traceback as `("error", traceback_str)`.

### Subprocess startup failure

If a LA subprocess fails to start (e.g., import error, cuda init error), `_la_worker_loop` catches it and puts `("error", traceback)` on `result_queue`. The next `MPFuture.result()` call in the main process will then raise. Watch for errors during `Driver.__init__` → LA creation.

### Deadlock symptoms

If training hangs forever, likely causes:
1. A `MPFuture.wait()` is blocking because the subprocess never put a result — e.g., it crashed silently. Check subprocess via `proxy._process.is_alive()`.
2. A command was sent but the subprocess is still processing a previous command (shouldn't happen given the invariant, but check if `_safe_wait` was accidentally bypassed somewhere).
3. Queue full (unlikely, Queues are unbounded by default).

### "Too many open files" error

Each `LAProxy` creates 2 `multiprocessing.Queue`s. With many workers this can exhaust file descriptors. Increase with `ulimit -n 65536` before starting.

### CUDA in subprocesses

Subprocesses use `spawn` start method (set in `DeepCFR/__init__.py`). CUDA tensors can be passed through `multiprocessing.Queue` (they use shared memory automatically). If you see CUDA errors in subprocesses, verify:
- `set_start_method("spawn")` ran before any subprocess creation
- The subprocess imports `DeepCFR` (triggering `__init__.py` → `set_start_method`)

### Gradient flow: MPFuture re-use

In `_get_adv_gradients`:
```python
grads_refs = [self._ray.remote(la.get_adv_grads, p_id) for la in self._la_handles]
self._ray.wait(grads_refs)       # MPFuture._consume() called → result cached
grads_vals = self._ray.get(grads_refs)  # returns cached values
grads_refs = [ref for ref, val in ... if val is not None]  # filtered MPFutures
# grads_refs passed to ps.apply_grads_adv(...)
# Inside PS._apply_grads: self._ray.get(batch) → MPFuture.result() → cached value
```
The MPFuture caches on first consume. Subsequent `.result()` calls return the cache without reading from the queue again. This is correct and intentional.

### State dict serialization

State dicts (torch tensors) are serialized through `multiprocessing.Queue` using pickle. This works for CPU tensors. For CUDA tensors: torch uses CUDA IPC under the hood when pickling across processes on the same machine (spawn context). If this causes issues, add `.cpu()` in `ps.get_adv_weights()` before returning.

### Testing a single subprocess

```python
from DeepCFR.workers.mp_backend import LAProxy
from DeepCFR.TrainingProfile import TrainingProfile
from PokerRL.game.games import StandardLeduc

tp = TrainingProfile(name="test", n_workers=1, game_cls=StandardLeduc)
proxy = LAProxy(tp, worker_id=0)
# Send a no-op command
future = proxy.generate_data(traverser=0, cfr_iter=0)
future.wait()  # blocks until done
proxy.shutdown()
```

### Log directory

Training logs go to: `~/poker_ai_data/logs/{sanitized_name}/{timestamp}/`
- TensorBoard event files: `{log_dir}/` (various subdirs per experiment)
- View offline: `tensorboard --logdir ~/poker_ai_data/logs/`

---

## Original Implementation Steps (with completion status)

| Step | Description | Status |
|------|-------------|--------|
| 1 | Create `DeepCFR/workers/mp_backend.py` | ✓ Done |
| 2 | Create `DeepCFR/utils/csv_logger.py` | ✗ Deferred |
| 3 | Modify `DeepCFR/__init__.py` | ✓ Done (simpler than planned) |
| 4 | Modify `DeepCFR/TrainingProfile.py` | ✓ Done |
| 5 | Modify `DeepCFR/workers/driver/Driver.py` | ✓ Done |
| 6 | Modify `DeepCFR/workers/driver/_HighLevelAlgo.py` | ✓ Done |
| 7 | Modify `DeepCFR/workers/la/local.py` | ✓ Done |
| 8 | Modify `DeepCFR/workers/ps/local.py` | ✓ Done |
| 9 | Modify `DeepCFR/workers/chief/local.py` | ✗ Not needed |
| 10 | Delete `dist.py` files | ✓ Done |
| 11 | Update `requirements.txt` | ✓ Done |
| 12 | Update entry point scripts | ✓ Done |
| 13 | Create SLURM scripts | ✓ Done |
| 14 | Update tests | ✗ Not done (tests may need updating if they import Ray) |
