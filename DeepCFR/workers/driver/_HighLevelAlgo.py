import time

import torch

from DeepCFR.EvalAgentDeepCFR import EvalAgentDeepCFR
from DeepCFR.workers.mp_backend import MPFuture
from PokerRL.rl.base_cls.HighLevelAlgoBase import HighLevelAlgoBase as _HighLevelAlgoBase


class HighLevelAlgo(_HighLevelAlgoBase):

    def __init__(self, t_prof, la_handles, ps_handles, chief_handle):
        super().__init__(t_prof=t_prof, chief_handle=chief_handle, la_handles=la_handles)
        self._ps_handles = ps_handles
        self._all_p_aranged = list(range(self._t_prof.n_seats))

        self._AVRG = EvalAgentDeepCFR.EVAL_MODE_AVRG_NET in self._t_prof.eval_modes_of_algo
        self._SINGLE = EvalAgentDeepCFR.EVAL_MODE_SINGLE in self._t_prof.eval_modes_of_algo

        self._adv_args = t_prof.module_args["adv_training"]
        if self._AVRG:
            self._avrg_args = t_prof.module_args["avrg_training"]

        if self._t_prof.log_verbose:
            self._exp_adv_loss_handles = [
                self._ray.remote(self._chief_handle.create_experiment, f"P{p}/Adv_Loss")
                for p in range(self._t_prof.n_seats)
            ]
            self._exp_avrg_loss_handles = [
                self._ray.remote(self._chief_handle.create_experiment, f"P{p}/Avrg_Loss")
                for p in range(self._t_prof.n_seats)
            ] if self._AVRG else None
        else:
            self._exp_adv_loss_handles = None
            self._exp_avrg_loss_handles = None

    def init(self):
        if self._AVRG:
            self._update_leaner_actors(update_adv_for_plyrs=self._all_p_aranged,
                                       update_avrg_for_plyrs=self._all_p_aranged)
        else:
            self._update_leaner_actors(update_adv_for_plyrs=self._all_p_aranged)

    def run_one_iter_alternating_update(self, cfr_iter):
        t_generating_data = 0.0
        t_batch_fetch_adv = 0.0
        t_training_adv = 0.0

        for p_learning in range(self._t_prof.n_seats):
            self._update_leaner_actors(update_adv_for_plyrs=self._all_p_aranged)
            if self._t_prof.print_progress:
                print("Generating Data...")
            t0 = time.time()
            self._generate_traversals(p_id=p_learning, cfr_iter=cfr_iter)
            t_generating_data += time.time() - t0

            if self._t_prof.print_progress:
                print("Training Advantage Net...")
            _t_fetch, _t_train = self._train_adv(p_id=p_learning, cfr_iter=cfr_iter)
            t_batch_fetch_adv += _t_fetch
            t_training_adv += _t_train

            if self._SINGLE:
                if self._t_prof.print_progress:
                    print("Pushing new net to chief...")
                self._push_newest_adv_net_to_chief(p_id=p_learning, cfr_iter=cfr_iter)

        if self._t_prof.print_progress:
            print("Synchronizing...")
        self._update_leaner_actors(update_adv_for_plyrs=self._all_p_aranged)

        return {
            "t_generating_data": t_generating_data,
            "t_batch_fetch_adv": t_batch_fetch_adv,
            "t_training_adv": t_training_adv,
        }

    def train_average_nets(self, cfr_iter):
        if self._t_prof.print_progress:
            print("Training Average Nets...")
        t_batch_fetch_avrg = 0.0
        t_training_avrg = 0.0
        for p in range(self._t_prof.n_seats):
            _f, _t = self._train_avrg(p_id=p, cfr_iter=cfr_iter)
            t_batch_fetch_avrg += _f
            t_training_avrg += _t

        return {
            "t_batch_fetch_avrg": t_batch_fetch_avrg,
            "t_training_avrg": t_training_avrg,
        }

    def _concat_batches(self, batches):
        """Concatenate a list of batch tuples along the batch dimension.

        Each batch is a tuple of tensors (pub_obs, range_idxs, legal_masks,
        values, loss_weights).  Returns a single concatenated tuple, or the
        sole element if only one batch is provided.
        """
        if len(batches) == 1:
            return batches[0]
        return tuple(torch.cat([b[i] for b in batches], dim=0) for i in range(len(batches[0])))

    def _train_adv(self, p_id, cfr_iter):
        """Train the ADV net for player p_id.

        Batches are fetched from LA subprocesses in round-robin and trained
        locally on the ParameterServer.  The next batch request is
        pre-dispatched before the current training step begins so that
        subprocess IPC overlaps with NN compute, hiding the round-trip latency.
        """
        t_batch_fetch = 0.0
        t_training = 0.0

        ps = self._ps_handles[p_id]
        if ps is None:
            return t_batch_fetch, t_training
        self._ray.wait([
            self._ray.remote(ps.reset_adv_net, cfr_iter)
        ])

        n_workers = len(self._la_handles)
        n_batches = self._adv_args.n_batches_adv_training
        SMOOTHING = 200
        accumulated_averaged_loss = 0.0
        accumulated_loss_count = 0

        if n_batches == 0:
            return t_batch_fetch, t_training

        # Pre-dispatch the first batch so it is in-flight before the loop starts.
        pending_future = self._la_handles[0 % n_workers].get_adv_batch(p_id)

        for epoch_nr in range(n_batches):
            global_step = cfr_iter * n_batches + epoch_nr

            # Collect the already-dispatched batch (likely ready by now).
            t0 = time.time()
            batch = pending_future.result() if isinstance(pending_future, MPFuture) else pending_future
            t_batch_fetch += time.time() - t0

            # Pre-dispatch NEXT batch BEFORE training so IPC overlaps with NN compute.
            if epoch_nr + 1 < n_batches:
                pending_future = self._la_handles[(epoch_nr + 1) % n_workers].get_adv_batch(p_id)

            if batch is None:
                continue

            t0 = time.time()
            ps = self._ps_handles[p_id]
            if ps is None:
                return t_batch_fetch, t_training
            loss = self._ray.get(self._ray.remote(ps.train_adv_step, batch))
            t_training += time.time() - t0

            if loss is not None:
                accumulated_averaged_loss += loss
                accumulated_loss_count += 1

            if (
                self._t_prof.log_verbose
                and ((epoch_nr + 1) % SMOOTHING == 0)
                and accumulated_loss_count > 0
            ):
                self._ray.wait([
                    self._ray.remote(
                        self._chief_handle.add_scalar,
                        self._exp_adv_loss_handles[p_id],
                        "DCFR_NN_Losses/Advantage",
                        global_step,
                        accumulated_averaged_loss / accumulated_loss_count,
                    )
                ])
                accumulated_averaged_loss = 0.0
                accumulated_loss_count = 0

        return t_batch_fetch, t_training

    def _generate_traversals(self, p_id, cfr_iter):
        n_workers = len(self._la_handles)
        total = self._t_prof.n_traversals_per_iter
        base, remainder = divmod(total, n_workers)
        counts = [base + (1 if i < remainder else 0) for i in range(n_workers)]
        futures = [
            self._ray.remote(la.generate_data, p_id, cfr_iter, counts[i])
            for i, la in enumerate(self._la_handles)
        ]
        self._drain_la_futures(futures)

    def _update_leaner_actors(self, update_adv_for_plyrs=None, update_avrg_for_plyrs=None):
        assert isinstance(update_adv_for_plyrs, list) or update_adv_for_plyrs is None
        assert isinstance(update_avrg_for_plyrs, list) or update_avrg_for_plyrs is None

        _update_adv_per_p = [
            True if (update_adv_for_plyrs is not None) and (p in update_adv_for_plyrs) else False
            for p in range(self._t_prof.n_seats)
        ]
        _update_avrg_per_p = [
            True if (update_avrg_for_plyrs is not None) and (p in update_avrg_for_plyrs) else False
            for p in range(self._t_prof.n_seats)
        ]

        la_batches = []
        n = len(self._la_handles)
        c = 0
        while n > c:
            s = min(n, c + self._t_prof.max_n_las_sync_simultaneously)
            la_batches.append(self._la_handles[c:s])
            if type(la_batches[-1]) is not list:
                la_batches[-1] = [la_batches[-1]]
            c = s

        w_adv = [None for _ in range(self._t_prof.n_seats)]
        w_avrg = [None for _ in range(self._t_prof.n_seats)]
        for p_id in range(self._t_prof.n_seats):
            ps = self._ps_handles[p_id]
            w_adv[p_id] = None if (not _update_adv_per_p[p_id] or ps is None) else self._ray.remote(
                ps.get_adv_weights)
            ps = self._ps_handles[p_id]
            w_avrg[p_id] = None if (not _update_avrg_per_p[p_id] or ps is None) else self._ray.remote(
                ps.get_avrg_weights)

        for batch in la_batches:
            futures = [self._ray.remote(la.update, w_adv, w_avrg) for la in batch]
            self._drain_la_futures(futures)

    def _drain_la_futures(self, futures):
        """Consume MPFuture results from LAProxy calls.

        In non-distributed mode _ray.wait() is a no-op, so results from LA
        subprocess commands accumulate on each LA's result_queue.  Calling this
        after every dispatch prevents stale results from being consumed by
        subsequent get_adv_batch / get_avrg_batch calls.
        """
        for f in futures:
            if isinstance(f, MPFuture):
                f.result()

    def _push_newest_adv_net_to_chief(self, p_id, cfr_iter):
        ps = self._ps_handles[p_id]
        if ps is None:
            return
        self._ray.wait([self._ray.remote(
            self._chief_handle.add_new_iteration_strategy_model,
            p_id,
            self._ray.remote(ps.get_adv_weights),
            cfr_iter,
        )])

    def _train_avrg(self, p_id, cfr_iter):
        """Train the AVRG net for player p_id.

        Uses the same all-LA dispatch pattern as _train_adv so each gradient
        step processes n_workers × batch_size data points.
        """
        t_batch_fetch = 0.0
        t_training = 0.0

        ps = self._ps_handles[p_id]
        if ps is None:
            return t_batch_fetch, t_training
        self._ray.wait([self._ray.remote(ps.reset_avrg_net)])

        n_batches = self._avrg_args.n_batches_avrg_training
        SMOOTHING = 200
        accumulated_averaged_loss = 0.0
        accumulated_loss_count = 0

        if cfr_iter > 0 and n_batches > 0:
            # Pre-dispatch the first round of batches from ALL LAs simultaneously.
            pending_futures = [la.get_avrg_batch(p_id) for la in self._la_handles]

            for epoch_nr in range(n_batches):
                global_step = cfr_iter * self._adv_args.n_batches_adv_training + epoch_nr

                # Collect batches from all LAs (they ran in parallel).
                t0 = time.time()
                batches = [f.result() if isinstance(f, MPFuture) else f for f in pending_futures]
                t_batch_fetch += time.time() - t0

                # Pre-dispatch NEXT round BEFORE training so IPC overlaps with NN compute.
                if epoch_nr + 1 < n_batches:
                    pending_futures = [la.get_avrg_batch(p_id) for la in self._la_handles]

                batches = [b for b in batches if b is not None]
                if not batches:
                    continue

                combined_batch = self._concat_batches(batches)

                t0 = time.time()
                ps = self._ps_handles[p_id]
                if ps is None:
                    return t_batch_fetch, t_training
                loss = self._ray.get(self._ray.remote(ps.train_avrg_step, combined_batch))
                t_training += time.time() - t0

                if loss is not None:
                    accumulated_averaged_loss += loss
                    accumulated_loss_count += 1

                if self._t_prof.log_verbose and ((epoch_nr + 1) % SMOOTHING == 0) and accumulated_loss_count > 0:
                    self._ray.wait([
                        self._ray.remote(
                            self._chief_handle.add_scalar,
                            self._exp_avrg_loss_handles[p_id],
                            "DCFR_NN_Losses/Average",
                            global_step,
                            accumulated_averaged_loss / accumulated_loss_count,
                        )
                    ])
                    accumulated_averaged_loss = 0.0
                    accumulated_loss_count = 0

        return t_batch_fetch, t_training
