# Deep CFR & Single Deep CFR

A scalable implementation of [Deep CFR](https://arxiv.org/pdf/1811.00164.pdf) [1] and its successor
[Single Deep CFR (SD-CFR)](https://arxiv.org/pdf/1901.07621.pdf) [2] in the
[PokerRL-2025](https://github.com/theGholland/PokerRL-2025) framework.

This codebase is designed for:
- Researchers to compare new methods to these baselines.
- Anyone wanting to learn about Deep RL in imperfect information games.

This implementation can be run on your local machine and on hundreds of cores on AWS or SLURM clusters.

## About this fork

This repository is based on the original [Deep-CFR](https://github.com/TinkeringCode/Deep-CFR) implementation by Eric Steinberger, extended for our experiments comparing **neural-network** and **LightGBM** advantage approximators in SD-CFR on **Flop5Holdem** and **Leduc**.

Main additions over the original repo:

- **LightGBM backend** for the advantage network (`--adv-model-type lightgbm`), with configurable hyperparameters and multi-threaded training.
- **SLURM scripts** under `slurm/` for training, checkpoint evaluation, and batch H2H matrix jobs.
- **`evaluate_checkpoint.py`** for head-to-head, vs-uniform, and LBR evaluation of saved checkpoints.
- **Analysis scripts** under `scripts/` to aggregate H2H matrix results and plot learning curves.
- **Automatic resource detection**, timestamped TensorBoard logging, GPU device flags, and per-actor performance stats (see fork notes below).

Checkpoints and logs are written under `~/poker_ai_data/` by default (eval agents, training logs, TensorBoard).

---

## Our experiments and results

We ran SD-CFR (SINGLE mode, random initialization) on **Flop5Holdem** with two advantage backends:

| Backend | Script / config | Notes |
|--------|------------------|--------|
| **NN (medium)** | `slurm/submit_flop5.sh` | Feedforward MLP on GPU parameter server |
| **NN (small / large)** | `submit_flop5_nn_small.sh`, `submit_flop5_nn_large.sh` | ~2× smaller / larger networks |
| **LightGBM (CPU)** | `slurm/submit_flop5_lightgbm_cpu.sh` | Gradient-boosted trees for advantages |

Each configuration was trained with **multiple independent seeds** (`--run-id 0–4` or higher), producing experiment names like `EXPERIMENT_SD-CFR_vs_Deep-CFR_FHP_NN_run0` and `EXPERIMENT_SD-CFR_vs_Deep-CFR_FHP_LightGBM_run5`.

### Head-to-head evaluation (Flop5Holdem)

After training, we evaluated checkpoints **head-to-head** (2M hands per seat) in a full matrix: every LGBM run vs every NN run at each iteration. Results are summarized in `slurm_out/<experiment_dir>/`:

- **`h2h-matrix-*.out`** — raw SLURM output per iteration (25 pairwise results + summary table).
- **`h2h_aggregated.json`** — per-iteration averages (total, per LGBM run, and all 25 pairs).
- **`h2h_plot.png`** — learning curve (LGBM perspective; positive = LGBM ahead).

Example findings from our runs (5×5 matrix, iteration 149):

| Setting | Total average (LGBM vs NN) |
|--------|----------------------------|
| Larger LightGBM budget (`slurm_out/big_lgbm/`) | ~**+268** MBB/G |
| Smaller LightGBM budget (`slurm_out/small_lgbm/`) | ~**+121** MBB/G |

Both configs show LGBM ahead on average at late iterations, with substantial run-to-run variance across the 25 pairings. Early iterations (especially iteration 0) are not meaningful for comparison — agents are essentially untrained. NN training is much faster per iteration; LightGBM iterations can take several minutes once buffers fill.

Comparison plots (e.g. two LGBM configs on one figure) are produced with `scripts/plot_h2h_results_compare.py`.

### Leduc exploitability

On **Leduc**, we compared NN vs LightGBM exploitability using `paper_experiment_leduc_exploitability_comparison.py` and `slurm/submit_leduc_*.sh`. LightGBM can reach lower exploitability early; NN often catches up at later iterations. See `slurm_out/leduc/` for job outputs.

---

## Reproducing our results

### 1. Setup

```bash
pip install -r requirements.txt
pip install .   # optional: install DeepCFR package
```

Training and evaluation expect checkpoints under `~/poker_ai_data/`. Activate your Python environment and adjust module loads in the SLURM scripts for your cluster.

### 2. Training (SLURM)

From the project root:

```bash
# Single NN run (medium network, run-id 0)
sbatch slurm/submit_flop5.sh 0

# Single LightGBM CPU run
sbatch slurm/submit_flop5_lightgbm_cpu.sh 5

# Submit multiple runs (edit scripts for run-id ranges)
python slurm/submit_all_runs.py
```

Local (non-SLURM) equivalent:

```bash
python paper_experiment_sdcfr_vs_deepcfr_h2h.py \
  --adv-model-type nn --run-id 0 --n-workers 62

python paper_experiment_sdcfr_vs_deepcfr_h2h.py \
  --adv-model-type lightgbm --adv-lgbm-device-type cpu --run-id 5 --n-workers 62
```

Monitor training: `tensorboard --logdir ~/poker_ai_data/logs`

### 3. Head-to-head evaluation (SLURM)

Edit experiment lists in `slurm/submit_evaluate_checkpoints_matrix.sh`, then:

```bash
# One job, one iteration
./slurm/submit_evaluate_checkpoints_matrix.sh

# One job per iteration (0, 5, 10, …, 145)
./slurm/submit_evaluate_checkpoints_matrix.sh --iter-range 0 145 5
```

Or evaluate a single pair locally:

```bash
python evaluate_checkpoint.py --mode h2h \
  --experiment1 EXPERIMENT_SD-CFR_vs_Deep-CFR_FHP_LightGBM_run5 \
  --experiment2 EXPERIMENT_SD-CFR_vs_Deep-CFR_FHP_NN_run0 \
  --iteration1 149 --iteration2 149 \
  --n-hands 2000000 --n-workers 22
```

Use `--iteration1 0 --iteration2 0` to compare initialized (untrained) agents.

### 4. Aggregate and plot H2H results

```bash
# Aggregate all h2h-matrix-*.out files in a directory
python scripts/aggregate_h2h_results.py slurm_out/big_lgbm

# Single learning curve
python scripts/plot_h2h_results.py slurm_out/big_lgbm/h2h_aggregated.json

# Compare two conditions on one plot
python scripts/plot_h2h_results_compare.py \
  slurm_out/big_lgbm/h2h_aggregated.json "Equal data" \
  slurm_out/small_lgbm/h2h_aggregated.json "Equal time" \
  -o slurm_out/h2h_compare.png

# Optional: plot all 25 pairwise series
python scripts/plot_h2h_results_compare.py ... --plot-all-pairs
```

### 5. Job resource report (SLURM)

On the cluster, summarize wall-clock time and memory per training job:

```bash
bash scripts/slurm_seff_report.sh
# → slurm_out/slurm_seff_report.txt
```

### 6. Leduc exploitability

```bash
sbatch slurm/submit_leduc_nn.sh
sbatch slurm/submit_leduc_lgbm.sh
# or
sbatch slurm/submit_leduc_both.sh
```

---

### Fork-specific notes (original README)

- **Automatic resource detection**: This fork probes available CPU cores, GPUs, and memory to size
  Ray actors and adjust batch sizes without manual configuration.
- **Linux shared memory**: PyTorch and Ray rely on `/dev/shm` for high-throughput data exchange.
  Increase this tmpfs (e.g., `sudo mount -o remount,size=32G /dev/shm` or Docker's `--shm-size`)
  to fully utilize system resources.
- **Additional enhancements over the original repository**: timestamped TensorBoard logging,
  Ray dashboard enabled by default, GPU device selection flags, and per-actor resource statistics
  recorded during training.

### Reproducing Results from Single Deep CFR (Steinberger 2019) [2]
The run-script `DeepCFR/paper_experiment_sdcfr_vs_deepcfr_h2h.py` launches one run of the Head-to-Head performance
comparison between Single Deep CFR and Deep CFR as presented in [2]. We ran the experiments on an m5.12xlarge instance
where we disabled hyper-threading. We set the instance up for distributed runs as explained in
[PokerRL-2025](https://github.com/theGholland/PokerRL-2025). To reproduce, you can simply clone this repository onto the
instance and start the script via
```
git clone https://github.com/TinkeringCode/Deep-CFR.git
cd Deep-CFR
python paper_experiment_sdcfr_vs_deepcfr_h2h.py
```
and watch the results coming in at `INSTANCE_IP:8888` in your browser.

VERY IMPORTANT NOTES:
- This implementation defines an iteration as one sequential update for BOTH players. Thus, **iteration 300 in the plot in [2]
  is equivalent to iteration 150 in the Tensorboard logs!**
- Results on iteration 0 have no meaning since they compare a random neural network to an exactly uniform strategy.


 
The action-probability comparison was conducted on a single CPU using `analyze_sdcfr_vs_dcfr_strategy.py`.
The root directory also contains scripts to reproduce our experiments on exploitability in Leduc and BigLeduc, and 
the experiment analyzing the effect of reservoir sampling on B^M with various capacities.


## (Single) Deep CFR on your Local Machine

### Install locally
This project runs on Python 3.12 and officially supports Linux (Mac has not been tested).

Install the project and its dependencies with

```
pip install .
```

This will register the `DeepCFR` package so it can be imported from anywhere. If you only want to install the
dependencies without installing the package itself, use

```
pip install -r requirements.txt
```


### Running experiments locally
To monitor training progress, launch TensorBoard in a separate terminal with

```
tensorboard --logdir ~/poker_ai_data/logs
```

Then open `http://localhost:6006` in your browser to view logs.  Ray's own
dashboard is started automatically and exposes cluster metrics at
`http://localhost:8265` (replace `localhost` with your machine's IP if running
remotely).  To run Deep CFR or SD-CFR with custom hyperparameters in any Poker
game supported by PokerRL-2025, build a script similar to `DeepCFR/leduc_example.py`. Run-scripts define
the hyperparameters, the game to be played, and the evaluation metrics. Here is a very minimalistic example showing a
few of the available settings:

```
from PokerRL.game.games import StandardLeduc  # or any other game

from DeepCFR.EvalAgentDeepCFR import EvalAgentDeepCFR
from DeepCFR.TrainingProfile import TrainingProfile
from DeepCFR.workers.driver.Driver import Driver

if __name__ == '__main__':
    ctrl = Driver(t_prof=TrainingProfile(name="SD-CFR_LEDUC_EXAMPLE",
    
                                         eval_agent_export_freq=20,  # export API to play against the agent
                                         
                                         nn_type="feedforward", # we also support recurrent nets
                                         max_buffer_size_adv=3e6,
                                         n_traversals_per_iter=1500,
                                         n_batches_adv_training=750,
                                         init_adv_model="last", # "last" or "random"

                                         game_cls=StandardLeduc, # The game to play     
                                         
                                         eval_modes_of_algo=(
                                             EvalAgentDeepCFR.EVAL_MODE_SINGLE,  # Single Deep CFR (SD-CFR)
                                         ),

                                         DISTRIBUTED=False, # Run locally
                                         ),
                  eval_methods={
                      "br": 3, # evaluate Best Response every 3 iterations.
                  })
    ctrl.run()
```

### Selecting GPU devices
Training scripts accept `--device-training`, `--device-parameter-server`, and `--device-inference` flags.
Each flag takes `cpu`, `cuda`, `cuda:<id>`, or `auto` (the default). When set to a CUDA device the
corresponding Ray worker reserves GPU resources; otherwise it runs on the CPU. Example:

```
python leduc_example.py --device-training cuda:0 --device-parameter-server cuda:0 --device-inference cuda:0
```
Note that you can specify one or both averaging methods under `eval_modes_of_algo`.
Choosing both is useful to compare them as they will share the value networks! However, we showed in [2] that SD-CFR
is expected to perform better, is faster, and requires less memory.

### Monitoring per-actor resource utilization
Each `LearnerActor` now records the wall-clock time, CPU usage, and (when training on CUDA) the GPU memory
and utilization consumed by its `generate_data` and `update` loops.  These statistics are aggregated and written to
TensorBoard via `GenerateData/*` and `Update/*` tags in the corresponding `*_Perf` experiment for each actor.

To tune performance, watch these graphs while adjusting the `TrainingProfile`'s `num_cpus` and `num_gpus` values per
actor.  Underutilized CPUs or GPUs suggest lowering the respective counts to schedule more actors, whereas sustained
values near 100% indicate a need for more resources or fewer workers per machine.

### Controlling per-actor memory
Ray kills workers that exceed their reserved memory.  By default the driver
divides roughly 80% of system RAM equally among learner-actors and parameter
servers.  Pass `memory_per_worker` to `TrainingProfile` to override this
allocation or set it to `0` to disable the limit entirely.  You can also scale
the automatic estimate for larger models with
`memory_per_worker_multiplier`.  Increasing these values helps prevent
premature actor death when networks require more RAM than the default
reservation.
                                         

## Cloud & Clusters
For deployment on AWS, whether single-core, many-core distributed, or on a cluster, please first follow
the tutorial in the corresponding section of [PokerRL-2025](https://github.com/theGholland/PokerRL-2025)'s README.

We recommend forking this repository so you can write your own scripts but still have remote access through git.
In your run-script set either the `DISTRIBUTED` or the `CLUSTER` option of the TrainingProfile to True
(see e.g. `DeepCFR/paper_experiment_sdcfr_vs_deepcfr_h2h.py`).
Moreover, you should specify the number of `LearnerActor` and evaluator workers (if applicable) you want to deploy.
Note that hyperparmeters ending with "_per_la" (e.g. the batch size) are effectively multiplied by the number of
workers. 

When running in DISTRIBUTED mode (i.e. one machine, many cores), simply ssh onto your AWS instance, get your code
onto it (e.g. through git cloning your forked repo) and start your run-script.
To fire up a cluster, define a `.yaml` cluster configuration that properly sets up your workers. Each of them
should have a copy of your forked repo as well as all dependencies on it.
Use `ray up ...` in an ssh session to the head of the cluster to start the job - more detailed instructions about 
the underlying framework we use for distributed computing can be found at [ray](https://github.com/ray-project/ray).





## Citing
If you use this repository in your research, you can cite it by citing PokerRL-2025 as follows:
```
@misc{steinberger2019pokerrl,
    author = {Eric Steinberger},
    title = {PokerRL-2025},
    year = {2019},
    publisher = {GitHub},
    journal = {GitHub repository},
    howpublished = {\url{https://github.com/theGholland/PokerRL-2025}},
}
```




## Authors
* **Eric Steinberger**





## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.





## References
[1] Brown, Noam, et al. "Deep Counterfactual Regret Minimization." arXiv preprint arXiv:1811.00164 (2018).

[2] Steinberger, Eric. "Single Deep Counterfactual Regret Minimization." arXiv preprint arXiv:1901.07621 (2019).
