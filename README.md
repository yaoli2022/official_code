# Epigenetic Age Prediction with Sequence-Conditioned Co-Methylation Graph Networks

## Proposed Model

The proposed **sequence-conditioned co-methylation graph clock** jointly models
sample-specific DNA methylation, local DNA sequence, and relationships among
CpG sites to predict biological age.

Each sample is represented as a graph containing 20,318 CpG sites:

- **Node information:** sample-specific methylation values, genomic annotations,
and the 122-bp sequence surrounding each CpG site.
- **Sequence conditioning:** a shared three-layer 1D CNN encodes each local
sequence into a multiplicative gate and a compact sequence projection.
- **Graph structure:** edges represent co-methylation and genomic relationships
among CpG sites and are constructed from training samples only.
- **Graph propagation:** Principal Neighbourhood Aggregation (PNA) integrates
information across related CpGs.
- **Age prediction:** an MLP maps the resulting site representations to
biological age.

The sequence-conditioning components contain 11,267 trainable parameters,
accounting for only 0.05% of the complete model.

---



## Repository Layout

```
official_code/
├── main_code/     all scripts (listed below)
├── data/          input data (see "Required data")
├── baselines/     reference-clock assets and baseline outputs
└── cache/, configs/, runs/, results/   created at run time
```


| Script                                      | Purpose                                                                                                                                                        |
| ------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `main_code/build_graph.py`                  | Builds and caches the co-methylation graph of each CV fold (step 1 below).                                                                                     |
| `main_code/train.py`                        | Trains and evaluates one configuration: sequence variant x graph operator x fold x seed (step 2 below).                                                        |
| `main_code/make_configs.py`                 | Writes ready-made run lists to `configs/*.tsv`, one `train.py` argument set per line.                                                                          |
| `main_code/freeze_order.py`                 | Optional. Pins the dataset concatenation order to `cache/file_order.json` so that fold membership is identical on every machine.                               |
| `main_code/baselines_dlorig.py`             | DeepMAge and ResNetAge baselines, 5 folds x 3 seeds, on the same split as the graph model.                                                                     |
| `main_code/baselines_wave0cfg.py`           | Elastic Net and tuned-MLP baselines, 5 folds x 3 seeds, on the same split. Imports the helper module `wave0_baselines.py`, which must be placed next to it.    |
| `main_code/train_resnetage.py`              | Single-fold ResNetAge training (fold 2); writes `baselines/resnetage_*`.                                                                                       |
| `main_code/inference_unhealthy.py`          | Runs the Horvath clock and the PNA checkpoints in `checkpoints/` on the disease cohort in `data/unhelathy-dataset/`.                                           |
| `main_code/inference_unhealthy_altumage.py` | Adds AltumAge predictions to the disease-cohort tables (requires TensorFlow).                                                                                  |
| `main_code/analyze_unhealthy_age_groups.py` | Summarises disease-cohort age acceleration per disease and age group.                                                                                          |
| `main_code/patch_mlp_cnnperm.py`            | Historical patch that added `--gnn mlp` and `--variant cnn_perm` to an earlier `train.py`; kept for record only. The shipped `train.py` already contains both. |


---



## Requirements

- Python 3.10 or newer
- PyTorch with CUDA
- PyTorch Geometric (provides `PNAConv`)
- numpy, pandas, scikit-learn
- TensorFlow, only for `inference_unhealthy_altumage.py`

One CUDA GPU is required for training. Training uses `batch_size 1`, so GPU
memory demand is modest; the dominant cost is the number of forward passes.

---



## Required Data

All paths are relative to the project root, which every script resolves from
the `GA_BASE` environment variable. The default is the parent directory of
`main_code/`, i.e. this folder, so nothing needs to be set when the commands
below are run from here. Set it only when running from elsewhere:

```bash
export GA_BASE=/path/to/official_code
```


| Path                                                            | Size              | Purpose                                                                                                             |
| --------------------------------------------------------------- | ----------------- | ------------------------------------------------------------------------------------------------------------------- |
| `data/all-organs4/all_organs/`                                  | 1.2 GB, 94 `.pkl` | Healthy methylation profiles. After the whole-blood filter this yields 3,707 samples from 37 studies.               |
| `data/cpgsite-info/GPL8490_HumanMethylation27_270596_v.1.2.csv` | 21 MB             | Illumina 27K manifest. Supplies the nine positional features and the 122 bp `TopGenomicSeq` window read by the CNN. |
| `data/multi_platform_cpgs.pkl`                                  | 260 KB            | The list of 20,318 CpG sites that defines the node set.                                                             |


Optional: `cache/file_order.json` pins the order in which the `.pkl` files are
concatenated, which determines fold membership. `freeze_order.py` writes it
from `data/reference_test_predictions.csv`; when the file is absent the
scripts fall back to directory order.

The reference-clock assets `baselines/coefficients.csv` (Horvath),
`baselines/AltumAge.h5` and `baselines/scaler.pkl` (AltumAge) are used only by
the inference scripts.

---



## Training the Proposed Model

The proposed model is `--variant cnn --gnn pna`, trained with
`--patience 10 --min-lr 1e-7 --tag _sched`.

### Step 1 — build the per-fold co-methylation graphs

The graph for each fold is derived from that fold's training partition only and
cached, so it is computed once and reused by all seeds.

```bash
python main_code/build_graph.py --folds 0 1 2 3 4 --thr-corr 0.70 --thr-dist 1e5
```

This writes `cache/graph_fold{0,1,2,3,4}_corr0.7_dist100000.pt`, each containing
the edge index, the 3-dimensional edge attributes, the PNA degree tensor, and
the ordered list of CpG identifiers. An index of all built graphs is appended to
`cache/graph_index.csv`.

Useful flags: `--force` rebuilds an existing cache, `--verify` recomputes the
graph with the unchunked reference implementation and reports the maximum
deviation, `--max-cpgs N` builds a small graph for a smoke test.

### Step 2 — train the model



#### Single run

```bash
python main_code/train.py --variant cnn --gnn pna --fold 2 --seed 0 --thr-corr 0.7 --thr-dist 1e5 --mlp-first 1024 --epochs 150 --lr 8e-5 --weight-decay 6e-4 --factor 0.4 --patience 10 --min-lr 1e-7 --batch-size 1 --tag _sched
```

The run directory is named from the arguments:

```
runs/cnn_fold<FOLD>_seed<SEED>_corr0.7_pna_sched/
```

Re-running the same command resumes an interrupted run from its last completed
epoch; a finished run exits immediately.

#### All 15 runs (5 folds x 3 seeds)

```bash
for f in 0 1 2 3 4; do for s in 0 1 2; do python main_code/train.py --variant cnn --gnn pna --fold $f --seed $s --thr-corr 0.7 --thr-dist 1e5 --mlp-first 1024 --epochs 150 --lr 8e-5 --weight-decay 6e-4 --factor 0.4 --patience 10 --min-lr 1e-7 --batch-size 1 --tag _sched; done; done
```



### Outputs

Each run writes the following into its own directory under `runs/`:


| File                   | Contents                                                                                                                                                           |
| ---------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `config.json`          | Every command-line argument as actually parsed. This is the authoritative record of what was run.                                                                  |
| `best_val.pt`          | Model weights at the best validation epoch, plus the epoch index, validation MAE, test MAE, and the config.                                                        |
| `ckpt.pt`              | Full training state: model, optimiser, scheduler, RNG state, best-so-far metrics. Used for resuming; removed when the run completes unless `--keep-ckpt` is given. |
| `metrics.csv`          | One row per epoch: `epoch, lr, secs, train_loss, train_mae, val_mae, val_mse, val_r2, test_mae, test_mse, test_r2`.                                                |
| `status.json`          | Progress summary, including `state`, `epoch`, `target_epochs`, `best_epoch`, `best_val_mae`, `test_mae_at_best_val`.                                               |
| `test_predictions.csv` | `true_age, predicted_age` for the 756 held-out test samples at the best epoch.                                                                                     |
| `val_predictions.csv`  | The same for the validation split.                                                                                                                                 |
| `test_age_groups.csv`  | Test metrics broken down by age group.                                                                                                                             |
| `log.txt`              | Full console log, appended across restarts.                                                                                                                        |




### Other configurations and baselines

The remaining rows of the results tables are produced by the same pipeline with
different arguments. Unless shown, `train.py` runs use its default schedule
(`--patience 4 --min-lr 1e-11`) and no `--tag`; all other arguments are as in
the single-run command above, looped over the same 5 folds x 3 seeds.


| Configuration                       | Command                                                                                          |
| ----------------------------------- | ------------------------------------------------------------------------------------------------ |
| Sequence-conditioned PNA (proposed) | `python main_code/train.py --variant cnn --gnn pna --patience 10 --min-lr 1e-7 --tag _sched ...` |
| Sequence-agnostic PNA               | `python main_code/train.py --variant none --gnn pna ...`                                         |
| Sequence-agnostic MLP               | `python main_code/train.py --variant none --gnn mlp ...`                                         |
| Sequence-conditioned MLP            | `python main_code/train.py --variant cnn --gnn nograph ...`                                      |
| Permuted-sequence PNA               | `python main_code/train.py --variant cnn_perm --gnn pna ...`                                     |
| Elastic Net                         | `python main_code/baselines_wave0cfg.py --model en`                                              |
| ResNetAge                           | `python main_code/baselines_dlorig.py --model resnetage`                                         |


`make_configs.py` writes ready-made run lists for these configurations to
`configs/*.tsv`; each line is one `train.py` argument set.

---



## Main Results

The evaluation uses 3,707 blood methylation profiles from 37 studies, with 756
samples reserved as a fixed held-out test set. Neural configurations are
evaluated over five folds and three random seeds per fold.

### Overall Predictive Performance


| Model                               | Evaluation         | MAE ↓             | MedAE ↓           | RMSE ↓            | R² ↑                |
| ----------------------------------- | ------------------ | ----------------- | ----------------- | ----------------- | ------------------- |
| **Sequence-conditioned PNA (ours)** | 15 runs            | **3.405 ± 0.210** | **2.122 ± 0.193** | **5.434 ± 0.244** | **0.9570 ± 0.0039** |
| Sequence-agnostic PNA               | 15 runs            | 3.550 ± 0.162     | 2.242 ± 0.169     | 5.584 ± 0.258     | 0.9546 ± 0.0042     |
| Sequence-agnostic MLP               | 15 runs            | 3.693 ± 0.139     | 2.326 ± 0.162     | 5.759 ± 0.225     | 0.9517 ± 0.0037     |
| Elastic Net                         | 5 folds            | 3.789 ± 0.034     | 2.771 ± 0.064     | 5.595 ± 0.060     | 0.9545 ± 0.0010     |
| ResNetAge                           | 15 runs            | 5.373 ± 0.354     | 3.726 ± 0.349     | 7.728 ± 0.425     | 0.9129 ± 0.0098     |
| AltumAge                            | External reference | 3.710             | 1.375             | 6.905             | 0.9307              |
| Horvath                             | External reference | 5.823             | 3.892             | 8.838             | 0.8865              |


Bold values indicate the best results among models trained on the study
partitions. AltumAge and Horvath are externally trained reference clocks and
are not directly matched comparisons.

### Correct Sequence–Site Correspondence


| Configuration                | Sequence assignment                   | Mean test MAE ↓ | Main finding                                                 |
| ---------------------------- | ------------------------------------- | --------------- | ------------------------------------------------------------ |
| **Sequence-conditioned PNA** | Correct CpG–sequence pairing          | **3.405**       | Best of the three configurations                             |
| Permuted-sequence PNA        | Sequences reassigned across CpG sites | 3.542           | Worse than the correctly paired model in all 15 matched runs |
| Sequence-agnostic PNA        | No sequence input                     | 3.550           | Matched graph control                                        |



| Attribution result                                     | Evidence                                                                                                            |
| ------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------- |
| Mean advantage of correct over permuted sequence       | 0.137 years                                                                                                         |
| Matched runs favouring correct sequence assignment     | 15/15                                                                                                               |
| Sequence-associated improvement removed by permutation | Approximately 95%                                                                                                   |
| Interpretation                                         | Most of the predictive value depends on the correct CpG–sequence correspondence rather than CpG-site identity alone |




### Factorial Ensemble Analysis

The following results average predictions across the three matched seeds within
each fold.


| Sequence conditioning | Graph propagation | Configuration                | Ensemble MAE ↓    |
| --------------------- | ----------------- | ---------------------------- | ----------------- |
| Yes                   | Yes               | **Sequence-conditioned PNA** | **3.128 ± 0.085** |
| No                    | Yes               | Sequence-agnostic PNA        | 3.455 ± 0.114     |
| Yes                   | No                | Sequence-conditioned MLP     | 3.391 ± 0.125     |
| No                    | No                | Sequence-agnostic MLP        | 3.637 ± 0.123     |



| Factorial finding                     | Evidence                                                                                                               |
| ------------------------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| Contribution of sequence conditioning | Removing sequence increases error in all five folds                                                                    |
| Contribution of graph propagation     | Removing graph propagation increases error in all five folds                                                           |
| Joint contribution                    | The complete model has the lowest ensemble MAE in every fold                                                           |
| Interpretation                        | Sequence context and graph propagation provide complementary predictive information under matched prediction averaging |




### Additional Findings


| Finding                | Result                                                                                                                                     |
| ---------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| Training variability   | The complete model has a run-to-run MAE standard deviation of 0.210 years, exceeding the individual component effects of 0.082–0.144 years |
| Evaluation implication | Single-run comparisons can obscure the contributions of sequence conditioning and graph propagation                                        |


