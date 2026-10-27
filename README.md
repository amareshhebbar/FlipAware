# FlipAware

Embedding models fine tuned to tell apart sentences that look alike but mean different things: negation, numbers, swapped roles, time order, quantifiers, hedges, swapped entities, and scope. Every release is a drop in replacement for its base model, checked on four independent test sets, and reported next to general retrieval.

Google Colab [LINK](https://colab.research.google.com/drive/1nv-rDvUbCsYr60k6zJ_0cRRgV1ZXKeVo?usp=sharing) or This [LINK](https://github.com/amareshhebbar/FlipAware/blob/main/notebooks/flipaware_pipeline.ipynb)

## What the pipeline produces

| Output | Where |
|---|---|
| Training data with every rejected row and the reason | HF dataset `FlipAware_Triples` (configs: default, candidates, seeds) |
| Cross generator test set written by a different model family | same dataset, config `xgen_test` |
| Passage retrieval traps: one sentence flipped inside a real passage | same dataset, config `passage_test` |
| Human written test set | same dataset, config `human_test` |
| One fine tuned model per base model, with model card, charts, results, TensorBoard logs | HF model repos `FlipAware_<base>` |
| Leaderboard of 25 open embedders plus the tuned models | HF Space `FlipAware_Leaderboard` |
| Write up draft with every number filled in | `results/REPORT.md` |
| Every stage log, training step, and eval metric | HF dataset `FlipAware_Runs`, Spaces `FlipAware_TensorBoard` and `FlipAware_MLflow`, folder `reports/<profile>/` |

## Stages

| # | Stage | What it does |
|---|---|---|
| 0 | s00_gpu_check | GPU, precision, versions, HuggingFace login, gated model access |
| 1 | s01_data | Seed sentences with their passages, human anchor pool, retrieval pairs, guard pairs, NevIR |
| 2 | s02_generate | Qwen3 writes a paraphrase and a trap per seed and trap type |
| 3 | s03_filter | NLI filter and word overlap check, with a reason for every rejection |
| 4 | s04_splits | Split by source sentence, hand check sheet, passage traps |
| 5 | s02_generate --set xgen | A different model family rewrites the held out test sentences |
| 6 | s03_filter --set xgen | Same filter for the cross generator set |
| 7 | s07_dataset | Uploads the dataset and creates the collection |
| 8 | s08_models | For each model: train, upload, test, print GOOD or BAD, then the next model |
| 9 | s09_human | Pulls approved human triples from the writer app |
| 10 | s10_analysis | Leaderboard models, weight blending, ONNX, MTEB, and a refresh of every eval |
| 11 | s11_finalize | Charts, cards, leaderboard Space, write up |
| 12 | s12_logs | Exports every log and metric to a HF dataset, live dashboards, and `reports/` for GitHub |

Every stage skips finished work, so a rerun continues where it stopped. Evaluation is incremental: when a new test set appears, only that set is computed.

## Experiments built in

| Experiment | Setting |
|---|---|
| Full vs LoRA, guard vs no guard | `ablation_models` |
| Per model hard negative mining | variant `mined_guard` |
| Data scaling curve | `scaling_models`, `scaling_sizes` |
| Blending tuned and base weights | `blend_models`, `blend_alphas` |
| Shorter, int8, and binary vectors | always on |
| Where embedders fail: sentence length and edit size | always on |
| Paired bootstrap confidence intervals | `bootstrap` |
| 3 seeds per headline model | `seeds` |

## Verdict

A model is GOOD when trap accuracy rises by at least `min_trap_gain`, NanoBEIR nDCG@10 drops by at most `max_ndcg_drop`, and the cross generator gain is at least `min_xgen_gain`. On RunPod, GOOD models go public and into the collection; BAD models stay private.

## Profiles

| Profile | Where | What runs |
|---|---|---|
| small | Colab, Kaggle, local | One tiny model, small data, every feature switched on to prove it works |
| full | RunPod (A40, 48GB) | 12 models, 20k triples, 3 seeds, 25 model leaderboard |
| xl | RunPod (A100 or H100, 80GB) | Qwen3 Embedding 4B and 8B, gte Qwen2 1.5B, 80k triples, Qwen3 32B generator, LoRA headline |

Override with `FLIP_PROFILE`. Pick models with `FLIP_MODELS=bge_small,embeddinggemma_300m`.

## Notebook

`notebooks/flipaware_pipeline.ipynb` runs on Colab and on RunPod Jupyter. It detects the platform, picks the profile, and runs one stage per cell with checks after each. On RunPod it can also start the whole run in the background.

Open on Colab: https://colab.research.google.com/github/amareshhebbar/flipaware/blob/main/notebooks/flipaware_pipeline.ipynb

## Reruns

Stages rerun by themselves when their inputs change: new generator settings move old generations aside, new generations refilter, new filtered data rebuilds the splits and clears later outputs. To redo work on purpose:

```bash
python -m flipaware.reset 2 --yes
```

## Colab

```python
import os
from google.colab import drive, userdata
from huggingface_hub import login
drive.mount("/content/drive")
os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
login(token=os.environ["HF_TOKEN"])
```

```bash
!git clone https://github.com/amareshhebbar/flipaware.git
%cd flipaware
!bash setup.sh
!bash run_all.sh
```

## RunPod

```bash
cd /workspace
git clone https://github.com/amareshhebbar/flipaware.git
cd flipaware
export HF_TOKEN=hf_xxx
FLIP_ONNX=1 FLIP_MTEB=1 bash setup.sh
python -c "import os; from huggingface_hub import login; login(token=os.environ['HF_TOKEN'])"
mkdir -p /workspace/flipaware_work
FLIP_STOP_POD=1 FLIP_MTEB=1 nohup bash run_all.sh > /workspace/flipaware_work/run.log 2>&1 &
tail -f /workspace/flipaware_work/run.log
```

## Human test set

```bash
python -m flipaware.human deploy
```

This creates a public writer Space and a private dataset for the submissions. Share the Space link. People write triples on sentences no generator has seen and check each other's work. Run `python -m flipaware.s09_human` or rerun the pipeline from stage 9 to pull approved triples into the test set:

```bash
FLIP_FROM=9 bash run_all.sh
```

Use a fine grained token limited to the submissions dataset for the Space: `WRITER_TOKEN=hf_xxx python -m flipaware.human deploy`.

## Options

| Variable | Effect |
|---|---|
| `FLIP_PROFILE` | `small`, `full`, or `xl` |
| `FLIP_MODELS` | Models to train, comma separated keys from `config.yaml` |
| `FLIP_LEADERBOARD` | Leaderboard models, comma separated |
| `FLIP_FROM` | Start from a stage number |
| `FLIP_PUBLISH=0` | Skip all HuggingFace uploads |
| `FLIP_ONNX=1` | Install optimum in setup and export ONNX with int8 weights |
| `FLIP_MTEB=1` | Install mteb in setup and run MTEB retrieval tasks |
| `FLIP_STOP_POD=1` | Stop the RunPod pod when the run ends |
| `FLIP_WORK` | Custom output folder |

## Tracking and logs

| Tracker | Default | Stored at | Shown on |
|---|---|---|---|
| TensorBoard | on | `<work>/runs/*/tb`, `<work>/tracking/tb_eval` | Each model repo (Training metrics tab), Space `FlipAware_TensorBoard` |
| MLflow | on | `<work>/tracking/mlflow.db` (one SQLite file) | Space `FlipAware_MLflow` |
| Aim | opt in | `<work>/tracking/aim` | Space `FlipAware_Aim` |
| ClearML | opt in | Offline sessions in `<work>/tracking/clearml_cache`, or your ClearML server when `CLEARML_API_ACCESS_KEY` is set | ClearML web app |

Pick trackers with `FLIP_TRACKERS`, for setup and run:

```bash
FLIP_TRACKERS=tensorboard,mlflow,aim,clearml bash setup.sh
FLIP_TRACKERS=tensorboard,mlflow,aim,clearml bash run_all.sh
```

A tracker that is not installed is skipped with a message; the run continues.

Every stage's console output is saved to `<work>/logs/`. Every training step and evaluation result is also appended to `<work>/tracking/metrics.jsonl`. Stage 12 bundles all of it, removes tokens, and publishes:

* HF dataset `FlipAware_Runs`: logs, metrics, TensorBoard events, MLflow database, Aim repo, ClearML sessions, raw results
* Spaces `FlipAware_TensorBoard` and `FlipAware_MLflow` (and `FlipAware_Aim` when used): live dashboards over those files
* `reports/<profile>/` in this repo: the text logs, metrics, results, and charts, ready to commit

| Variable | Effect |
|---|---|
| `FLIP_TRACKERS` | Trackers to use, default `tensorboard,mlflow` |
| `FLIP_MLFLOW_URI` | Send MLflow to your own server instead of the local file |
| `FLIP_DASHBOARDS=0` | Skip the dashboard Spaces |
| `FLIP_GITHUB_REPORTS=0` | Skip the `reports/` copy |

Open locally:

```bash
tensorboard --logdir <work>/runs
mlflow ui --backend-store-uri sqlite:///<work>/tracking/mlflow.db
aim up --repo <work>/tracking/aim
```
