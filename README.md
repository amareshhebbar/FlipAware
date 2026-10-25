# FlipAware

Embedding models fine tuned to tell apart sentences that look alike but mean different things: negation, numbers, swapped roles, time order, quantifiers, hedges, swapped entities, and scope. Every release is a drop in replacement for its base model and reports general retrieval next to the trap test.

## Pipeline

| Stage | What it does |
|---|---|
| s00_gpu_check | GPU, precision, package versions, HF token, gated model access |
| s01_data | Seed sentences, retrieval pairs, guard pairs |
| s02_generate | Qwen3 writes a paraphrase and a trap for each seed and trap type |
| s03_filter | NLI filter and word overlap check |
| s04_splits | Train, dev, test split by source sentence, plus a hand check sheet |
| s05_train | Every base model, variant, and seed |
| s06_eval | Trap test and NanoBEIR for base and tuned models |
| s07_report | Summary, charts, gallery |
| s08_publish | Dataset, models, model cards, collection on HuggingFace |

Every stage skips work that is already done, so a rerun continues where it stopped.

## Profiles

The profile is picked from where the code runs. Colab, Kaggle, and local machines use `small`. RunPod uses `full`. Override with `FLIP_PROFILE=small` or `FLIP_PROFILE=full`. Output goes to Google Drive on Colab when mounted, and to the network volume on RunPod.

## Colab

```python
from google.colab import drive, userdata
import os
drive.mount("/content/drive")
os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
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
bash setup.sh
FLIP_STOP_POD=1 nohup bash run_all.sh > /workspace/flipaware_work/run.log 2>&1 &
tail -f /workspace/flipaware_work/run.log
```

## Options

| Variable | Effect |
|---|---|
| `FLIP_PROFILE` | `small` or `full` |
| `FLIP_FROM` | Start from a stage number, for example `FLIP_FROM=5` |
| `FLIP_PUBLISH=0` | Skip publishing |
| `FLIP_STOP_POD=1` | Stop the RunPod pod when the run ends |
| `FLIP_WORK` | Custom output folder |

## Training curves

```bash
tensorboard --logdir <work>/runs
```

Curves are also uploaded to each model repo, where HuggingFace shows them under Training metrics.
