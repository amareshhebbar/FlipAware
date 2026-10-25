# FlipAware

Embedding models fine tuned to tell apart sentences that look alike but mean different things: negation, numbers, swapped roles, time order, quantifiers, hedges, swapped entities, and scope. Every release is a drop in replacement for its base model and reports general retrieval next to the trap test.

### What we are building

Embedding models turn sentences into vectors so similar meanings sit close together. Today’s popular embedders are lazy: they mostly match shared words. So “the drug is safe” and “the drug is not safe” come out nearly identical, and search returns the wrong document.

FlipAware Embed fixes that. We take three existing small embedders and train them to notice the small edits that flip meaning. These are negation, numbers, who did what to whom, time order, some vs all, might vs will, swapped names, and scope. While they learn this, we make sure they stay just as good at normal search.

### Why it matters: 

It is a real, visible flaw with a clean before and after proof. It is also the fastest project in Weight Lab to finish end to end, so it sets up the pipeline (data, train, evaluate, publish) that every later project reuses.


## Pipeline

| Stage | What it does |
|---|---|
| s00_gpu_check | GPU, precision, package versions, HuggingFace login, gated model access |
| s01_data | Seed sentences, retrieval pairs, guard pairs |
| s02_generate | Qwen3 writes a paraphrase and a trap for each seed and trap type |
| s03_filter | NLI filter and word overlap check |
| s04_splits | Train, dev, test split by source sentence, plus a hand check sheet |
| s05_dataset | Uploads the triples dataset and creates the collection |
| s06_models | For each model: train, upload to HuggingFace, test, print GOOD or BAD, then the next model |
| s07_finalize | Cross model charts, refreshed model cards, final table |

Every step skips work that is already done, so a rerun continues where it stopped.

## Verdict

A model is GOOD when trap accuracy rises by at least `min_trap_gain` and NanoBEIR nDCG@10 drops by at most `max_ndcg_drop` (see `verdict` in `config.yaml`). On RunPod, GOOD models go public and into the collection; BAD models stay private.

## Profiles

Colab, Kaggle, and local machines use `small`: one tiny model on a small dataset to prove every stage works. RunPod uses `full`: all models, full dataset, 3 seeds. Override with `FLIP_PROFILE=small` or `FLIP_PROFILE=full`, and pick models with `FLIP_MODELS=bge_small,embeddinggemma_300m`.

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
bash setup.sh
python -c "import os; from huggingface_hub import login; login(token=os.environ['HF_TOKEN'])"
mkdir -p /workspace/flipaware_work
FLIP_STOP_POD=1 nohup bash run_all.sh > /workspace/flipaware_work/run.log 2>&1 &
tail -f /workspace/flipaware_work/run.log
```

## Options

| Variable | Effect |
|---|---|
| `FLIP_PROFILE` | `small` or `full` |
| `FLIP_MODELS` | Comma separated model keys from `config.yaml` |
| `FLIP_FROM` | Start from a stage number, for example `FLIP_FROM=6` |
| `FLIP_PUBLISH=0` | Skip all HuggingFace uploads |
| `FLIP_STOP_POD=1` | Stop the RunPod pod when the run ends |
| `FLIP_WORK` | Custom output folder |

## Training curves

Curves are logged to TensorBoard, which is free and runs locally. They are also uploaded to each model repo, where HuggingFace shows them under Training metrics, and drawn as `assets/curve.png`.

```bash
tensorboard --logdir <work>/runs
```