import json
import os
import random
import time
from pathlib import Path

import yaml

PKG = Path(__file__).resolve().parent
REPO = PKG.parent


def log(*parts):
    print(time.strftime("%H:%M:%S"), *parts, flush=True)


def detect_platform():
    if os.environ.get("RUNPOD_POD_ID"):
        return "runpod"
    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE"):
        return "kaggle"
    if os.environ.get("COLAB_RELEASE_TAG") or Path("/content/sample_data").is_dir():
        return "colab"
    return "local"


def detect_profile(platform):
    return os.environ.get("FLIP_PROFILE") or ("full" if platform == "runpod" else "small")


def detect_work(platform, profile):
    if os.environ.get("FLIP_WORK"):
        base = Path(os.environ["FLIP_WORK"])
    elif platform == "runpod":
        base = Path("/workspace/flipaware_work")
    elif platform == "colab":
        drive = Path("/content/drive/MyDrive")
        base = drive / "flipaware_work" if drive.is_dir() else Path("/content/flipaware_work")
    elif platform == "kaggle":
        base = Path("/kaggle/working/flipaware_work")
    else:
        base = REPO / "work"
    return base / profile


def load_cfg():
    cfg = yaml.safe_load((PKG / "config.yaml").read_text())
    platform = detect_platform()
    profile = detect_profile(platform)
    if profile not in cfg["profiles"]:
        raise SystemExit(f"unknown profile {profile}")
    run = dict(cfg["profiles"][profile])
    if os.environ.get("FLIP_MODELS"):
        keys = [k.strip() for k in os.environ["FLIP_MODELS"].split(",") if k.strip()]
        bad = [k for k in keys if k not in cfg["models"]]
        if bad:
            raise SystemExit(f"unknown models in FLIP_MODELS: {bad}; choose from {list(cfg['models'])}")
        run["models"] = keys
        run["ablation_models"] = [k for k in (run.get("ablation_models") or []) if k in keys]
    work = detect_work(platform, profile)
    paths = {"work": work}
    for name in ("data", "runs", "results", "assets", "publish"):
        paths[name] = work / name
        paths[name].mkdir(parents=True, exist_ok=True)
    cfg.update(platform=platform, profile=profile, run=run, paths=paths)
    return cfg


def hf_token():
    if os.environ.get("HF_TOKEN"):
        return os.environ["HF_TOKEN"]
    try:
        from huggingface_hub import get_token

        return get_token()
    except Exception:
        return None


def run_module(mod, *args):
    import subprocess
    import sys

    return subprocess.run([sys.executable, "-m", mod, *[str(a) for a in args]], cwd=str(REPO)).returncode


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False))
    tmp.replace(path)


def read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except ValueError:
                continue
    return rows


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(path)


def append_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def gpu_info():
    import torch

    if not torch.cuda.is_available():
        return {"cuda": False, "bf16": False, "cc_major": 0}
    p = torch.cuda.get_device_properties(0)
    return {
        "cuda": True,
        "name": p.name,
        "vram_gb": round(p.total_memory / 1024**3, 1),
        "cc": f"{p.major}.{p.minor}",
        "cc_major": p.major,
        "bf16": p.major >= 8,
        "count": torch.cuda.device_count(),
    }


def train_precision(mcfg):
    info = gpu_info()
    if info["bf16"]:
        return "bf16"
    if not info["cuda"] or mcfg.get("no_fp16"):
        return "fp32"
    return "fp16"


def eval_dtype():
    import torch

    return torch.bfloat16 if gpu_info()["bf16"] else torch.float32


def prompts(mcfg):
    return mcfg.get("query_prompt") or "", mcfg.get("doc_prompt") or "", mcfg.get("sym_prompt") or ""


def mrl_dims(cfg, full):
    return sorted({d for d in cfg["train"]["matryoshka_dims"] if d < full} | {full}, reverse=True)


def set_seq(model, n):
    tok = getattr(model, "tokenizer", None)
    lim = getattr(tok, "model_max_length", n) or n
    if not isinstance(lim, int) or lim > 100000:
        lim = n
    model.max_seq_length = min(n, lim)


def load_st(mcfg, path=None, dtype=None):
    import torch
    from sentence_transformers import SentenceTransformer

    kw = {
        "trust_remote_code": bool(mcfg.get("trust_remote_code", False)),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }
    if dtype is not None:
        kw["model_kwargs"] = {"torch_dtype": dtype}
    if path is None and mcfg.get("config_kwargs"):
        kw["config_kwargs"] = dict(mcfg["config_kwargs"])
    return SentenceTransformer(str(path) if path else mcfg["id"], **kw)


def encode(model, texts, prompt, batch_size=64):
    kw = {"batch_size": batch_size, "convert_to_tensor": True, "show_progress_bar": False}
    if prompt:
        kw["prompt"] = prompt
    return model.encode(texts, **kw).float()


def trap_eval(model, rows, prompt, dims=None):
    import torch.nn.functional as F

    a = encode(model, [r["anchor"] for r in rows], prompt)
    p = encode(model, [r["paraphrase"] for r in rows], prompt)
    t = encode(model, [r["trap"] for r in rows], prompt)
    full = a.shape[1]
    out = {}
    for d in dims or [full]:
        aa, pp, tt = (F.normalize(x[:, :d], dim=-1) for x in (a, p, t))
        sp = (aa * pp).sum(-1)
        st = (aa * tt).sum(-1)
        ok = (sp > st).cpu().tolist()
        margins = (sp - st).cpu().tolist()
        rec = {"accuracy": sum(ok) / len(ok), "margin": sum(margins) / len(margins)}
        if d == full:
            by = {}
            for ty in sorted({r["type"] for r in rows}):
                vals = [o for o, r in zip(ok, rows) if r["type"] == ty]
                by[ty] = sum(vals) / len(vals)
            rec["by_type"] = by
            rec["correct"] = ok
            rec["margins"] = margins
        out[str(d)] = rec
    return {"full": full, "dims": out}


def seed_all(seed):
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_dir(cfg, mkey, vname, seed):
    return cfg["paths"]["runs"] / f"{mkey}__{vname}__s{seed}"


def runs_for(cfg, m):
    run = cfg["run"]
    head = cfg["headline_variant"]
    plan = [(head, s) for s in run["seeds"]]
    if m in (run.get("ablation_models") or []):
        plan += [(v, run["seeds"][0]) for v in cfg["variants"] if v != head]
    return plan


def plan_runs(cfg):
    return [(m, v, s) for m in cfg["run"]["models"] for v, s in runs_for(cfg, m)]


def dev_acc(cfg, m, v, s):
    p = run_dir(cfg, m, v, s) / "train_log.json"
    if not p.exists():
        return None
    vals = [e["eval_trapdev_accuracy"] for e in read_json(p) if "eval_trapdev_accuracy" in e]
    return vals[-1] if vals else None


def best_seed(cfg, m, seeds):
    head = cfg["headline_variant"]
    return max((dev_acc(cfg, m, head, s) or 0.0, s) for s in seeds)[1]


def repo_name(cfg, mkey):
    return f"{cfg['project']['name']}_{cfg['models'][mkey]['tail']}{cfg['run']['repo_suffix']}"


def dataset_name(cfg):
    return f"{cfg['project']['name']}_Triples{cfg['run']['repo_suffix']}"


if __name__ == "__main__":
    print(detect_platform())