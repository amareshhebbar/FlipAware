import json
import os
import random
import re
import subprocess
import sys
import time
from pathlib import Path

import yaml

PKG = Path(__file__).resolve().parent
REPO = PKG.parent
SPLIT = re.compile(r"(?<=[.!?])\s+")
WORD = re.compile(r"[a-z0-9']+")
TRAP_SETS = {"trap": "test.jsonl", "trap_xgen": "xgen_test.jsonl", "trap_human": "human_test.jsonl"}
LISTS = ("ablation_models", "scaling_models", "blend_models", "onnx_models")


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


def env_list(name):
    v = os.environ.get(name)
    return [k.strip() for k in v.split(",") if k.strip()] if v else None


def load_cfg():
    cfg = yaml.safe_load((PKG / "config.yaml").read_text())
    platform = detect_platform()
    profile = detect_profile(platform)
    if profile not in cfg["profiles"]:
        raise SystemExit(f"unknown profile {profile}")
    run = dict(cfg["profiles"][profile])
    keys = env_list("FLIP_MODELS")
    if keys:
        run["models"] = keys
        for k in LISTS:
            run[k] = [m for m in (run.get(k) or []) if m in keys]
    lb = env_list("FLIP_LEADERBOARD")
    if lb is not None:
        run["leaderboard"] = lb
    bad = [m for m in run["models"] + (run.get("leaderboard") or []) if m not in cfg["models"]]
    if bad:
        raise SystemExit(f"unknown model keys {bad}; choose from {list(cfg['models'])}")
    head = run.get("headline_variant", cfg["headline_variant"])
    cfg["headline_variant"] = head
    variants = dict(cfg["variants"])
    for n in run.get("scaling_sizes") or []:
        variants[f"n{n}"] = {**variants[head], "limit": int(n), "scale": True}
    cfg["variants"] = variants
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


def words(s):
    return set(WORD.findall(s.lower()))


def jaccard(a, b):
    x, y = words(a), words(b)
    return len(x & y) / max(1, len(x | y))


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


def emb_dim(model):
    fn = getattr(model, "get_embedding_dimension", None) or model.get_sentence_embedding_dimension
    return fn()


def mrl_dims(cfg, full):
    return sorted({d for d in cfg["train"]["matryoshka_dims"] if d < full} | {full}, reverse=True)


def set_seq(model, n):
    tok = getattr(model, "tokenizer", None)
    lim = getattr(tok, "model_max_length", n) or n
    if not isinstance(lim, int) or lim > 100000:
        lim = n
    model.max_seq_length = min(n, lim)


def load_st(mcfg, path=None, dtype=None, device=None):
    import torch
    from sentence_transformers import SentenceTransformer

    kw = {
        "trust_remote_code": bool(mcfg.get("trust_remote_code", False)),
        "device": device or ("cuda" if torch.cuda.is_available() else "cpu"),
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


def quant_acc(a, p, t):
    import torch

    allv = torch.cat([a, p, t])
    lo = allv.min(0).values
    hi = allv.max(0).values
    step = (hi - lo).clamp_min(1e-8) / 255

    def q8(x):
        return (((x - lo) / step).round() - 128).clamp(-128, 127)

    qa, qp, qt = q8(a), q8(p), q8(t)
    int8 = ((qa * qp).sum(-1) > (qa * qt).sum(-1)).float().mean().item()
    ba, bp, bt = a > 0, p > 0, t > 0
    binary = ((ba == bp).sum(-1) > (ba == bt).sum(-1)).float().mean().item()
    return {"int8": int8, "binary": binary}


def trap_eval(model, rows, prompt, dims=None, quant=False):
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
    res = {"full": full, "n": len(rows), "dims": out}
    if quant:
        res["quant"] = quant_acc(a, p, t)
    return res


def passage_eval(model, items, qp, dp):
    import torch
    import torch.nn.functional as F

    n = len(items)
    q = F.normalize(encode(model, [x["query"] for x in items], qp, 32), dim=-1)
    docs = [x["original"] for x in items] + [x["twin"] for x in items]
    d = F.normalize(encode(model, docs, dp, 32), dim=-1)
    sims = q @ d.T
    idx = torch.arange(n, device=sims.device)
    pair = (sims[idx, idx] > sims[idx, idx + n]).float()
    rank = (sims > sims[idx, idx].unsqueeze(1)).sum(1).float()
    by = {}
    for ty in sorted({x["type"] for x in items}):
        sel = [i for i, x in enumerate(items) if x["type"] == ty]
        by[ty] = pair[sel].mean().item()
    return {
        "n": n,
        "pair_acc": pair.mean().item(),
        "recall1": (rank == 0).float().mean().item(),
        "mrr": (1.0 / (rank + 1)).mean().item(),
        "by_type": by,
    }


def nevir_eval(model, rows, qp, dp):
    import torch.nn.functional as F

    q1 = F.normalize(encode(model, [r["q1"] for r in rows], qp, 32), dim=-1)
    q2 = F.normalize(encode(model, [r["q2"] for r in rows], qp, 32), dim=-1)
    d1 = F.normalize(encode(model, [r["doc1"] for r in rows], dp, 32), dim=-1)
    d2 = F.normalize(encode(model, [r["doc2"] for r in rows], dp, 32), dim=-1)
    a = ((q1 * d1).sum(-1) > (q1 * d2).sum(-1)) & ((q2 * d2).sum(-1) > (q2 * d1).sum(-1))
    return {"n": len(rows), "pair_acc": a.float().mean().item()}


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
    s0 = run["seeds"][0]
    plan = [(head, s) for s in run["seeds"]]
    if m in (run.get("ablation_models") or []):
        plan += [(v, s0) for v, x in cfg["variants"].items() if v != head and not x.get("scale")]
    if m in (run.get("scaling_models") or []):
        plan += [(v, s0) for v, x in cfg["variants"].items() if x.get("scale")]
    return plan


def plan_runs(cfg):
    return [(m, v, s) for m in cfg["run"]["models"] for v, s in runs_for(cfg, m)]


def dev_acc(cfg, m, v, s):
    p = run_dir(cfg, m, v, s) / "train_log.json"
    if not p.exists():
        return None
    vals = [e["eval_trapdev_accuracy"] for e in read_json(p) if "eval_trapdev_accuracy" in e]
    return vals[-1] if vals else None


def done_seeds(cfg, m):
    head = cfg["headline_variant"]
    return [s for s in cfg["run"]["seeds"] if (run_dir(cfg, m, head, s) / "done.json").exists()]


def best_seed(cfg, m, seeds=None):
    head = cfg["headline_variant"]
    seeds = seeds if seeds is not None else done_seeds(cfg, m)
    if not seeds:
        return None
    return max((dev_acc(cfg, m, head, s) or 0.0, s) for s in seeds)[1]


def repo_name(cfg, mkey):
    return f"{cfg['project']['name']}_{cfg['models'][mkey]['tail']}{cfg['run']['repo_suffix']}"


def dataset_name(cfg):
    return f"{cfg['project']['name']}_Triples{cfg['run']['repo_suffix']}"


def space_name(cfg, kind):
    return f"{cfg['project']['name']}_{kind}{cfg['run']['repo_suffix']}"


def edit_check(anchor, para, trap, flt):
    a, p, t = (" ".join(x.lower().split()) for x in (anchor, para, trap))
    if len({a, p, t}) < 3:
        return "duplicate"
    aw, tw = set(WORD.findall(a)), set(WORD.findall(t))
    if not aw or not tw:
        return "duplicate"
    keep = len(aw & tw) / len(aw)
    added = len(tw - aw)
    if keep < flt["trap_keep_min"] or added > max(flt["trap_add_max"], int(0.3 * len(aw))):
        return "trap_low_overlap"
    return None


def file_sig(path):
    path = Path(path)
    if not path.exists():
        return None
    n = 0
    with path.open("rb") as f:
        for _ in f:
            n += 1
    return f"{path.stat().st_size}:{n}"


def text_sig(*parts):
    import hashlib

    return hashlib.sha1(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()[:16]


def stamp_path(cfg, name):
    return cfg["paths"]["data"] / f".stamp_{name}.json"


def stamp_ok(cfg, name, value):
    p = stamp_path(cfg, name)
    return p.exists() and read_json(p).get("sig") == value


def write_stamp(cfg, name, value):
    write_json(stamp_path(cfg, name), {"sig": value, "ts": time.time()})


RESET = {
    2: ["data/raw.jsonl", "data/.stamp_gen_main.json"],
    3: ["data/filtered.jsonl", "data/candidates.jsonl", "results/filter_stats.json", "data/.stamp_filter_main.json"],
    4: ["data/train.jsonl", "data/dev.jsonl", "data/test.jsonl", "data/passage_test.jsonl", "results/handcheck.csv", "data/.stamp_splits.json"],
    5: ["data/raw_xgen.jsonl", "data/.stamp_gen_xgen.json"],
    6: ["data/xgen_test.jsonl", "data/candidates_xgen.jsonl", "results/filter_stats_xgen.json", "data/.stamp_filter_xgen.json"],
    8: ["runs/*", "results/base_*.json", "results/run_*.json", "results/model_*.json", "results/blend_*.json", "results/onnx_*.json", "results/mteb_*.json", "results/collection.json"],
    11: ["results/summary.*", "results/leaderboard.*", "results/REPORT.md", "assets/*"],
}


def reset_from(cfg, n):
    import shutil

    work = cfg["paths"]["work"]
    gone = []
    for stage, pats in RESET.items():
        if stage < n:
            continue
        for pat in pats:
            for p in work.glob(pat):
                if p.is_dir():
                    shutil.rmtree(p)
                else:
                    p.unlink()
                gone.append(str(p.relative_to(work)))
    log(f"reset from stage {n}: removed {len(gone)} items")
    return gone


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "workdir":
        print(load_cfg()["paths"]["work"])
    else:
        print(detect_platform())
