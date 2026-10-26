import gc
import sys

import torch

from flipaware import tracking
from flipaware.common import (
    TRAP_SETS,
    emb_dim,
    eval_dtype,
    load_cfg,
    load_st,
    log,
    mrl_dims,
    nevir_eval,
    passage_eval,
    prompts,
    read_json,
    read_jsonl,
    run_dir,
    set_seq,
    trap_eval,
    write_json,
)


def out_path(cfg, kind, m, v=None, s=None):
    res = cfg["paths"]["results"]
    return res / (f"base_{m}.json" if kind == "base" else f"run_{m}__{v}__s{s}.json")


def benches(cfg, m, kind):
    data = cfg["paths"]["data"]
    run = cfg["run"]
    have = {}
    for name, fname in TRAP_SETS.items():
        rows = read_jsonl(data / fname)
        if rows:
            have[name] = rows
    for name, fname in (("passage", "passage_test.jsonl"), ("nevir", "nevir_test.jsonl")):
        rows = read_jsonl(data / fname)
        if rows:
            have[name] = rows
    if kind == "run" or m in run["models"] or run.get("lb_nanobeir"):
        have["nanobeir"] = None
    return have


def nanobeir(cfg, mcfg, model):
    from sentence_transformers.evaluation import NanoBEIREvaluator

    q, d, _ = prompts(mcfg)
    kw = {"dataset_names": cfg["run"]["nanobeir"], "batch_size": 32, "show_progress_bar": False}
    if q:
        kw["query_prompts"] = q
    if d:
        kw["corpus_prompts"] = d
    ev = NanoBEIREvaluator(**kw)
    scores = ev(model)
    detail = {k: float(v) for k, v in scores.items() if k.endswith("ndcg@10")}
    return {"mean": float(scores[ev.primary_metric]), "detail": detail}


def compute(cfg, mcfg, model, name, rows, full_dims=True):
    q, d, s = prompts(mcfg)
    if name == "trap":
        dims = mrl_dims(cfg, emb_dim(model)) if full_dims else None
        return trap_eval(model, rows, s, dims, quant=True)
    if name.startswith("trap_"):
        return trap_eval(model, rows, s)
    if name == "passage":
        return passage_eval(model, rows, q, d)
    if name == "nevir":
        return nevir_eval(model, rows, q, d)
    if name == "nanobeir":
        return nanobeir(cfg, mcfg, model)
    raise ValueError(name)


def evaluate_model(cfg, mcfg, model, sets, existing=None, full_dims=True):
    out = dict(existing or {})
    for name, rows in sets.items():
        if name in out:
            continue
        log(f"  {name}")
        out[name] = compute(cfg, mcfg, model, name, rows, full_dims)
    return out


def headline(r):
    parts = []
    if r.get("trap"):
        parts.append(f"trap {r['trap']['dims'][str(r['trap']['full'])]['accuracy']:.4f}")
    for k in ("trap_xgen", "trap_human"):
        if r.get(k):
            parts.append(f"{k} {r[k]['dims'][str(r[k]['full'])]['accuracy']:.4f}")
    if r.get("passage"):
        parts.append(f"passage {r['passage']['pair_acc']:.4f}")
    if r.get("nevir"):
        parts.append(f"nevir {r['nevir']['pair_acc']:.4f}")
    if r.get("nanobeir"):
        parts.append(f"nanobeir {r['nanobeir']['mean']:.4f}")
    return "  ".join(parts)


def run_eval(cfg, kind, m, v=None, s=None):
    out = out_path(cfg, kind, m, v, s)
    existing = read_json(out) if out.exists() else {}
    sets = benches(cfg, m, kind)
    missing = {k: x for k, x in sets.items() if k not in existing}
    if not missing:
        return out
    mcfg = cfg["models"][m]
    path = None if kind == "base" else run_dir(cfg, m, v, s) / "model"
    log(f"eval {kind} {m} {v or ''} {'' if s is None else 's' + str(s)}: {', '.join(missing)}")
    model = load_st(mcfg, path=path, dtype=eval_dtype())
    set_seq(model, cfg["run"]["eval_seq_len"])
    r = evaluate_model(cfg, mcfg, model, missing, existing)
    r.update(kind=kind, model=m, variant=v, seed=s, base_id=mcfg["id"])
    if kind == "run":
        r["meta"] = read_json(run_dir(cfg, m, v, s) / "done.json")
    write_json(out, r)
    log(headline(r))
    tracking.log_eval(cfg, kind, m, v, s, r)
    model = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out


def main():
    cfg = load_cfg()
    a = sys.argv[1:]
    if a[0] == "base":
        run_eval(cfg, "base", a[1])
    else:
        run_eval(cfg, "run", a[1], a[2], int(a[3]))


if __name__ == "__main__":
    main()
