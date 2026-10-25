import random
import re

import numpy as np
import torch
from sentence_transformers import CrossEncoder

from flipaware.common import load_cfg, log, read_jsonl, write_json, write_jsonl

WORD = re.compile(r"[a-z0-9']+")


def words(s):
    return set(WORD.findall(s.lower()))


def jaccard(a, b):
    x, y = words(a), words(b)
    return len(x & y) / max(1, len(x | y))


def entail_index(ce):
    conf = getattr(ce, "config", None) or getattr(getattr(ce, "model", None), "config", None)
    labels = {}
    if conf is not None and getattr(conf, "id2label", None):
        labels = {str(v).lower(): int(k) for k, v in conf.id2label.items()}
    if "entailment" not in labels:
        labels = {"contradiction": 0, "entailment": 1, "neutral": 2}
    return labels["entailment"]


def main():
    cfg = load_cfg()
    run, flt = cfg["run"], cfg["filter"]
    data, res = cfg["paths"]["data"], cfg["paths"]["results"]
    out = data / "filtered.jsonl"
    if out.exists():
        log("skip filter")
        return
    raw = read_jsonl(data / "raw.jsonl")
    stats = {}
    for r in raw:
        st = stats.setdefault(r["type"], {"generated": 0, "skipped": 0, "unparsed": 0, "candidates": 0, "kept": 0})
        st["generated"] += 1
        if r.get("skip"):
            st["skipped"] += 1
        elif r.get("error"):
            st["unparsed"] += 1
    clean = []
    for r in raw:
        if "paraphrase" not in r:
            continue
        a, p, t = r["anchor"], r["paraphrase"], r["trap"]
        if len({a.lower(), p.lower(), t.lower()}) < 3:
            continue
        clean.append(r)
        stats[r["type"]]["candidates"] += 1
    if not clean:
        raise SystemExit("no usable generations")
    ce = CrossEncoder(run["nli_model"], device="cuda" if torch.cuda.is_available() else "cpu")
    ent = entail_index(ce)
    pairs = []
    for r in clean:
        pairs += [(r["anchor"], r["paraphrase"]), (r["paraphrase"], r["anchor"]), (r["anchor"], r["trap"]), (r["trap"], r["anchor"])]
    probs = ce.predict(pairs, batch_size=64, apply_softmax=True, convert_to_numpy=True, show_progress_bar=False)
    probs = np.asarray(probs).reshape(len(clean), 4, -1)[:, :, ent]
    keep = []
    for r, pr in zip(clean, probs):
        ap, pa, at, ta = (float(x) for x in pr)
        ov = jaccard(r["anchor"], r["trap"])
        para_ok = min(ap, pa) >= flt["entail_min"]
        trap_ok = min(at, ta) < flt["equiv_max"] and ov >= flt["trap_overlap_min"]
        if para_ok and trap_ok:
            row = {k: r[k] for k in ("id", "type", "seed_id", "anchor", "paraphrase", "trap")}
            row.update(nli_para=round(min(ap, pa), 4), nli_trap=round(min(at, ta), 4), overlap=round(ov, 3))
            keep.append(row)
    cap = run.get("target_per_type")
    if cap:
        rng = random.Random(5)
        by = {}
        for r in keep:
            by.setdefault(r["type"], []).append(r)
        keep = []
        for ty, rows in by.items():
            rng.shuffle(rows)
            keep += rows[:cap]
    for r in keep:
        stats[r["type"]]["kept"] += 1
    write_jsonl(out, keep)
    write_json(res / "filter_stats.json", stats)
    for ty, st in stats.items():
        log(ty, st)
    log(f"kept {len(keep)} triples")


if __name__ == "__main__":
    main()
