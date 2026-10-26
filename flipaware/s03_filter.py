import argparse
import random

import numpy as np
import torch

from flipaware.common import jaccard, load_cfg, log, read_jsonl, write_json, write_jsonl

KEYS = ("generated", "skipped", "unparsed", "duplicate", "candidates", "paraphrase_not_entailed", "trap_entailed", "trap_low_overlap", "over_cap", "kept")
OUT_COLS = ("id", "type", "seed_id", "anchor", "paraphrase", "trap", "nli_para", "nli_trap", "overlap")


def entail_index(ce):
    conf = getattr(ce, "config", None) or getattr(getattr(ce, "model", None), "config", None)
    labels = {}
    if conf is not None and getattr(conf, "id2label", None):
        labels = {str(v).lower(): int(k) for k, v in conf.id2label.items()}
    if "entailment" not in labels:
        labels = {"contradiction": 0, "entailment": 1, "neutral": 2}
    return labels["entailment"]


def score(run, clean):
    from sentence_transformers import CrossEncoder

    ce = CrossEncoder(run["nli_model"], device="cuda" if torch.cuda.is_available() else "cpu")
    ent = entail_index(ce)
    pairs = []
    for r in clean:
        pairs += [(r["anchor"], r["paraphrase"]), (r["paraphrase"], r["anchor"]), (r["anchor"], r["trap"]), (r["trap"], r["anchor"])]
    probs = ce.predict(pairs, batch_size=64, apply_softmax=True, convert_to_numpy=True, show_progress_bar=False)
    return np.asarray(probs).reshape(len(clean), 4, -1)[:, :, ent]


def run_filter(cfg, raw, cap):
    flt = cfg["filter"]
    stats, cands, clean = {}, {}, []
    for r in raw:
        st = stats.setdefault(r["type"], {k: 0 for k in KEYS})
        st["generated"] += 1
        rec = {
            "id": r["id"],
            "type": r["type"],
            "seed_id": r["seed_id"],
            "anchor": r["anchor"],
            "paraphrase": r.get("paraphrase"),
            "trap": r.get("trap"),
            "generator": r.get("generator"),
            "nli_para": None,
            "nli_trap": None,
            "overlap": None,
            "kept": False,
            "reason": None,
        }
        cands[r["id"]] = rec
        if r.get("skip"):
            st["skipped"] += 1
            rec["reason"] = "skipped"
        elif "paraphrase" not in r:
            st["unparsed"] += 1
            rec["reason"] = "unparsed"
        elif len({r["anchor"].lower(), r["paraphrase"].lower(), r["trap"].lower()}) < 3:
            st["duplicate"] += 1
            rec["reason"] = "duplicate"
        else:
            st["candidates"] += 1
            clean.append(r)
    if not clean:
        raise SystemExit("no usable generations")
    keep = []
    for r, pr in zip(clean, score(cfg["run"], clean)):
        ap, pa, at, ta = (float(x) for x in pr)
        rec = cands[r["id"]]
        rec["nli_para"] = round(min(ap, pa), 4)
        rec["nli_trap"] = round(min(at, ta), 4)
        rec["overlap"] = round(jaccard(r["anchor"], r["trap"]), 3)
        if rec["nli_para"] < flt["entail_min"]:
            rec["reason"] = "paraphrase_not_entailed"
        elif rec["nli_trap"] >= flt["equiv_max"]:
            rec["reason"] = "trap_entailed"
        elif rec["overlap"] < flt["trap_overlap_min"]:
            rec["reason"] = "trap_low_overlap"
        else:
            keep.append(rec)
            continue
        stats[r["type"]][rec["reason"]] += 1
    if cap:
        rng = random.Random(5)
        by = {}
        for r in keep:
            by.setdefault(r["type"], []).append(r)
        keep = []
        for ty, rows in by.items():
            rng.shuffle(rows)
            keep += rows[:cap]
            for r in rows[cap:]:
                r["reason"] = "over_cap"
                stats[ty]["over_cap"] += 1
    for r in keep:
        r["kept"] = True
        r["reason"] = "kept"
        stats[r["type"]]["kept"] += 1
    return keep, list(cands.values()), stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="main", choices=["main", "xgen"])
    which = ap.parse_args().set
    cfg = load_cfg()
    data, res = cfg["paths"]["data"], cfg["paths"]["results"]
    if which == "main":
        src, out, cand, stat, cap = "raw.jsonl", "filtered.jsonl", "candidates.jsonl", "filter_stats.json", cfg["run"].get("target_per_type")
    else:
        src, out, cand, stat, cap = "raw_xgen.jsonl", "xgen_test.jsonl", "candidates_xgen.jsonl", "filter_stats_xgen.json", None
    if (data / out).exists() and (data / cand).exists():
        log(f"skip filter ({which})")
        return
    raw = read_jsonl(data / src)
    if not raw:
        log(f"no generations for {which}, skipping")
        return
    keep, cands, stats = run_filter(cfg, raw, cap)
    cols = OUT_COLS if which == "main" else ("id", "type", "seed_id", "anchor", "paraphrase", "trap")
    write_jsonl(data / out, [{k: r[k] for k in cols} for r in keep])
    write_jsonl(data / cand, cands)
    write_json(res / stat, stats)
    for ty, st in stats.items():
        log(ty, st)
    log(f"kept {len(keep)} of {len(raw)} generations ({which})")


if __name__ == "__main__":
    main()
