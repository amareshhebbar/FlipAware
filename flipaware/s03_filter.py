import argparse
import json
import random
import sys

import numpy as np
import torch

from flipaware.common import edit_check, file_sig, jaccard, load_cfg, log, read_jsonl, stamp_ok, text_sig, write_json, write_jsonl, write_stamp

KEYS = ("generated", "skipped", "unparsed", "duplicate", "trap_low_overlap", "candidates", "paraphrase_not_entailed", "trap_entailed", "over_cap", "kept")
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
            "attempts": r.get("attempts"),
            "nli_para": None,
            "nli_trap": None,
            "overlap": None,
            "kept": False,
            "reason": None,
        }
        cands[r["id"]] = rec
        if r.get("skip"):
            rec["reason"] = "skipped"
            st["skipped"] += 1
            continue
        if "paraphrase" not in r:
            rec["reason"] = "unparsed"
            st["unparsed"] += 1
            continue
        rec["overlap"] = round(jaccard(r["anchor"], r["trap"]), 3)
        bad = edit_check(r["anchor"], r["paraphrase"], r["trap"], flt)
        if bad:
            rec["reason"] = bad
            st[bad] += 1
            continue
        st["candidates"] += 1
        clean.append(r)
    keep = []
    if clean:
        for r, pr in zip(clean, score(cfg["run"], clean)):
            ap, pa, at, ta = (float(x) for x in pr)
            rec = cands[r["id"]]
            rec["nli_para"] = round(min(ap, pa), 4)
            rec["nli_trap"] = round(min(at, ta), 4)
            if rec["nli_para"] < flt["entail_min"]:
                rec["reason"] = "paraphrase_not_entailed"
            elif rec["nli_trap"] >= flt["equiv_max"]:
                rec["reason"] = "trap_entailed"
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


def report(stats, cands, which):
    print(f"\nfilter results ({which})")
    print(f"{'type':<12} {'gen':>5} {'skip':>5} {'unpar':>5} {'dup':>5} {'overl':>5} {'para':>5} {'trap':>5} {'kept':>5} {'rate':>6}")
    tot = {k: 0 for k in KEYS}
    for ty, st in stats.items():
        for k in KEYS:
            tot[k] += st[k]
        rate = st["kept"] / max(1, st["generated"]) * 100
        print(f"{ty:<12} {st['generated']:>5} {st['skipped']:>5} {st['unparsed']:>5} {st['duplicate']:>5} {st['trap_low_overlap']:>5} {st['paraphrase_not_entailed']:>5} {st['trap_entailed']:>5} {st['kept']:>5} {rate:>5.1f}%")
    rate = tot["kept"] / max(1, tot["generated"])
    print(f"{'all':<12} {tot['generated']:>5} {tot['skipped']:>5} {tot['unparsed']:>5} {tot['duplicate']:>5} {tot['trap_low_overlap']:>5} {tot['paraphrase_not_entailed']:>5} {tot['trap_entailed']:>5} {tot['kept']:>5} {rate * 100:>5.1f}%\n", flush=True)
    rej = [c for c in cands if not c["kept"]]
    for reason in sorted({c["reason"] for c in rej}):
        ex = next(c for c in rej if c["reason"] == reason)
        print(f"example rejected as {reason}: " + json.dumps({k: ex.get(k) for k in ("type", "anchor", "paraphrase", "trap")}, ensure_ascii=False)[:500])
    return rate


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
    sig = text_sig(file_sig(data / src), cfg["filter"], cfg["run"]["nli_model"], cap)
    stamp = f"filter_{which}"
    if file_sig(data / out) and stamp_ok(cfg, stamp, sig):
        log(f"skip filter ({which}), inputs unchanged")
        return
    raw = read_jsonl(data / src)
    if not raw:
        if which == "xgen":
            log("no cross generator generations, skipping")
            return
        raise SystemExit("raw.jsonl is empty, run stage 2 first")
    keep, cands, stats = run_filter(cfg, raw, cap)
    cols = OUT_COLS if which == "main" else ("id", "type", "seed_id", "anchor", "paraphrase", "trap")
    write_jsonl(data / out, [{k: r[k] for k in cols} for r in keep])
    write_jsonl(data / cand, cands)
    write_json(res / stat, stats)
    rate = report(stats, cands, which)
    if which == "main" and len(keep) < 20:
        log(f"only {len(keep)} triples kept, too few to split. Read the rejected examples above, then rerun stages 2 and 3")
        sys.exit(1)
    if rate < cfg["filter"]["min_keep_rate"]:
        log(f"warning: keep rate {rate * 100:.1f}% is low")
    write_stamp(cfg, stamp, sig)
    log(f"kept {len(keep)} of {len(raw)} generations ({which})")


if __name__ == "__main__":
    main()
