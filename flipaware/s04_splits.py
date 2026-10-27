import csv
import math
import random

from flipaware.common import SPLIT, file_sig, load_cfg, log, read_jsonl, reset_from, stamp_ok, text_sig, write_jsonl, write_stamp

COLS = ("id", "type", "seed_id", "anchor", "paraphrase", "trap")


def build_splits(cfg):
    data = cfg["paths"]["data"]
    rows = read_jsonl(data / "filtered.jsonl")
    if not rows:
        raise SystemExit("filtered.jsonl is empty")
    seeds = sorted({r["seed_id"] for r in rows})
    random.Random(21).shuffle(seeds)
    a, b = int(0.8 * len(seeds)), int(0.9 * len(seeds))
    part = {s: "train" for s in seeds[:a]}
    part.update({s: "dev" for s in seeds[a:b]})
    part.update({s: "test" for s in seeds[b:]})
    for split in ("train", "dev", "test"):
        sub = [{k: r[k] for k in COLS} for r in rows if part[r["seed_id"]] == split]
        write_jsonl(data / f"{split}.jsonl", sub)
        log(f"{split}: {len(sub)}")


def build_handcheck(cfg):
    rows = read_jsonl(cfg["paths"]["data"] / "filtered.jsonl")
    n = cfg["run"]["handcheck"]
    types = sorted({r["type"] for r in rows})
    per = math.ceil(n / len(types))
    rng = random.Random(33)
    picked = []
    for ty in types:
        pool = [r for r in rows if r["type"] == ty]
        picked += rng.sample(pool, min(per, len(pool)))
    path = cfg["paths"]["results"] / "handcheck.csv"
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "type", "anchor", "paraphrase", "trap", "paraphrase_ok", "trap_ok"])
        for r in picked[:n]:
            w.writerow([r["id"], r["type"], r["anchor"], r["paraphrase"], r["trap"], "", ""])
    log(f"hand check sheet: {path}")


def build_passages(cfg):
    data = cfg["paths"]["data"]
    ctx = {s["seed_id"]: s.get("context") for s in read_jsonl(data / "seeds.jsonl")}
    items = []
    for r in read_jsonl(data / "test.jsonl"):
        c = ctx.get(r["seed_id"])
        if not c or r["anchor"] not in c or len(SPLIT.split(c)) < 2:
            continue
        items.append(
            {
                "id": r["id"],
                "type": r["type"],
                "query": r["paraphrase"],
                "original": c,
                "twin": c.replace(r["anchor"], r["trap"], 1),
                "anchor": r["anchor"],
                "trap": r["trap"],
            }
        )
    random.Random(41).shuffle(items)
    cap = cfg["run"].get("passage_max")
    if cap:
        items = items[:cap]
    write_jsonl(data / "passage_test.jsonl", items)
    log(f"passage test items: {len(items)}")


def main():
    cfg = load_cfg()
    data, res = cfg["paths"]["data"], cfg["paths"]["results"]
    if not file_sig(data / "filtered.jsonl") or not read_jsonl(data / "filtered.jsonl"):
        raise SystemExit("filtered.jsonl is empty. Stage 3 kept nothing: read its rejected examples, then rerun stages 2 and 3")
    sig = text_sig(file_sig(data / "filtered.jsonl"), file_sig(data / "seeds.jsonl"))
    if (data / "test.jsonl").exists() and not stamp_ok(cfg, "splits", sig):
        log("filtered data changed: rebuilding splits and clearing every later output that used the old data")
        reset_from(cfg, 4)
    jobs = [
        (data / "test.jsonl", build_splits),
        (res / "handcheck.csv", build_handcheck),
        (data / "passage_test.jsonl", build_passages),
    ]
    for path, fn in jobs:
        if path.exists():
            log(f"skip {path.name}")
            continue
        fn(cfg)
    write_stamp(cfg, "splits", sig)


if __name__ == "__main__":
    main()
