import csv
import math
import random

from flipaware.common import load_cfg, log, read_jsonl, write_jsonl

COLS = ("id", "type", "seed_id", "anchor", "paraphrase", "trap")


def main():
    cfg = load_cfg()
    data, res = cfg["paths"]["data"], cfg["paths"]["results"]
    if (data / "test.jsonl").exists():
        log("skip splits")
        return
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
    n = cfg["run"]["handcheck"]
    types = sorted({r["type"] for r in rows})
    per = math.ceil(n / len(types))
    rng = random.Random(33)
    picked = []
    for ty in types:
        pool = [r for r in rows if r["type"] == ty]
        picked += rng.sample(pool, min(per, len(pool)))
    path = res / "handcheck.csv"
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "type", "anchor", "paraphrase", "trap", "paraphrase_ok", "trap_ok"])
        for r in picked[:n]:
            w.writerow([r["id"], r["type"], r["anchor"], r["paraphrase"], r["trap"], "", ""])
    log(f"hand check sheet: {path}")


if __name__ == "__main__":
    main()
