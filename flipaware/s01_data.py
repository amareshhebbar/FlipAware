import random
import re

from datasets import load_dataset

from flipaware.common import load_cfg, log, write_jsonl

NUM = re.compile(
    r"\d|\b(one|two|three|four|five|six|seven|eight|nine|ten|hundred|thousand|million|billion|dozen|half|twice)\b",
    re.I,
)
SPLIT = re.compile(r"(?<=[.!?])\s+")


def good(s):
    w = s.split()
    if not 8 <= len(w) <= 30:
        return False
    if not s[0].isupper() or s[-1] not in ".!?":
        return False
    if "http" in s or "www." in s:
        return False
    return sum(c.isascii() for c in s) >= 0.98 * len(s)


def collect(stream, field, limit, seen):
    out = []
    if limit <= 0:
        return out
    for row in stream:
        for s in SPLIT.split((row.get(field) or "").strip()):
            s = " ".join(s.split())
            k = s.lower()
            if s and good(s) and k not in seen:
                seen.add(k)
                out.append(s)
                if len(out) >= limit:
                    return out
    return out


def build_seeds(run, path):
    seen = set()
    pool = run["seed_pool"]
    a = collect(load_dataset("sentence-transformers/gooaq", split="train", streaming=True), "answer", pool // 2, seen)
    b = collect(
        load_dataset("sentence-transformers/all-nli", "pair", split="train", streaming=True), "anchor", pool - len(a), seen
    )
    sents = a + b
    random.Random(13).shuffle(sents)
    rows = [{"seed_id": i, "text": s, "has_number": bool(NUM.search(s))} for i, s in enumerate(sents)]
    write_jsonl(path, rows)
    log(f"seeds: {len(rows)} ({sum(r['has_number'] for r in rows)} with numbers)")


def build_retrieval(run, path):
    nq = load_dataset("sentence-transformers/natural-questions", split="train").shuffle(seed=13)
    n = min(run["retrieval_pairs"], len(nq))
    sub = nq.select(range(n))
    rows = [{"anchor": q, "positive": a[:2000]} for q, a in zip(list(sub["query"]), list(sub["answer"]))]
    write_jsonl(path, rows)
    log(f"retrieval pairs: {len(rows)}")


def build_guard(run, path):
    rng = random.Random(17)
    n = run["guard_pairs"]
    sts = load_dataset("sentence-transformers/stsb", split="train")
    rows = [{"s1": a, "s2": b} for a, b in zip(list(sts["sentence1"]), list(sts["sentence2"]))]
    rng.shuffle(rows)
    rows = rows[: int(n * 0.7)]
    need = n - len(rows)
    if need > 0:
        nq = load_dataset("sentence-transformers/natural-questions", split="train").shuffle(seed=29)
        sub = nq.select(range(min(len(nq), need * 2)))
        qs = list(sub["query"])[:need]
        ans = list(sub["answer"])[need : 2 * need]
        rows += [{"s1": q, "s2": a[:500]} for q, a in zip(qs, ans)]
    rng.shuffle(rows)
    write_jsonl(path, rows)
    log(f"guard pairs: {len(rows)}")


def main():
    cfg = load_cfg()
    run = cfg["run"]
    data = cfg["paths"]["data"]
    jobs = [
        (data / "seeds.jsonl", build_seeds),
        (data / "retrieval.jsonl", build_retrieval),
        (data / "guard_src.jsonl", build_guard),
    ]
    for path, fn in jobs:
        if path.exists():
            log(f"skip {path.name}")
            continue
        fn(run, path)


if __name__ == "__main__":
    main()
