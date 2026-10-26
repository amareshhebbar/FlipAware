import random
import re

from datasets import load_dataset

from flipaware.common import SPLIT, load_cfg, log, write_jsonl

NUM = re.compile(
    r"\d|\b(one|two|three|four|five|six|seven|eight|nine|ten|hundred|thousand|million|billion|dozen|half|twice)\b",
    re.I,
)


def good(s):
    w = s.split()
    if not 8 <= len(w) <= 30:
        return False
    if not s[0].isupper() or s[-1] not in ".!?":
        return False
    if "http" in s or "www." in s:
        return False
    return sum(c.isascii() for c in s) >= 0.98 * len(s)


def collect(stream, field, limit, seen, keep_context):
    out = []
    if limit <= 0:
        return out
    for row in stream:
        text = " ".join((row.get(field) or "").split())
        parts = SPLIT.split(text)
        ctx = text if keep_context and len(parts) >= 2 and len(text) <= 2000 else None
        for s in parts:
            k = s.lower()
            if s and good(s) and k not in seen:
                seen.add(k)
                out.append((s, ctx))
                if len(out) >= limit:
                    return out
                break
    return out


def gooaq():
    return load_dataset("sentence-transformers/gooaq", split="train", streaming=True)


def allnli():
    return load_dataset("sentence-transformers/all-nli", "pair", split="train", streaming=True)


def build_seeds(cfg):
    run, data = cfg["run"], cfg["paths"]["data"]
    seen = set()
    pool = run["seed_pool"]
    a = collect(gooaq(), "answer", pool // 2, seen, True)
    b = collect(allnli(), "anchor", pool - len(a), seen, False)
    rows = [(s, c, "gooaq") for s, c in a] + [(s, c, "all-nli") for s, c in b]
    random.Random(13).shuffle(rows)
    seeds = [
        {"seed_id": i, "text": s, "source": src, "has_number": bool(NUM.search(s)), "context": c}
        for i, (s, c, src) in enumerate(rows)
    ]
    write_jsonl(data / "seeds.jsonl", seeds)
    log(f"seeds: {len(seeds)} ({sum(r['has_number'] for r in seeds)} with numbers, {sum(bool(r['context']) for r in seeds)} with passages)")
    n = run.get("human_anchor_pool") or 0
    if n:
        h = collect(gooaq().skip(pool * 4), "answer", n // 2, seen, False)
        h += collect(allnli().skip(pool * 4), "anchor", n - len(h), seen, False)
        write_jsonl(data / "human_anchors.jsonl", [{"text": s} for s, _ in h])
        log(f"human anchor pool: {len(h)} sentences never used for generation")


def build_retrieval(cfg):
    run, data = cfg["run"], cfg["paths"]["data"]
    nq = load_dataset("sentence-transformers/natural-questions", split="train").shuffle(seed=13)
    n = min(run["retrieval_pairs"], len(nq))
    sub = nq.select(range(n))
    rows = [{"anchor": q, "positive": a[:2000]} for q, a in zip(list(sub["query"]), list(sub["answer"]))]
    write_jsonl(data / "retrieval.jsonl", rows)
    log(f"retrieval pairs: {len(rows)}")


def build_guard(cfg):
    run, data = cfg["run"], cfg["paths"]["data"]
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
    write_jsonl(data / "guard_src.jsonl", rows)
    log(f"guard pairs: {len(rows)}")


def build_nevir(cfg):
    run, data = cfg["run"], cfg["paths"]["data"]
    try:
        ds = load_dataset("orionweller/NevIR", split="test")
        rows = [{"q1": r["q1"], "q2": r["q2"], "doc1": r["doc1"], "doc2": r["doc2"]} for r in ds]
    except Exception as ex:
        log(f"NevIR unavailable, skipping: {ex!r}")
        rows = []
    cap = run.get("nevir_max")
    if cap:
        rows = rows[:cap]
    write_jsonl(data / "nevir_test.jsonl", rows)
    log(f"NevIR test pairs: {len(rows)}")


def main():
    cfg = load_cfg()
    data = cfg["paths"]["data"]
    jobs = [
        ("seeds.jsonl", build_seeds),
        ("retrieval.jsonl", build_retrieval),
        ("guard_src.jsonl", build_guard),
        ("nevir_test.jsonl", build_nevir),
    ]
    for name, fn in jobs:
        if (data / name).exists():
            log(f"skip {name}")
            continue
        fn(cfg)


if __name__ == "__main__":
    main()
