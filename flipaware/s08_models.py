from flipaware.common import (
    best_seed,
    done_seeds,
    load_cfg,
    log,
    read_json,
    read_jsonl,
    repo_name,
    run_dir,
    run_module,
    runs_for,
    write_json,
)
from flipaware.publish import add_item, api, ensure_collection, publish_model
from flipaware.report import context, model_entry, table

LABELS = (
    ("trap", "trap test"),
    ("trap_xgen", "cross generator"),
    ("trap_human", "human set"),
    ("passage", "passage traps"),
    ("nevir", "NevIR"),
    ("ndcg", "NanoBEIR ndcg"),
)


def show(e):
    v = e["verdict"]
    b, t = e["base"], e["tuned"]
    lines = ["", "=" * 72, f"RESULT  {e['tail']}  ->  {v['label']}"]
    for k, label in LABELS:
        if b.get(k) is None or not t.get(k):
            continue
        d = t[k]["mean"] - b[k]
        lines.append(f"  {label:<16} base {b[k] * 100:5.1f}   tuned {t[k]['mean'] * 100:5.1f} ± {t[k]['std'] * 100:.1f}   ({d * 100:+.1f})")
    c = (e.get("confidence") or {}).get("trap")
    if c:
        lines.append(f"  95% interval on trap gain: {c['lo'] * 100:+.1f} to {c['hi'] * 100:+.1f}")
    for ty, bv in b["by_type"].items():
        tv = t["by_type"].get(ty)
        if tv:
            lines.append(f"    {ty:<12} {bv * 100:5.1f} -> {tv['mean'] * 100:5.1f}")
    for r in v["reasons"]:
        lines.append(f"  reason: {r}")
    lines.append("=" * 72)
    print("\n".join(lines), flush=True)


def main():
    cfg = load_cfg()
    run = cfg["run"]
    res = cfg["paths"]["results"]
    head = cfg["headline_variant"]
    hf = api()
    slug = ensure_collection(cfg, hf) if hf else None
    ctx = context(cfg)
    test = read_jsonl(cfg["paths"]["data"] / "test.jsonl")
    vcfg = run["verdict"]
    models = run["models"]
    for i, m in enumerate(models, 1):
        state_path = res / f"model_{m}.json"
        if state_path.exists() and read_json(state_path).get("status") == "done":
            log(f"[{i}/{len(models)}] {m} already done")
            continue
        log(f"[{i}/{len(models)}] {m}: training")
        fails = []
        for v, s in runs_for(cfg, m):
            if (run_dir(cfg, m, v, s) / "done.json").exists():
                continue
            if run_module("flipaware.train", m, v, s) != 0:
                fails.append(f"{v} s{s}")
                log(f"train failed: {m} {v} s{s}")
        seeds = done_seeds(cfg, m)
        if not seeds:
            write_json(state_path, {"model": m, "status": "failed", "stage": "train", "fails": fails})
            log(f"{m}: no finished headline run, moving to next model")
            continue
        best = best_seed(cfg, m, seeds)
        pre = {
            "id": cfg["models"][m]["id"],
            "tail": cfg["models"][m]["tail"],
            "repo": repo_name(cfg, m),
            "best_seed": best,
            "meta": read_json(run_dir(cfg, m, head, best) / "done.json"),
        }
        if hf:
            log(f"[{i}/{len(models)}] {m}: uploading to HuggingFace")
            try:
                publish_model(cfg, hf, m, pre, ctx, private=True)
            except Exception as ex:
                log(f"upload failed: {ex!r}")
        log(f"[{i}/{len(models)}] {m}: testing")
        if run_module("flipaware.evaluate", "base", m) != 0:
            write_json(state_path, {"model": m, "status": "failed", "stage": "eval_base", "fails": fails})
            continue
        for v, s in runs_for(cfg, m):
            if (run_dir(cfg, m, v, s) / "done.json").exists() and run_module("flipaware.evaluate", "run", m, v, s) != 0:
                fails.append(f"eval {v} s{s}")
        e = model_entry(cfg, m, test)
        if not e or not e.get("tuned"):
            write_json(state_path, {"model": m, "status": "failed", "stage": "eval", "fails": fails})
            continue
        good = e["verdict"]["good"]
        private = bool(run["private"]) or (not good and vcfg["private_if_bad"])
        if hf:
            try:
                repo = publish_model(cfg, hf, m, e, ctx, private=private, card_only=True)
                if slug and (good or run.get("collect_all")):
                    note = (
                        f"Base {e['id']}. Trap accuracy {e['base']['trap'] * 100:.1f} to {e['tuned']['trap']['mean'] * 100:.1f}"
                        + (f", NanoBEIR nDCG@10 {e['base']['ndcg'] * 100:.1f} to {e['tuned']['ndcg']['mean'] * 100:.1f}" if e["base"].get("ndcg") is not None else "")
                    )
                    add_item(hf, slug, repo, "model", note)
            except Exception as ex:
                log(f"publish failed: {ex!r}")
        show(e)
        write_json(state_path, {"model": m, "status": "done", "repo": e["repo"], "private": private, "verdict": e["verdict"], "fails": fails})
    entries = [e for e in (model_entry(cfg, m, test) for m in models) if e]
    print("\n" + table(entries) + "\n", flush=True)


if __name__ == "__main__":
    main()
