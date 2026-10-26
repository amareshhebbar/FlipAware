import sys

from flipaware.common import best_seed, eval_dtype, load_cfg, load_st, log, run_dir, set_seq, write_json


def scores_of(res):
    items = getattr(res, "task_results", res)
    out = {}
    for t in items:
        try:
            out[t.task_name] = float(t.get_score())
        except Exception:
            continue
    return out


def main():
    import mteb

    cfg = load_cfg()
    kind, m = sys.argv[1], sys.argv[2]
    res = cfg["paths"]["results"]
    out = res / f"mteb_{kind}_{m}.json"
    if out.exists():
        return
    mcfg = cfg["models"][m]
    if kind == "base":
        model = load_st(mcfg, dtype=eval_dtype())
    else:
        best = best_seed(cfg, m)
        if best is None:
            return
        model = load_st(mcfg, path=run_dir(cfg, m, cfg["headline_variant"], best) / "model", dtype=eval_dtype())
    set_seq(model, cfg["run"]["eval_seq_len"])
    tasks = mteb.get_tasks(tasks=cfg["run"]["mteb_tasks"], languages=["eng"])
    folder = res / "mteb" / f"{kind}_{m}"
    if hasattr(mteb, "evaluate"):
        r = mteb.evaluate(model, tasks=tasks)
    else:
        r = mteb.MTEB(tasks=tasks).run(model, output_folder=str(folder))
    sc = scores_of(r)
    write_json(out, {"kind": kind, "model": m, "scores": sc, "mean": sum(sc.values()) / max(1, len(sc))})
    log(f"mteb {kind} {m}: {sc}")


if __name__ == "__main__":
    main()
