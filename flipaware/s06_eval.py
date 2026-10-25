import gc
import traceback

import torch

from flipaware.common import (
    eval_dtype,
    load_cfg,
    load_st,
    log,
    mrl_dims,
    plan_runs,
    prompts,
    read_json,
    read_jsonl,
    run_dir,
    set_seq,
    trap_eval,
    write_json,
)


def evaluate(cfg, mcfg, model, test):
    from sentence_transformers.evaluation import NanoBEIREvaluator

    q, d, s = prompts(mcfg)
    trap = trap_eval(model, test, s, mrl_dims(cfg, model.get_sentence_embedding_dimension()))
    kw = {"dataset_names": cfg["run"]["nanobeir"], "batch_size": 32, "show_progress_bar": False}
    if q:
        kw["query_prompts"] = q
    if d:
        kw["corpus_prompts"] = d
    ev = NanoBEIREvaluator(**kw)
    scores = ev(model)
    detail = {k: float(v) for k, v in scores.items() if k.endswith("ndcg@10")}
    return {"trap": trap, "nanobeir": {"mean": float(scores[ev.primary_metric]), "detail": detail}}


def release():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    cfg = load_cfg()
    run = cfg["run"]
    res = cfg["paths"]["results"]
    test = read_jsonl(cfg["paths"]["data"] / "test.jsonl")
    if not test:
        raise SystemExit("test split is empty")
    dtype = eval_dtype()
    fails = []
    jobs = [("base", m, None, None) for m in run["models"]]
    jobs += [("run", m, v, s) for m, v, s in plan_runs(cfg)]
    for kind, m, v, s in jobs:
        mcfg = cfg["models"][m]
        if kind == "base":
            out = res / f"base_{m}.json"
            path = None
        else:
            out = res / f"run_{m}__{v}__s{s}.json"
            rd = run_dir(cfg, m, v, s)
            if not (rd / "done.json").exists():
                continue
            path = rd / "model"
        if out.exists():
            continue
        log(f"eval {kind} {m} {v or ''} {'' if s is None else 's' + str(s)}")
        model = None
        try:
            model = load_st(mcfg, path=path, dtype=dtype)
            set_seq(model, run["eval_seq_len"])
            r = evaluate(cfg, mcfg, model, test)
            r.update(kind=kind, model=m, variant=v, seed=s, base_id=mcfg["id"])
            if kind == "run":
                r["meta"] = read_json(run_dir(cfg, m, v, s) / "done.json")
            write_json(out, r)
            top = r["trap"]["dims"][str(r["trap"]["full"])]
            log(f"trap {top['accuracy']:.4f} nanobeir {r['nanobeir']['mean']:.4f}")
        except Exception as ex:
            fails.append({"kind": kind, "model": m, "variant": v, "seed": s, "error": repr(ex)})
            traceback.print_exc()
        finally:
            model = None
            release()
    write_json(res / "eval_failures.json", fails)
    log(f"eval finished, {len(fails)} failed")


if __name__ == "__main__":
    main()
