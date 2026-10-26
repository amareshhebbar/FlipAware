import gc
import sys

import torch

from flipaware.common import best_seed, load_cfg, load_st, log, read_jsonl, run_dir, set_seq, write_json
from flipaware.evaluate import evaluate_model


def main():
    cfg = load_cfg()
    m = sys.argv[1]
    run, res = cfg["run"], cfg["paths"]["results"]
    mcfg = cfg["models"][m]
    head = cfg["headline_variant"]
    best = best_seed(cfg, m)
    if best is None:
        log(f"blend {m}: no tuned model")
        return
    alphas = [a for a in run["blend_alphas"] if not (res / f"blend_{m}__a{a}.json").exists()]
    if not alphas:
        return
    sets = {"trap": read_jsonl(cfg["paths"]["data"] / "test.jsonl"), "nanobeir": None}
    tuned = load_st(mcfg, path=run_dir(cfg, m, head, best) / "model", dtype=torch.float32)
    set_seq(tuned, run["eval_seq_len"])
    base = load_st(mcfg, dtype=torch.float32, device="cpu")
    bsd = {k: v.detach().clone() for k, v in base.state_dict().items()}
    del base
    tsd = {k: v.detach().cpu().clone() for k, v in tuned.state_dict().items()}
    matched = [k for k, v in tsd.items() if k in bsd and bsd[k].shape == v.shape and v.is_floating_point()]
    log(f"blend {m}: {len(matched)}/{len(tsd)} tensors matched, alphas {alphas}")
    for a in alphas:
        new = dict(tsd)
        for k in matched:
            new[k] = (1 - a) * bsd[k].float() + a * tsd[k].float()
        tuned.load_state_dict(new, strict=False)
        r = evaluate_model(cfg, mcfg, tuned, sets, full_dims=False)
        r.update(kind="blend", model=m, alpha=a, seed=best)
        write_json(res / f"blend_{m}__a{a}.json", r)
        log(f"alpha {a}: trap {r['trap']['dims'][str(r['trap']['full'])]['accuracy']:.4f} nanobeir {r['nanobeir']['mean']:.4f}")
    tuned = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
