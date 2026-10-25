import csv
import statistics

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from flipaware.common import dataset_name, load_cfg, log, plan_runs, read_json, read_jsonl, repo_name, run_dir, write_json

YES = {"1", "y", "yes", "true", "ok"}


def agg(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None
    return {"mean": sum(xs) / len(xs), "std": statistics.stdev(xs) if len(xs) > 1 else 0.0, "n": len(xs)}


def flat(r):
    top = r["trap"]["dims"][str(r["trap"]["full"])]
    return {
        "trap": top["accuracy"],
        "margin": top["margin"],
        "ndcg": r["nanobeir"]["mean"],
        "by_type": top["by_type"],
        "dims": {k: v["accuracy"] for k, v in r["trap"]["dims"].items()},
        "correct": top["correct"],
        "margins": top["margins"],
    }


def dev_acc(d):
    p = d / "train_log.json"
    if not p.exists():
        return None
    vals = [e["eval_trapdev_accuracy"] for e in read_json(p) if "eval_trapdev_accuracy" in e]
    return vals[-1] if vals else None


def gallery(test, base, tuned, k=8):
    idx = [i for i, (b, t) in enumerate(zip(base["correct"], tuned["correct"])) if not b and t]
    idx.sort(key=lambda i: -tuned["margins"][i])
    return [
        {
            "type": test[i]["type"],
            "anchor": test[i]["anchor"],
            "paraphrase": test[i]["paraphrase"],
            "trap": test[i]["trap"],
            "base_margin": round(base["margins"][i], 4),
            "tuned_margin": round(tuned["margins"][i], 4),
        }
        for i in idx[:k]
    ]


def handcheck(res):
    p = res / "handcheck_done.csv"
    if not p.exists():
        return None
    with p.open() as f:
        rows = list(csv.DictReader(f))
    out = {}
    for col in ("paraphrase_ok", "trap_ok"):
        vals = [(r.get(col) or "").strip().lower() for r in rows]
        vals = [v for v in vals if v]
        if vals:
            out[col] = {"checked": len(vals), "precision": sum(v in YES for v in vals) / len(vals)}
    return out


def grouped(path, labels, series, names, ylabel, errs=None, title=None):
    fig, ax = plt.subplots(figsize=(max(6, 0.9 * len(labels) + 2), 4.4))
    x = np.arange(len(labels))
    w = 0.8 / len(series)
    for i, vals in enumerate(series):
        ax.bar(
            x + (i - (len(series) - 1) / 2) * w,
            vals,
            w,
            label=names[i],
            yerr=None if errs is None else errs[i],
            capsize=3,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    if title:
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def mrl_plot(path, base, tuned, name, title):
    ds = sorted(int(d) for d in base["dims"])
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(ds, [base["dims"][str(d)] * 100 for d in ds], marker="o", label="Base")
    tv = [tuned["dims"].get(str(d)) for d in ds]
    if all(tv):
        ax.errorbar(ds, [t["mean"] * 100 for t in tv], yerr=[t["std"] * 100 for t in tv], marker="o", capsize=3, label=name)
    ax.set_xscale("log", base=2)
    ax.set_xticks(ds)
    ax.set_xticklabels([str(d) for d in ds])
    ax.set_xlabel("Vector dimensions")
    ax.set_ylabel("Trap accuracy (%)")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def curve_plot(path, logs, title):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    for name, hist in logs:
        tr = [(e["step"], e["loss"]) for e in hist if "loss" in e and "step" in e]
        ev = [(e["step"], e["eval_trapdev_accuracy"] * 100) for e in hist if "eval_trapdev_accuracy" in e]
        if tr:
            a1.plot(*zip(*tr), label=name)
        if ev:
            a2.plot(*zip(*ev), marker="o", label=name)
    a1.set_title("Training loss")
    a1.set_xlabel("Step")
    a2.set_title("Dev trap accuracy (%)")
    a2.set_xlabel("Step")
    for a in (a1, a2):
        a.grid(alpha=0.3)
        a.legend()
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    cfg = load_cfg()
    run = cfg["run"]
    res, assets = cfg["paths"]["results"], cfg["paths"]["assets"]
    head = cfg["headline_variant"]
    name = cfg["project"]["name"]
    test = read_jsonl(cfg["paths"]["data"] / "test.jsonl")
    plan = plan_runs(cfg)
    summary = {
        "project": name,
        "profile": cfg["profile"],
        "dataset": dataset_name(cfg),
        "trap_types": list(cfg["trap_types"]),
        "test_rows": len(test),
        "models": {},
    }
    if (res / "filter_stats.json").exists():
        summary["filter"] = read_json(res / "filter_stats.json")
    summary["handcheck"] = handcheck(res)
    for m in run["models"]:
        bp = res / f"base_{m}.json"
        if not bp.exists():
            continue
        mcfg = cfg["models"][m]
        base = flat(read_json(bp))
        e = {
            "id": mcfg["id"],
            "tail": mcfg["tail"],
            "repo": repo_name(cfg, m),
            "base": {k: base[k] for k in ("trap", "margin", "ndcg", "by_type", "dims")},
        }
        heads = []
        for s in run["seeds"]:
            p = res / f"run_{m}__{head}__s{s}.json"
            if p.exists():
                heads.append((s, flat(read_json(p))))
        if heads:
            hv = [h for _, h in heads]
            e["tuned"] = {
                "trap": agg([h["trap"] for h in hv]),
                "margin": agg([h["margin"] for h in hv]),
                "ndcg": agg([h["ndcg"] for h in hv]),
                "by_type": {ty: agg([h["by_type"].get(ty) for h in hv]) for ty in base["by_type"]},
                "dims": {dd: agg([h["dims"].get(dd) for h in hv]) for dd in base["dims"]},
                "seeds": len(hv),
            }
            best = max((dev_acc(run_dir(cfg, m, head, s)) or 0.0, s) for s, _ in heads)[1]
            e["best_seed"] = best
            e["meta"] = read_json(run_dir(cfg, m, head, best) / "done.json")
            e["seconds"] = sum(
                read_json(run_dir(cfg, mm, v, s) / "done.json")["seconds"]
                for mm, v, s in plan
                if mm == m and (run_dir(cfg, mm, v, s) / "done.json").exists()
            )
            e["gallery"] = gallery(test, base, dict(heads)[best])
            grouped(
                assets / f"types_{m}.png",
                list(base["by_type"]),
                [[base["by_type"][t] * 100 for t in base["by_type"]], [e["tuned"]["by_type"][t]["mean"] * 100 for t in base["by_type"]]],
                ["Base", name],
                "Trap accuracy (%)",
                errs=[None, [e["tuned"]["by_type"][t]["std"] * 100 for t in base["by_type"]]],
                title=mcfg["tail"],
            )
            mrl_plot(assets / f"mrl_{m}.png", base, e["tuned"], name, mcfg["tail"])
            logs = []
            for s, _ in heads:
                lp = run_dir(cfg, m, head, s) / "train_log.json"
                if lp.exists():
                    logs.append((f"seed {s}", read_json(lp)))
            if logs:
                curve_plot(assets / f"curve_{m}.png", logs, mcfg["tail"])
        if m in (run.get("ablation_models") or []):
            abl = {}
            for v in cfg["variants"]:
                p = res / f"run_{m}__{v}__s{run['seeds'][0]}.json"
                if p.exists():
                    f = flat(read_json(p))
                    abl[v] = {"trap": f["trap"], "ndcg": f["ndcg"], "margin": f["margin"]}
            if abl:
                e["ablation"] = abl
        summary["models"][m] = e
    done = [m for m, e in summary["models"].items() if e.get("tuned")]
    if done:
        labels = [summary["models"][m]["tail"] for m in done]
        grouped(
            assets / "headline.png",
            labels,
            [[summary["models"][m]["base"]["trap"] * 100 for m in done], [summary["models"][m]["tuned"]["trap"]["mean"] * 100 for m in done]],
            ["Base", name],
            "Trap accuracy (%)",
            errs=[None, [summary["models"][m]["tuned"]["trap"]["std"] * 100 for m in done]],
            title="Look alike sentence test",
        )
        grouped(
            assets / "retrieval.png",
            labels,
            [[summary["models"][m]["base"]["ndcg"] * 100 for m in done], [summary["models"][m]["tuned"]["ndcg"]["mean"] * 100 for m in done]],
            ["Base", name],
            "NanoBEIR nDCG@10",
            errs=[None, [summary["models"][m]["tuned"]["ndcg"]["std"] * 100 for m in done]],
            title="General retrieval",
        )
    write_json(res / "summary.json", summary)
    lines = ["| Model | Base trap | Tuned trap | Base nDCG@10 | Tuned nDCG@10 |", "|---|---|---|---|---|"]
    for m in done:
        e = summary["models"][m]
        lines.append(
            f"| {e['tail']} | {e['base']['trap'] * 100:.1f} | {e['tuned']['trap']['mean'] * 100:.1f} "
            f"| {e['base']['ndcg'] * 100:.1f} | {e['tuned']['ndcg']['mean'] * 100:.1f} |"
        )
    (res / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    log(f"report written for {len(done)} models")


if __name__ == "__main__":
    main()
