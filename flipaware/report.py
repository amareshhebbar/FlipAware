import csv
import statistics

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from flipaware.common import (
    TRAP_SETS,
    best_seed,
    dataset_name,
    log,
    read_json,
    read_jsonl,
    repo_name,
    run_dir,
    words,
    write_json,
)

YES = {"1", "y", "yes", "true", "ok"}
NUMS = ("trap", "margin", "trap_xgen", "trap_human", "passage", "passage_r1", "nevir", "ndcg")
LEN_BUCKETS = [(8, 12, "8 to 12"), (13, 17, "13 to 17"), (18, 22, "18 to 22"), (23, 999, "23 or more")]
EDIT_BUCKETS = [(1, 2, "1 to 2"), (3, 4, "3 to 4"), (5, 999, "5 or more")]


def agg(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None
    return {"mean": sum(xs) / len(xs), "std": statistics.stdev(xs) if len(xs) > 1 else 0.0, "n": len(xs)}


def top(r):
    return r["dims"][str(r["full"])]


def flat(r):
    out = {"correct": {}, "margins": {}}
    if r.get("trap"):
        t = top(r["trap"])
        out.update(
            trap=t["accuracy"],
            margin=t["margin"],
            by_type=t["by_type"],
            dims={k: v["accuracy"] for k, v in r["trap"]["dims"].items()},
            quant=r["trap"].get("quant"),
        )
        out["correct"]["trap"] = t["correct"]
        out["margins"]["trap"] = t["margins"]
    for k in ("trap_xgen", "trap_human"):
        if r.get(k):
            t = top(r[k])
            out[k] = t["accuracy"]
            out[k + "_by_type"] = t["by_type"]
            out["correct"][k] = t["correct"]
    if r.get("passage"):
        out["passage"] = r["passage"]["pair_acc"]
        out["passage_r1"] = r["passage"]["recall1"]
    if r.get("nevir"):
        out["nevir"] = r["nevir"]["pair_acc"]
    if r.get("nanobeir"):
        out["ndcg"] = r["nanobeir"]["mean"]
    return out


def public(f):
    return {k: v for k, v in f.items() if k not in ("correct", "margins")}


def paired_boot(b, t, n, seed=0):
    b, t = np.asarray(b, float), np.asarray(t, float)
    if len(b) != len(t) or not len(b):
        return None
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(b), (n, len(b)))
    d = (t[idx] - b[idx]).mean(1)
    return {"delta": float((t - b).mean()), "lo": float(np.percentile(d, 2.5)), "hi": float(np.percentile(d, 97.5)), "p_not_better": float((d <= 0).mean())}


def bucket(v, buckets):
    for lo, hi, name in buckets:
        if lo <= v <= hi:
            return name
    return None


def dilution(test, base_ok, tuned_ok):
    out = {}
    for key, fn, buckets in (
        ("length", lambda r: len(r["anchor"].split()), LEN_BUCKETS),
        ("edit", lambda r: len(words(r["anchor"]) ^ words(r["trap"])), EDIT_BUCKETS),
    ):
        res = {}
        for r, b, t in zip(test, base_ok, tuned_ok):
            name = bucket(fn(r), buckets)
            if name is None:
                continue
            x = res.setdefault(name, {"n": 0, "base": 0, "tuned": 0})
            x["n"] += 1
            x["base"] += b
            x["tuned"] += t
        out[key] = {n: {"n": v["n"], "base": v["base"] / v["n"], "tuned": v["tuned"] / v["n"]} for n, v in res.items()}
    return out


def gallery(test, base, tuned, k=8):
    bc, tc = base["correct"]["trap"], tuned["correct"]["trap"]
    tm = tuned["margins"]["trap"]
    bm = base["margins"]["trap"]
    idx = [i for i, (b, t) in enumerate(zip(bc, tc)) if not b and t]
    idx.sort(key=lambda i: -tm[i])
    return [
        {
            "type": test[i]["type"],
            "anchor": test[i]["anchor"],
            "paraphrase": test[i]["paraphrase"],
            "trap": test[i]["trap"],
            "base_margin": round(bm[i], 4),
            "tuned_margin": round(tm[i], 4),
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


def context(cfg):
    res, data = cfg["paths"]["results"], cfg["paths"]["data"]
    ctx = {
        "project": cfg["project"]["name"],
        "profile": cfg["profile"],
        "dataset": dataset_name(cfg),
        "trap_types": list(cfg["trap_types"]),
        "test_rows": len(read_jsonl(data / "test.jsonl")),
        "xgen_rows": len(read_jsonl(data / "xgen_test.jsonl")),
        "human_rows": len(read_jsonl(data / "human_test.jsonl")),
        "passage_rows": len(read_jsonl(data / "passage_test.jsonl")),
        "nevir_rows": len(read_jsonl(data / "nevir_test.jsonl")),
        "generator": cfg["run"]["generator"],
        "xgen_generator": cfg["run"]["xgen_generator"],
        "handcheck": handcheck(res),
    }
    for k, f in (("filter", "filter_stats.json"), ("filter_xgen", "filter_stats_xgen.json"), ("human_stats", "human_stats.json")):
        if (res / f).exists():
            ctx[k] = read_json(res / f)
    return ctx


def grouped(path, labels, series, names, ylabel, errs=None, title=None):
    fig, ax = plt.subplots(figsize=(max(6, 0.9 * len(labels) + 2), 4.4))
    x = np.arange(len(labels))
    w = 0.8 / len(series)
    for i, vals in enumerate(series):
        ax.bar(x + (i - (len(series) - 1) / 2) * w, vals, w, label=names[i], yerr=None if errs is None else errs[i], capsize=3)
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
        if a.get_legend_handles_labels()[0]:
            a.legend()
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def line_plot(path, xs_list, ys_list, names, xlabel, ylabel, title, xlog=False, annotate=None):
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for i, (xs, ys, n) in enumerate(zip(xs_list, ys_list, names)):
        ax.plot(xs, ys, marker="o", label=n)
        if annotate:
            for x, y, t in zip(xs, ys, annotate[i]):
                ax.annotate(t, (x, y), textcoords="offset points", xytext=(4, 4), fontsize=8)
    if xlog:
        ax.set_xscale("symlog", linthresh=100)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def verdict(cfg, e):
    v = cfg["run"]["verdict"]
    gain = e["tuned"]["trap"]["mean"] - e["base"]["trap"]
    drop = (e["base"].get("ndcg") or 0) - ((e["tuned"].get("ndcg") or {}).get("mean") or 0)
    reasons = []
    if gain < v["min_trap_gain"]:
        reasons.append(f"trap accuracy gain {gain * 100:.1f} points is below the {v['min_trap_gain'] * 100:.1f} point target")
    if drop > v["max_ndcg_drop"]:
        reasons.append(f"retrieval nDCG@10 dropped {drop * 100:.1f} points, limit is {v['max_ndcg_drop'] * 100:.1f}")
    xg = None
    if e["base"].get("trap_xgen") is not None and e["tuned"].get("trap_xgen"):
        xg = e["tuned"]["trap_xgen"]["mean"] - e["base"]["trap_xgen"]
        if xg < v.get("min_xgen_gain", -1):
            reasons.append(f"cross generator gain {xg * 100:.1f} points: the gain may come from the generator's style")
    return {"good": not reasons, "label": "GOOD" if not reasons else "BAD", "trap_gain": gain, "ndcg_drop": drop, "xgen_gain": xg, "reasons": reasons}


def load_flat(path):
    return flat(read_json(path)) if path.exists() else None


def model_entry(cfg, m, test):
    run = cfg["run"]
    res, assets = cfg["paths"]["results"], cfg["paths"]["assets"]
    head = cfg["headline_variant"]
    name = cfg["project"]["name"]
    mcfg = cfg["models"][m]
    base = load_flat(res / f"base_{m}.json")
    if not base or "trap" not in base:
        return None
    e = {"id": mcfg["id"], "tail": mcfg["tail"], "repo": repo_name(cfg, m), "base": public(base)}
    heads = []
    for s in run["seeds"]:
        f = load_flat(res / f"run_{m}__{head}__s{s}.json")
        if f and "trap" in f:
            heads.append((s, f))
    if heads:
        hv = [h for _, h in heads]
        tuned = {k: agg([h.get(k) for h in hv]) for k in NUMS if any(h.get(k) is not None for h in hv)}
        tuned["by_type"] = {ty: agg([h["by_type"].get(ty) for h in hv]) for ty in base["by_type"]}
        tuned["dims"] = {dd: agg([h["dims"].get(dd) for h in hv]) for dd in base["dims"]}
        if all(h.get("quant") for h in hv):
            tuned["quant"] = {q: agg([h["quant"][q] for h in hv]) for q in hv[0]["quant"]}
        tuned["seeds"] = len(hv)
        e["tuned"] = tuned
        best = best_seed(cfg, m, [s for s, _ in heads])
        bf = dict(heads)[best]
        e["best_seed"] = best
        e["meta"] = read_json(run_dir(cfg, m, head, best) / "done.json")
        e["seconds"] = sum(
            read_json(p)["seconds"] for p in cfg["paths"]["runs"].glob(f"{m}__*/done.json")
        )
        e["gallery"] = gallery(test, base, bf)
        e["confidence"] = {}
        for k in TRAP_SETS:
            if k in base["correct"] and k in bf["correct"]:
                e["confidence"][k] = paired_boot(base["correct"][k], bf["correct"][k], run["bootstrap"])
        e["dilution"] = dilution(test, base["correct"]["trap"], bf["correct"]["trap"])
        types = list(base["by_type"])
        grouped(
            assets / f"types_{m}.png",
            types,
            [[base["by_type"][t] * 100 for t in types], [tuned["by_type"][t]["mean"] * 100 for t in types]],
            ["Base", name],
            "Trap accuracy (%)",
            errs=[None, [tuned["by_type"][t]["std"] * 100 for t in types]],
            title=mcfg["tail"],
        )
        mrl_plot(assets / f"mrl_{m}.png", base, tuned, name, mcfg["tail"])
        logs = []
        for s, _ in heads:
            lp = run_dir(cfg, m, head, s) / "train_log.json"
            if lp.exists():
                logs.append((f"seed {s}", read_json(lp)))
        if logs:
            curve_plot(assets / f"curve_{m}.png", logs, mcfg["tail"])
        e["verdict"] = verdict(cfg, e)
    s0 = run["seeds"][0]
    abl = {}
    for v, x in cfg["variants"].items():
        if x.get("scale") or m not in (run.get("ablation_models") or []):
            continue
        f = load_flat(res / f"run_{m}__{v}__s{s0}.json")
        if f and "trap" in f:
            abl[v] = {k: f.get(k) for k in ("trap", "ndcg", "trap_xgen", "passage")}
    if abl:
        e["ablation"] = abl
    if m in (run.get("scaling_models") or []) and e.get("tuned"):
        pts = [{"n": 0, "trap": base["trap"], "ndcg": base.get("ndcg")}]
        for v, x in cfg["variants"].items():
            if x.get("scale"):
                f = load_flat(res / f"run_{m}__{v}__s{s0}.json")
                if f and "trap" in f:
                    pts.append({"n": x["limit"], "trap": f["trap"], "ndcg": f.get("ndcg"), "trap_xgen": f.get("trap_xgen")})
        f = load_flat(res / f"run_{m}__{head}__s{s0}.json")
        n_full = read_json(run_dir(cfg, m, head, s0) / "done.json")["rows"]["trap"] if f else None
        if f and n_full:
            pts.append({"n": n_full, "trap": f["trap"], "ndcg": f.get("ndcg"), "trap_xgen": f.get("trap_xgen")})
        pts.sort(key=lambda p: p["n"])
        if len(pts) > 2:
            e["scaling"] = pts
    if m in (run.get("blend_models") or []) and e.get("tuned"):
        pts = [{"alpha": 0.0, "trap": base["trap"], "ndcg": base.get("ndcg")}]
        for a in run["blend_alphas"]:
            p = res / f"blend_{m}__a{a}.json"
            if p.exists():
                f = flat(read_json(p))
                pts.append({"alpha": a, "trap": f["trap"], "ndcg": f.get("ndcg")})
        bf = dict(heads)[e["best_seed"]]
        pts.append({"alpha": 1.0, "trap": bf["trap"], "ndcg": bf.get("ndcg")})
        if len(pts) > 2 and all(p["ndcg"] is not None for p in pts):
            lim = base["ndcg"] - run["verdict"]["max_ndcg_drop"]
            ok = [p for p in pts if p["ndcg"] >= lim]
            e["blend"] = {"points": pts, "best": max(ok, key=lambda p: p["trap"]) if ok else None}
    for key, fname in (("onnx", f"onnx_{m}.json"), ("mteb_base", f"mteb_base_{m}.json"), ("mteb_tuned", f"mteb_run_{m}.json")):
        if (res / fname).exists():
            e[key] = read_json(res / fname)
    return e


def leaderboard(cfg, entries):
    res = cfg["paths"]["results"]
    rows = []
    for m in cfg["run"].get("leaderboard") or []:
        f = load_flat(res / f"base_{m}.json")
        if f and "trap" in f:
            rows.append({"model": cfg["models"][m]["id"], "kind": "base", **{k: f.get(k) for k in NUMS if k != "margin"}, "binary": (f.get("quant") or {}).get("binary"), "by_type": f["by_type"]})
    user = cfg["project"]["hf_user"]
    for m, e in entries.items():
        if e.get("tuned") and e.get("verdict", {}).get("good", True):
            bf = load_flat(res / f"run_{m}__{cfg['headline_variant']}__s{e['best_seed']}.json")
            rows.append({"model": f"{user}/{e['repo']}", "kind": cfg["project"]["name"], **{k: bf.get(k) for k in NUMS if k != "margin"}, "binary": (bf.get("quant") or {}).get("binary"), "by_type": bf["by_type"]})
    rows.sort(key=lambda r: -(r["trap"] or 0))
    write_json(res / "leaderboard.json", rows)
    types = list(cfg["trap_types"])
    with (res / "leaderboard.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["rank", "model", "kind", "trap", "trap_xgen", "trap_human", "passage", "nevir", "binary"] + types)
        for i, r in enumerate(rows, 1):
            fmt = lambda x: "" if x is None else f"{x * 100:.1f}"
            w.writerow([i, r["model"], r["kind"], fmt(r["trap"]), fmt(r["trap_xgen"]), fmt(r["trap_human"]), fmt(r["passage"]), fmt(r["nevir"]), fmt(r["binary"])] + [fmt(r["by_type"].get(t)) for t in types])
    return rows


def leaderboard_plot(path, rows, name):
    if not rows:
        return
    rows = rows[:40]
    fig, ax = plt.subplots(figsize=(8, max(3, 0.3 * len(rows) + 1)))
    y = np.arange(len(rows))[::-1]
    colors = ["#c2410c" if r["kind"] == name else "#64748b" for r in rows]
    ax.barh(y, [r["trap"] * 100 for r in rows], color=colors)
    ax.set_yticks(y)
    ax.set_yticklabels([r["model"].split("/")[-1] for r in rows], fontsize=8)
    ax.set_xlabel("Trap accuracy (%)")
    ax.set_title(f"Look alike sentence test: open embedders vs {name}")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def build_summary(cfg):
    res, assets = cfg["paths"]["results"], cfg["paths"]["assets"]
    name = cfg["project"]["name"]
    test = read_jsonl(cfg["paths"]["data"] / "test.jsonl")
    summary = context(cfg)
    summary["models"] = {}
    for m in cfg["run"]["models"]:
        e = model_entry(cfg, m, test)
        if e:
            summary["models"][m] = e
    done = {m: e for m, e in summary["models"].items() if e.get("tuned")}
    if done:
        ms = list(done.values())
        labels = [e["tail"] for e in ms]
        grouped(
            assets / "headline.png",
            labels,
            [[e["base"]["trap"] * 100 for e in ms], [e["tuned"]["trap"]["mean"] * 100 for e in ms]],
            ["Base", name],
            "Trap accuracy (%)",
            errs=[None, [e["tuned"]["trap"]["std"] * 100 for e in ms]],
            title="Look alike sentence test",
        )
        if all(e["base"].get("ndcg") is not None and e["tuned"].get("ndcg") for e in ms):
            grouped(
                assets / "retrieval.png",
                labels,
                [[e["base"]["ndcg"] * 100 for e in ms], [e["tuned"]["ndcg"]["mean"] * 100 for e in ms]],
                ["Base", name],
                "NanoBEIR nDCG@10",
                errs=[None, [e["tuned"]["ndcg"]["std"] * 100 for e in ms]],
                title="General retrieval",
            )
        bench = [k for k in ("trap", "trap_xgen", "trap_human", "passage", "nevir") if all(e["base"].get(k) is not None and e["tuned"].get(k) for e in ms)]
        if len(bench) > 1:
            gain = {k: [(e["tuned"][k]["mean"] - e["base"][k]) * 100 for e in ms] for k in bench}
            grouped(assets / "benchmarks.png", labels, [gain[k] for k in bench], bench, "Gain over base (points)", title="Gain on every test set")
        agg_d = {}
        for key in ("length", "edit"):
            names = [b[2] for b in (LEN_BUCKETS if key == "length" else EDIT_BUCKETS)]
            agg_d[key] = {}
            for n in names:
                vals = [e["dilution"][key][n] for e in ms if n in e["dilution"][key]]
                if vals:
                    agg_d[key][n] = {"base": float(np.mean([v["base"] for v in vals])), "tuned": float(np.mean([v["tuned"] for v in vals])), "n": int(sum(v["n"] for v in vals))}
        summary["dilution"] = agg_d
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for ax, key, xl in ((axes[0], "length", "Anchor length (words)"), (axes[1], "edit", "Words changed by the trap")):
            ks = list(agg_d[key])
            ax.plot(ks, [agg_d[key][k]["base"] * 100 for k in ks], marker="o", label="Base models")
            ax.plot(ks, [agg_d[key][k]["tuned"] * 100 for k in ks], marker="o", label=name)
            ax.set_xlabel(xl)
            ax.set_ylabel("Trap accuracy (%)")
            ax.grid(alpha=0.3)
            ax.legend()
        fig.suptitle("Small edits in long sentences are the hardest to see")
        fig.tight_layout()
        fig.savefig(assets / "dilution.png", dpi=160)
        plt.close(fig)
        sc = [(e["tail"], e["scaling"]) for e in ms if e.get("scaling")]
        if sc:
            line_plot(
                assets / "scaling.png",
                [[p["n"] for p in pts] for _, pts in sc],
                [[p["trap"] * 100 for p in pts] for _, pts in sc],
                [t for t, _ in sc],
                "Training triples",
                "Trap accuracy (%)",
                "How much data does it take",
                xlog=True,
            )
        bl = [(e["tail"], e["blend"]["points"]) for e in ms if e.get("blend")]
        if bl:
            line_plot(
                assets / "blend.png",
                [[p["ndcg"] * 100 for p in pts] for _, pts in bl],
                [[p["trap"] * 100 for p in pts] for _, pts in bl],
                [t for t, _ in bl],
                "NanoBEIR nDCG@10",
                "Trap accuracy (%)",
                "Blending tuned and base weights",
                annotate=[[f"{p['alpha']:g}" for p in pts] for _, pts in bl],
            )
    lb = leaderboard(cfg, summary["models"])
    summary["leaderboard"] = lb
    leaderboard_plot(assets / "leaderboard.png", lb, name)
    write_json(res / "summary.json", summary)
    (res / "summary.md").write_text(table(list(summary["models"].values())) + "\n")
    log(f"summary written for {len(done)} models, leaderboard rows {len(lb)}")
    return summary


def pts(x):
    return "" if x is None else f"{x * 100:+.1f}"


def table(entries):
    lines = [
        "| Model | Base trap | Tuned trap | Gain | Cross generator gain | Human set gain | nDCG@10 change | Verdict |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for e in entries:
        if not e.get("tuned"):
            continue
        v = e["verdict"]
        hg = None
        if e["base"].get("trap_human") is not None and e["tuned"].get("trap_human"):
            hg = e["tuned"]["trap_human"]["mean"] - e["base"]["trap_human"]
        lines.append(
            f"| {e['tail']} | {e['base']['trap'] * 100:.1f} | {e['tuned']['trap']['mean'] * 100:.1f} | {pts(v['trap_gain'])} "
            f"| {pts(v['xgen_gain'])} | {pts(hg)} | {pts(-v['ndcg_drop'])} | {v['label']} |"
        )
    return "\n".join(lines)
