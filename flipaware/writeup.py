from flipaware.modelcard import links


def p(x):
    return "n/a" if x is None else f"{x * 100:.1f}"


def gain(e, k):
    if e["base"].get(k) is None or not e["tuned"].get(k):
        return None
    return e["tuned"][k]["mean"] - e["base"][k]


def report(cfg, s):
    name = cfg["project"]["name"]
    lk = links(cfg, s)
    done = [e for e in s["models"].values() if e.get("tuned")]
    good = [e for e in done if e["verdict"]["good"]]
    lb = [r for r in s.get("leaderboard", []) if r["kind"] == "base"]
    out = [
        f"# {name}: embedders that notice when one word flips the meaning",
        "",
        "## The problem",
        "",
        "Embedding models score \"the drug is safe\" and \"the drug is not safe\" as nearly the same sentence. "
        "Search and RAG built on them return the wrong passage whenever a question turns on a negation, a number, an order of events, or who did what to whom.",
        "",
    ]
    if lb:
        best = max(lb, key=lambda r: r["trap"])
        worst = min(lb, key=lambda r: r["trap"])
        mean = sum(r["trap"] for r in lb) / len(lb)
        out += [
            "## How common is it",
            "",
            f"I tested {len(lb)} popular open embedders on {s['test_rows']} held out look alike triples. Average trap accuracy: {p(mean)}%. "
            f"Best: {best['model']} at {p(best['trap'])}%. Worst: {worst['model']} at {p(worst['trap'])}%. A random guess scores 50%.",
            "",
            f"![Leaderboard]({lk['leaderboard']}/resolve/main/leaderboard.png)",
            "",
        ]
    out += [
        "## What I built",
        "",
        f"* A data pipeline: real sentences from GooAQ and AllNLI, paraphrases and traps written locally by {s['generator']}, filtered with an NLI model, split by source sentence. Every rejected row and its reason is published.",
        f"* {len(done)} fine tuned embedders across {len({e['id'].split('/')[0] for e in done})} model families, trained with Matryoshka, cached contrastive loss, the trap as a hard negative, and a guard loss that keeps general retrieval in place.",
        f"* Four independent test sets: the held out split, a cross generator set written by {s['xgen_generator']}, a human written set, and passage level traps. Plus NevIR.",
        "",
        "## Results",
        "",
        "| Model | Trap | Cross generator | Human | Passage | NevIR | NanoBEIR change | Verdict |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for e in done:
        g = lambda k: "" if gain(e, k) is None else f"{gain(e, k) * 100:+.1f}"
        out.append(f"| {e['tail']} | {g('trap')} | {g('trap_xgen')} | {g('trap_human')} | {g('passage')} | {g('nevir')} | {g('ndcg')} | {e['verdict']['label']} |")
    out += ["", f"{len(good)} of {len(done)} models passed every check. Numbers are gains in points over each base model.", ""]
    d = s.get("dilution")
    if d and d.get("length"):
        ks = list(d["length"])
        out += [
            "## Why embedders miss it",
            "",
            f"Mean pooling averages one changed word into the whole sentence. Base models drop from {p(d['length'][ks[0]]['base'])}% on short anchors "
            f"to {p(d['length'][ks[-1]]['base'])}% on long ones; tuned models hold {p(d['length'][ks[0]]['tuned'])}% to {p(d['length'][ks[-1]]['tuned'])}%.",
            "",
        ]
    sc = [e for e in done if e.get("scaling")]
    if sc:
        e = sc[0]
        out += ["## How much data it takes", "", "| Triples | Trap |", "|---|---|"] + [f"| {x['n']} | {p(x['trap'])} |" for x in e["scaling"]] + [""]
    bl = [e for e in done if e.get("blend") and e["blend"].get("best")]
    if bl:
        out += ["## A free extra", "", "Blending tuned and base weights trades trap accuracy against retrieval with no training:", ""]
        for e in bl:
            b = e["blend"]["best"]
            out.append(f"* {e['tail']}: alpha {b['alpha']:g} keeps trap accuracy at {p(b['trap'])}% inside the retrieval budget")
        out.append("")
    out += [
        "## Honest limits",
        "",
        "* English only, eight trap types.",
        f"* {s['generator']} writes the training data and shares a family with the Qwen3 embedders; the cross generator and human sets are there to catch that.",
        "* NanoBEIR is a quick proxy for general retrieval.",
        "",
        "## Links",
        "",
        f"* Dataset: {lk['dataset']}",
        f"* Leaderboard: {lk['leaderboard']}",
        f"* Code: {lk['code']}",
        "",
    ]
    return "\n".join(out)
