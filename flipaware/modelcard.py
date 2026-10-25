import json


def pct(x):
    return f"{x * 100:.1f}"


def ms(d):
    return f"{d['mean'] * 100:.1f} ± {d['std'] * 100:.1f}"


def esc(s):
    return s.replace("|", "\\|").replace("\n", " ")


def kwarg(prompt):
    return f", prompt={json.dumps(prompt)}" if prompt else ""


def header(cfg, m, ctx):
    mcfg = cfg["models"][m]
    ds = f"{cfg['project']['hf_user']}/{ctx['dataset']}"
    return [
        "---",
        f"license: {mcfg['license']}",
        f"base_model: {mcfg['id']}",
        "library_name: sentence-transformers",
        "pipeline_tag: sentence-similarity",
        "language:",
        "- en",
        "tags:",
        "- sentence-transformers",
        "- embeddings",
        "- retrieval",
        "- hard-negatives",
        "- matryoshka",
        "- negation",
        "datasets:",
        f"- {ds}",
        "---",
        "",
    ]


def usage(cfg, m, e):
    mcfg = cfg["models"][m]
    q, d, s = mcfg.get("query_prompt") or "", mcfg.get("doc_prompt") or "", mcfg.get("sym_prompt") or ""
    return [
        "## Usage",
        "",
        "```python",
        "from sentence_transformers import SentenceTransformer",
        "",
        f'model = SentenceTransformer("{cfg["project"]["hf_user"]}/{e["repo"]}")',
        "",
        f'a, b = model.encode(["The drug is safe for children.", "The drug is not safe for children."]{kwarg(s)})',
        "print(model.similarity(a, b))",
        "",
        f'queries = model.encode(["is the drug safe for kids"]{kwarg(q)})',
        f'docs = model.encode(["The drug is not safe for children under 12."]{kwarg(d)})',
        "print(model.similarity(queries, docs))",
        "```",
        "",
        "Vectors from this model are not compatible with vectors from the base model. Re embed your corpus after switching.",
        "",
    ]


def training(cfg, e, ctx):
    p = cfg["project"]
    meta = e["meta"]
    ds = f"{p['hf_user']}/{ctx['dataset']}"
    hours = e.get("seconds", meta.get("seconds", 0)) / 3600
    return [
        "## Training",
        "",
        f"* Data: [{ds}](https://huggingface.co/datasets/{ds}), anchor, paraphrase, and look alike trap across {len(ctx['trap_types'])} trap types, "
        "plus Natural Questions retrieval pairs so search skill does not decay.",
        "* Loss: CachedMultipleNegativesRankingLoss inside MatryoshkaLoss, with the trap as a hard negative.",
        "* Guard: CosineSimilarityLoss pulling similarity on general sentence pairs back to the base model's own scores.",
        f"* Matryoshka dims: {', '.join(str(x) for x in meta['dims'])}",
        f"* Epochs {meta['epochs']}, batch {meta['batch_size']} (mini batch {meta['mini_batch']}), learning rate {meta['lr']}, "
        f"max length {meta['max_seq_len']}, precision {meta['precision']}",
        f"* Hardware: {meta.get('gpu')}, {hours:.2f} GPU hours",
        "* Training curves: see the Training metrics tab and `assets/curve.png`.",
        f"* Code: https://github.com/{p['github']}",
        "",
    ]


def model_card(cfg, m, e, ctx, assets):
    mcfg = cfg["models"][m]
    name = cfg["project"]["name"]
    out = header(cfg, m, ctx)
    out += [
        f"# {e['repo']}",
        "",
        f"`{mcfg['id']}` fine tuned to tell apart sentences that look alike but mean different things, while keeping its normal search quality. "
        "It is a drop in replacement for the base model: same architecture, same vector size, same prompts.",
        "",
    ]
    if not e.get("tuned"):
        out += ["Evaluation is running. Results will appear here when it finishes.", ""]
        out += usage(cfg, m, e) + training(cfg, e, ctx)
        return "\n".join(out)
    b, t, v = e["base"], e["tuned"], e["verdict"]
    if "types.png" in assets:
        out += ["![Trap accuracy by type](assets/types.png)", ""]
    out += [
        "## Results",
        "",
        f"Trap accuracy is the share of held out test triples where the true paraphrase scores closer to the anchor than the look alike trap. "
        f"Test sentences never appear in training ({ctx['test_rows']} triples). Tuned numbers are mean ± std over {t['seeds']} seed(s).",
        "",
        "| Model | Trap accuracy | Trap margin | NanoBEIR nDCG@10 |",
        "|---|---|---|---|",
        f"| Base | {pct(b['trap'])} | {b['margin']:.3f} | {pct(b['ndcg'])} |",
        f"| {name} | {ms(t['trap'])} | {t['margin']['mean']:.3f} | {ms(t['ndcg'])} |",
        "",
        f"Verdict: **{v['label']}**. Trap accuracy {v['trap_gain'] * 100:+.1f} points, retrieval nDCG@10 {-v['ndcg_drop'] * 100:+.1f} points.",
        "",
    ]
    for r in v["reasons"]:
        out.append(f"* {r}")
    if v["reasons"]:
        out.append("")
    out += ["## Accuracy by trap type", "", "| Trap type | Base | Tuned |", "|---|---|---|"]
    for ty, bv in b["by_type"].items():
        tv = t["by_type"].get(ty)
        out.append(f"| {ty.replace('_', ' ')} | {pct(bv)} | {ms(tv) if tv else 'n/a'} |")
    out += ["", "## Shorter vectors", "", "| Dimensions | Base | Tuned |", "|---|---|---|"]
    for dim in sorted(b["dims"], key=int, reverse=True):
        tv = t["dims"].get(dim)
        out.append(f"| {dim} | {pct(b['dims'][dim])} | {ms(tv) if tv else 'n/a'} |")
    if "mrl.png" in assets:
        out += ["", "![Shorter vectors](assets/mrl.png)"]
    if e.get("gallery"):
        out += ["", "## Examples the base model got wrong", "", "| Type | Anchor | Paraphrase | Trap |", "|---|---|---|---|"]
        for g in e["gallery"]:
            out.append(f"| {g['type'].replace('_', ' ')} | {esc(g['anchor'])} | {esc(g['paraphrase'])} | {esc(g['trap'])} |")
    if e.get("ablation"):
        out += ["", "## Ablation", "", "| Variant | Trap accuracy | NanoBEIR nDCG@10 |", "|---|---|---|"]
        for vn, r in e["ablation"].items():
            out.append(f"| {vn.replace('_', ' ')} | {pct(r['trap'])} | {pct(r['ndcg'])} |")
    if "headline.png" in assets:
        out += ["", f"## All {name} models", "", "![All models](assets/headline.png)"]
        if "retrieval.png" in assets:
            out += ["", "![General retrieval](assets/retrieval.png)"]
    if "curve.png" in assets:
        out += ["", "## Training curves", "", "![Curves](assets/curve.png)"]
    out += [""]
    out += usage(cfg, m, e) + training(cfg, e, ctx)
    out += [
        "## Prior work",
        "",
        "NevIR (Weller et al., 2024) showed that neural retrievers fail when two documents differ only by negation. "
        "This model trains a fix across eight trap types and reports general retrieval next to it, so any regression is visible.",
        "",
        "## Limitations",
        "",
        "* English only.",
        "* Training triples are generated by Qwen3 and filtered by an NLI model, so some noise remains.",
        "* Eight trap types; other meaning flips are not covered.",
        "* NanoBEIR is a small sample of BEIR, used as a quick check for general retrieval.",
        "",
    ]
    return "\n".join(out)


def dataset_card(cfg, ctx, counts):
    p = cfg["project"]
    lines = [
        "---",
        "license: cc-by-sa-4.0",
        "language:",
        "- en",
        "task_categories:",
        "- sentence-similarity",
        "tags:",
        "- hard-negatives",
        "- negation",
        "- embeddings",
        "configs:",
        "- config_name: default",
        "  data_files:",
        "  - split: train",
        "    path: train.parquet",
        "  - split: validation",
        "    path: dev.parquet",
        "  - split: test",
        "    path: test.parquet",
        "---",
        "",
        f"# {ctx['dataset']}",
        "",
        "Triples of an anchor sentence, a true paraphrase, and a look alike trap that shares most words but means something different. "
        f"Used to train and test the {p['name']} embedders.",
        "",
        "## Trap types",
        "",
    ]
    for ty, rule in cfg["trap_types"].items():
        lines.append(f"* **{ty.replace('_', ' ')}**: {rule}")
    lines += ["", "## Splits", "", "| Split | Rows |", "|---|---|"]
    for k, v in counts.items():
        lines.append(f"| {k} | {v} |")
    lines += [
        "",
        "Splits are made by source sentence, so no anchor in test appears in train.",
        "",
        "## How it was built",
        "",
        "* Source sentences from GooAQ answers and AllNLI premises.",
        f"* Paraphrase and trap written by {cfg['run']['generator']} with thinking off.",
        f"* Filtered with {cfg['run']['nli_model']}: the paraphrase must entail the anchor both ways; the trap must not, and must share at least "
        f"{int(cfg['filter']['trap_overlap_min'] * 100)}% of its words with the anchor.",
        "",
    ]
    hc = ctx.get("handcheck")
    if hc:
        lines += ["## Hand check", "", "| Field | Checked | Precision |", "|---|---|---|"]
        for k, v in hc.items():
            lines.append(f"| {k} | {v['checked']} | {v['precision'] * 100:.1f} |")
        lines.append("")
    lines += ["## Columns", "", "`id`, `type`, `anchor`, `paraphrase`, `trap`", "", f"Code: https://github.com/{p['github']}", ""]
    return "\n".join(lines)