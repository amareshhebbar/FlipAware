import json


def pct(x):
    return f"{x * 100:.1f}"


def ms(d):
    return f"{d['mean'] * 100:.1f} ± {d['std'] * 100:.1f}"


def esc(s):
    return s.replace("|", "\\|").replace("\n", " ")


def kwarg(prompt):
    return f", prompt={json.dumps(prompt)}" if prompt else ""


def model_card(cfg, mkey, e, summary, assets):
    p = cfg["project"]
    m = cfg["models"][mkey]
    user = p["hf_user"]
    name = p["name"]
    ds = f"{user}/{summary['dataset']}"
    q, d, s = m.get("query_prompt") or "", m.get("doc_prompt") or "", m.get("sym_prompt") or ""
    b, t, meta = e["base"], e["tuned"], e["meta"]
    out = [
        "---",
        f"license: {m['license']}",
        f"base_model: {m['id']}",
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
        f"# {e['repo']}",
        "",
        f"`{m['id']}` fine tuned to tell apart sentences that look alike but mean different things, while keeping its normal search quality. "
        "It is a drop in replacement for the base model: same architecture, same vector size, same prompts.",
        "",
        "![Trap accuracy](assets/headline.png)",
        "",
        "## Results",
        "",
        f"Trap accuracy is the share of held out test triples where the true paraphrase scores closer to the anchor than the look alike trap. "
        f"Test sentences never appear in training ({summary['test_rows']} triples). Tuned numbers are mean ± std over {t['seeds']} seed(s).",
        "",
        "| Model | Trap accuracy | Trap margin | NanoBEIR nDCG@10 |",
        "|---|---|---|---|",
        f"| Base | {pct(b['trap'])} | {b['margin']:.3f} | {pct(b['ndcg'])} |",
        f"| {name} | {ms(t['trap'])} | {t['margin']['mean']:.3f} | {ms(t['ndcg'])} |",
        "",
        "## Accuracy by trap type",
        "",
        "| Trap type | Base | Tuned |",
        "|---|---|---|",
    ]
    for ty, bv in b["by_type"].items():
        tv = t["by_type"].get(ty)
        out.append(f"| {ty.replace('_', ' ')} | {pct(bv)} | {ms(tv) if tv else 'n/a'} |")
    if f"types_{mkey}.png" in assets:
        out += ["", "![By trap type](assets/types.png)"]
    out += ["", "## Shorter vectors", "", "| Dimensions | Base | Tuned |", "|---|---|---|"]
    for dim in sorted(b["dims"], key=int, reverse=True):
        tv = t["dims"].get(dim)
        out.append(f"| {dim} | {pct(b['dims'][dim])} | {ms(tv) if tv else 'n/a'} |")
    if f"mrl_{mkey}.png" in assets:
        out += ["", "![Matryoshka](assets/mrl.png)"]
    if e.get("gallery"):
        out += [
            "",
            "## Examples the base model got wrong",
            "",
            "| Type | Anchor | Paraphrase | Trap |",
            "|---|---|---|---|",
        ]
        for g in e["gallery"]:
            out.append(f"| {g['type'].replace('_', ' ')} | {esc(g['anchor'])} | {esc(g['paraphrase'])} | {esc(g['trap'])} |")
    if e.get("ablation"):
        out += ["", "## Ablation", "", "| Variant | Trap accuracy | NanoBEIR nDCG@10 |", "|---|---|---|"]
        for v, r in e["ablation"].items():
            out.append(f"| {v.replace('_', ' ')} | {pct(r['trap'])} | {pct(r['ndcg'])} |")
    out += [
        "",
        "## Usage",
        "",
        "```python",
        "from sentence_transformers import SentenceTransformer",
        "",
        f'model = SentenceTransformer("{user}/{e["repo"]}")',
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
        "## Training",
        "",
        f"* Data: [{ds}](https://huggingface.co/datasets/{ds}), anchor, paraphrase, and look alike trap across {len(summary['trap_types'])} trap types, "
        "plus Natural Questions retrieval pairs so search skill does not decay.",
        "* Loss: CachedMultipleNegativesRankingLoss inside MatryoshkaLoss, with the trap as a hard negative.",
        "* Guard: CosineSimilarityLoss pulling similarity on general sentence pairs back to the base model's own scores.",
        f"* Matryoshka dims: {', '.join(str(x) for x in meta['dims'])}",
        f"* Epochs {meta['epochs']}, batch {meta['batch_size']} (mini batch {meta['mini_batch']}), learning rate {meta['lr']}, max length {meta['max_seq_len']}, precision {meta['precision']}",
        f"* Hardware: {meta.get('gpu')}, {e.get('seconds', 0) / 3600:.2f} GPU hours across all runs for this base model",
        f"* Code: https://github.com/{p['github']}",
        "",
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


def dataset_card(cfg, summary, counts):
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
        f"# {summary['dataset']}",
        "",
        "Triples of an anchor sentence, a true paraphrase, and a look alike trap that shares most words but means something different. "
        f"Used to train and test the {p['name']} embedders.",
        "",
        "## Trap types",
        "",
    ]
    for ty, rule in cfg["trap_types"].items():
        lines.append(f"* **{ty.replace('_', ' ')}**: {rule}")
    lines += [
        "",
        "## Splits",
        "",
        "| Split | Rows |",
        "|---|---|",
    ]
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
    hc = summary.get("handcheck")
    if hc:
        lines += ["## Hand check", "", "| Field | Checked | Precision |", "|---|---|---|"]
        for k, v in hc.items():
            lines.append(f"| {k} | {v['checked']} | {v['precision'] * 100:.1f} |")
        lines.append("")
    lines += [
        "## Columns",
        "",
        "`id`, `type`, `anchor`, `paraphrase`, `trap`",
        "",
        f"Code: https://github.com/{p['github']}",
        "",
    ]
    return "\n".join(lines)
