import json

BENCH_NAMES = {
    "trap": "Trap test (same generator, held out sentences)",
    "trap_xgen": "Cross generator test (written by a different model family)",
    "trap_human": "Human written test",
    "passage": "Passage retrieval traps (pairwise)",
    "passage_r1": "Passage retrieval traps (recall@1)",
    "nevir": "NevIR negation retrieval (pairwise)",
    "ndcg": "NanoBEIR nDCG@10 (general retrieval)",
}


def pct(x):
    return "n/a" if x is None else f"{x * 100:.1f}"


def ms(d):
    return "n/a" if not d else f"{d['mean'] * 100:.1f} ± {d['std'] * 100:.1f}"


def esc(s):
    return s.replace("|", "\\|").replace("\n", " ")


def kwarg(prompt):
    return f", prompt={json.dumps(prompt)}" if prompt else ""


def links(cfg, ctx):
    u, p = cfg["project"]["hf_user"], cfg["project"]["name"]
    sfx = cfg["run"]["repo_suffix"]
    return {
        "dataset": f"https://huggingface.co/datasets/{u}/{ctx['dataset']}",
        "leaderboard": f"https://huggingface.co/spaces/{u}/{p}_Leaderboard{sfx}",
        "code": f"https://github.com/{cfg['project']['github']}",
        "runs": f"https://huggingface.co/datasets/{u}/{p}_Runs{sfx}",
        "tensorboard": f"https://huggingface.co/spaces/{u}/{p}_TensorBoard{sfx}",
        "mlflow": f"https://huggingface.co/spaces/{u}/{p}_MLflow{sfx}",
    }


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
    rid = f"{cfg['project']['hf_user']}/{e['repo']}"
    rc = ", trust_remote_code=True" if mcfg.get("trust_remote_code") else ""
    out = [
        "## Usage",
        "",
        "```python",
        "from sentence_transformers import SentenceTransformer",
        "",
        f'model = SentenceTransformer("{rid}"{rc})',
        "",
        f'a, b = model.encode(["The drug is safe for children.", "The drug is not safe for children."]{kwarg(s)})',
        "print(model.similarity(a, b))",
        "",
        f'queries = model.encode(["is the drug safe for kids"]{kwarg(q)})',
        f'docs = model.encode(["The drug is not safe for children under 12."]{kwarg(d)})',
        "print(model.similarity(queries, docs))",
        "```",
        "",
        "Shorter vectors: pass `truncate_dim=256` (or any trained size) when loading.",
        "",
    ]
    if e.get("onnx"):
        out += [
            "ONNX on CPU, with int8 weights:",
            "",
            "```python",
            f'model = SentenceTransformer("{rid}", backend="onnx", model_kwargs={{"file_name": "onnx/model_quantized.onnx"}})',
            "```",
            "",
            "In the browser with transformers.js:",
            "",
            "```js",
            'import { pipeline } from "@huggingface/transformers";',
            f'const embed = await pipeline("feature-extraction", "{rid}", {{ dtype: "q8" }});',
            'const out = await embed(["The drug is not safe for children."], { pooling: "mean", normalize: true });',
            "```",
            "",
        ]
    out += ["Vectors from this model are not compatible with vectors from the base model. Re embed your corpus after switching.", ""]
    return out


def training(cfg, e, ctx):
    meta = e["meta"]
    lk = links(cfg, ctx)
    hours = e.get("seconds", meta.get("seconds", 0)) / 3600
    return [
        "## Training",
        "",
        f"* Data: [{ctx['dataset']}]({lk['dataset']}), anchor, paraphrase, and look alike trap across {len(ctx['trap_types'])} trap types, "
        "plus Natural Questions retrieval pairs so search skill does not decay.",
        "* Loss: CachedMultipleNegativesRankingLoss inside MatryoshkaLoss, with the trap as a hard negative.",
        "* Guard: CosineSimilarityLoss pulling similarity on general sentence pairs back to the base model's own scores.",
        f"* Matryoshka dims: {', '.join(str(x) for x in meta['dims'])}",
        f"* Epochs {meta['epochs']}, batch {meta['batch_size']} (mini batch {meta['mini_batch']}), learning rate {meta['lr']}, "
        f"max length {meta['max_seq_len']}, precision {meta['precision']}, LoRA {'yes' if meta.get('lora') else 'no'}",
        f"* Hardware: {meta.get('gpu')}, {hours:.2f} GPU hours for all runs of this base model",
        "* Training curves: the Training metrics tab and `assets/curve.png`.",
        f"* Every log and metric: [{cfg['project']['name']} Runs]({lk['runs']}), live in [TensorBoard]({lk['tensorboard']}) and [MLflow]({lk['mlflow']}).",
        f"* Code: {lk['code']}",
        "",
    ]


def model_card(cfg, m, e, ctx, assets):
    mcfg = cfg["models"][m]
    name = cfg["project"]["name"]
    lk = links(cfg, ctx)
    out = header(cfg, m, ctx)
    out += [
        f"# {e['repo']}",
        "",
        f"`{mcfg['id']}` fine tuned to tell apart sentences that look alike but mean different things, while keeping its normal search quality. "
        "It is a drop in replacement for the base model: same architecture, same vector size, same prompts.",
        "",
        f"Dataset: [{ctx['dataset']}]({lk['dataset']})  |  Leaderboard: [{name} Leaderboard]({lk['leaderboard']})  |  Logs: [{name} Runs]({lk['runs']})  |  Code: [GitHub]({lk['code']})",
        "",
    ]
    if not e.get("tuned"):
        out += ["Evaluation is running. Results will appear here when it finishes.", ""]
        return "\n".join(out + usage(cfg, m, e) + training(cfg, e, ctx))
    b, t, v = e["base"], e["tuned"], e["verdict"]
    if "types.png" in assets:
        out += ["![Trap accuracy by type](assets/types.png)", ""]
    out += [
        "## Results",
        "",
        f"Trap accuracy is the share of triples where the true paraphrase scores closer to the anchor than the look alike trap. "
        f"Tuned numbers are mean ± std over {t['seeds']} seed(s). Test sentences never appear in training.",
        "",
        "| Test | Rows | Base | Tuned |",
        "|---|---|---|---|",
    ]
    rows = {"trap": ctx["test_rows"], "trap_xgen": ctx.get("xgen_rows"), "trap_human": ctx.get("human_rows"), "passage": ctx.get("passage_rows"), "passage_r1": ctx.get("passage_rows"), "nevir": ctx.get("nevir_rows")}
    for k, label in BENCH_NAMES.items():
        if b.get(k) is None or not t.get(k):
            continue
        out.append(f"| {label} | {rows.get(k) or ''} | {pct(b[k])} | {ms(t[k])} |")
    out += [
        "",
        f"Verdict: **{v['label']}**. Trap accuracy {v['trap_gain'] * 100:+.1f} points, retrieval nDCG@10 {-v['ndcg_drop'] * 100:+.1f} points"
        + (f", cross generator {v['xgen_gain'] * 100:+.1f} points." if v.get("xgen_gain") is not None else "."),
        "",
    ]
    out += [f"* {r}" for r in v["reasons"]]
    if v["reasons"]:
        out.append("")
    conf = {k: c for k, c in (e.get("confidence") or {}).items() if c}
    if conf:
        out += ["## Confidence", "", "Paired bootstrap over test triples, best seed against the base model.", "", "| Test | Gain | 95% interval | Chance it is not better |", "|---|---|---|---|"]
        for k, c in conf.items():
            out.append(f"| {BENCH_NAMES[k].split(' (')[0]} | {c['delta'] * 100:+.1f} | {c['lo'] * 100:+.1f} to {c['hi'] * 100:+.1f} | {c['p_not_better'] * 100:.1f}% |")
        out.append("")
    out += ["## Accuracy by trap type", "", "| Trap type | Base | Tuned |", "|---|---|---|"]
    for ty, bv in b["by_type"].items():
        out.append(f"| {ty.replace('_', ' ')} | {pct(bv)} | {ms(t['by_type'].get(ty))} |")
    out += ["", "## Shorter and quantized vectors", "", "| Vector | Base | Tuned |", "|---|---|---|"]
    for dim in sorted(b["dims"], key=int, reverse=True):
        out.append(f"| {dim} dims float | {pct(b['dims'][dim])} | {ms(t['dims'].get(dim))} |")
    if b.get("quant") and t.get("quant"):
        out.append(f"| int8 | {pct(b['quant']['int8'])} | {ms(t['quant']['int8'])} |")
        out.append(f"| binary (1 bit per dim) | {pct(b['quant']['binary'])} | {ms(t['quant']['binary'])} |")
    if e.get("onnx"):
        o = e["onnx"]
        out.append(f"| ONNX fp32 (CPU) | | {pct(o.get('onnx_fp32'))} |")
        out.append(f"| ONNX int8 weights (CPU) | | {pct(o.get('onnx_int8'))} |")
    if "mrl.png" in assets:
        out += ["", "![Shorter vectors](assets/mrl.png)"]
    if e.get("gallery"):
        out += ["", "## Examples the base model got wrong", "", "| Type | Anchor | Paraphrase | Trap |", "|---|---|---|---|"]
        for g in e["gallery"]:
            out.append(f"| {g['type'].replace('_', ' ')} | {esc(g['anchor'])} | {esc(g['paraphrase'])} | {esc(g['trap'])} |")
    if e.get("ablation"):
        out += ["", "## Ablation", "", "| Variant | Trap | Cross generator | Passage | NanoBEIR nDCG@10 |", "|---|---|---|---|---|"]
        for vn, r in e["ablation"].items():
            out.append(f"| {vn.replace('_', ' ')} | {pct(r.get('trap'))} | {pct(r.get('trap_xgen'))} | {pct(r.get('passage'))} | {pct(r.get('ndcg'))} |")
        out += ["", "`mined guard` trains only on the triples this base model got wrong, plus a sample of easy ones."]
    if e.get("scaling"):
        out += ["", "## How much data it takes", "", "| Training triples | Trap | Cross generator | NanoBEIR nDCG@10 |", "|---|---|---|---|"]
        for p in e["scaling"]:
            out.append(f"| {p['n']} | {pct(p['trap'])} | {pct(p.get('trap_xgen'))} | {pct(p.get('ndcg'))} |")
        if "scaling.png" in assets:
            out += ["", "![Scaling](assets/scaling.png)"]
    if e.get("blend"):
        bl = e["blend"]
        out += ["", "## Blending tuned and base weights", "", "Weights blended as (1 - alpha) * base + alpha * tuned, with no extra training.", "", "| alpha | Trap | NanoBEIR nDCG@10 |", "|---|---|---|"]
        for p in bl["points"]:
            out.append(f"| {p['alpha']:g} | {pct(p['trap'])} | {pct(p['ndcg'])} |")
        if bl.get("best"):
            out += ["", f"Best blend inside the retrieval budget: alpha {bl['best']['alpha']:g}."]
        if "blend.png" in assets:
            out += ["", "![Blend](assets/blend.png)"]
    if e.get("dilution"):
        out += ["", "## Where embedders fail", "", "| Anchor length | Rows | Base | Tuned |", "|---|---|---|---|"]
        for k, x in e["dilution"]["length"].items():
            out.append(f"| {k} words | {x['n']} | {pct(x['base'])} | {pct(x['tuned'])} |")
        out += ["", "| Words changed | Rows | Base | Tuned |", "|---|---|---|---|"]
        for k, x in e["dilution"]["edit"].items():
            out.append(f"| {k} | {x['n']} | {pct(x['base'])} | {pct(x['tuned'])} |")
        if "dilution.png" in assets:
            out += ["", "![Dilution](assets/dilution.png)"]
    if e.get("mteb_base") and e.get("mteb_tuned"):
        mb, mt = e["mteb_base"]["scores"], e["mteb_tuned"]["scores"]
        out += ["", "## MTEB retrieval", "", "| Task | Base | Tuned |", "|---|---|---|"]
        for k in mt:
            out.append(f"| {k} | {pct(mb.get(k))} | {pct(mt[k])} |")
        out.append(f"| mean | {pct(e['mteb_base']['mean'])} | {pct(e['mteb_tuned']['mean'])} |")
    if "leaderboard.png" in assets:
        out += ["", "## Compared with other open embedders", "", f"Full table: [{name} Leaderboard]({lk['leaderboard']})", "", "![Leaderboard](assets/leaderboard.png)"]
    if "curve.png" in assets:
        out += ["", "## Training curves", "", "![Curves](assets/curve.png)"]
    out += [""]
    out += usage(cfg, m, e) + training(cfg, e, ctx)
    out += [
        "## Prior work",
        "",
        "NevIR (Weller et al., 2024) showed that neural retrievers fail when two documents differ only by negation. "
        "This model trains a fix across eight trap types, checks it on human written and cross generator test sets, "
        "and reports general retrieval next to it, so any regression is visible.",
        "",
        "## Limitations",
        "",
        "* English only.",
        f"* Training triples are written by {ctx['generator']}, which belongs to the same family as the Qwen3 embedders. The cross generator and human test sets exist to check for that bias.",
        "* Eight trap types; other meaning flips are not covered.",
        "* NanoBEIR is a small sample of BEIR, used as a quick check for general retrieval.",
        "",
    ]
    return "\n".join(out)


def dataset_card(cfg, ctx, counts, extras=None):
    from flipaware.s02_generate import SYSTEM, TEMPLATE

    extras = extras or {}
    p, run, flt = cfg["project"], cfg["run"], cfg["filter"]
    repo = f"{p['hf_user']}/{ctx['dataset']}"
    lk = links(cfg, ctx)
    desc = {
        "candidates": "Every generation with its NLI scores, word overlap, and the reason it was kept or rejected",
        "seeds": "Source sentences, the public dataset each came from, and the passage around it",
        "xgen_test": f"Test set written by a different model family ({ctx.get('xgen_generator')}) on the same held out sentences",
        "human_test": "Test set written and cross checked by people",
        "passage_test": "Passage retrieval traps: one sentence inside a real passage is flipped",
    }
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
        "- synthetic",
        "configs:",
        "- config_name: default",
        "  default: true",
        "  data_files:",
        "  - split: train",
        "    path: train.parquet",
        "  - split: validation",
        "    path: dev.parquet",
        "  - split: test",
        "    path: test.parquet",
    ]
    for name in desc:
        if name in extras:
            lines += [f"- config_name: {name}", "  data_files:", f"  - split: {'test' if name.endswith('_test') else 'train'}", f"    path: {name}.parquet"]
    lines += [
        "---",
        "",
        f"# {ctx['dataset']}",
        "",
        "Triples of an anchor sentence, a true paraphrase, and a look alike trap that shares most words but means something different. "
        f"Built by an automated pipeline and used to train and test the {p['name']} embedders.",
        "",
        f"Leaderboard: [{p['name']} Leaderboard]({lk['leaderboard']})  |  Code: [GitHub]({lk['code']})",
        "",
        "```python",
        "from datasets import load_dataset",
        "",
        f'triples = load_dataset("{repo}")',
    ]
    for name in desc:
        if name in extras:
            lines.append(f'{name} = load_dataset("{repo}", "{name}")')
    lines += ["```", "", "## Trap types", ""]
    for ty, rule in cfg["trap_types"].items():
        lines.append(f"* **{ty.replace('_', ' ')}**: {rule}")
    lines += ["", "## Contents", "", "| Config | Rows | What it holds |", "|---|---|---|"]
    for k, v in counts.items():
        lines.append(f"| default ({k}) | {v} | Filtered triples |")
    for name, d in desc.items():
        if name in extras:
            lines.append(f"| {name} | {extras[name]} | {d} |")
    lines += [
        "",
        "Splits are made by source sentence, so no anchor in test appears in train or validation.",
        "",
        "## Pipeline",
        "",
        "| Step | What happens |",
        "|---|---|",
        "| 1. Seeds | Real English sentences of 8 to 30 words from GooAQ answers and AllNLI premises, deduplicated and tagged when they contain a number |",
        f"| 2. Generate | {run['generator']} runs locally with thinking off and writes one paraphrase and one trap per seed and trap type, or skips when the rule cannot apply |",
        f"| 3. Filter | {run['nli_model']} scores entailment both ways. Kept only when the paraphrase entails the anchor both ways (at least {flt['entail_min']}), "
        f"the trap does not (below {flt['equiv_max']}), and the trap shares at least {int(flt['trap_overlap_min'] * 100)}% of its words with the anchor |",
        "| 4. Split | 80 / 10 / 10 by source sentence |",
        f"| 5. Cross generator | {ctx.get('xgen_generator')} rewrites the held out test sentences, filtered the same way |",
        "| 6. Passages | Test sentences that came from a multi sentence answer are flipped inside that answer |",
        "| 7. Human | People write and cross check new triples on sentences never shown to any generator |",
        "",
    ]
    for key, title in (("filter", "Filter statistics"), ("filter_xgen", "Filter statistics, cross generator")):
        stats = ctx.get(key)
        if not stats:
            continue
        lines += [f"## {title}", "", "| Trap type | Generated | Skipped | Unparsed | Paraphrase rejected | Trap rejected | Low overlap | Kept | Keep rate |", "|---|---|---|---|---|---|---|---|---|"]
        tot = {}
        for ty, st in stats.items():
            for k, v in st.items():
                tot[k] = tot.get(k, 0) + v
            lines.append(
                f"| {ty.replace('_', ' ')} | {st['generated']} | {st['skipped']} | {st['unparsed']} | {st.get('paraphrase_not_entailed', 0)} "
                f"| {st.get('trap_entailed', 0)} | {st.get('trap_low_overlap', 0)} | {st['kept']} | {st['kept'] / max(1, st['generated']) * 100:.1f}% |"
            )
        lines.append(
            f"| **all** | {tot.get('generated', 0)} | {tot.get('skipped', 0)} | {tot.get('unparsed', 0)} | {tot.get('paraphrase_not_entailed', 0)} "
            f"| {tot.get('trap_entailed', 0)} | {tot.get('trap_low_overlap', 0)} | {tot.get('kept', 0)} | {tot.get('kept', 0) / max(1, tot.get('generated', 1)) * 100:.1f}% |"
        )
        lines.append("")
    hs = ctx.get("human_stats")
    if hs:
        lines += [
            "## Human test set",
            "",
            f"{hs['submitted']} triples written, {hs['checks']} checks. A triple is kept when other people approve it more often than they reject it, "
            f"and its anchor never appears in training. Approved: {hs['approved']}, rejected: {hs['rejected']}, not yet checked: {hs['unchecked']}.",
            "",
        ]
    hc = ctx.get("handcheck")
    if hc:
        lines += ["## Hand check of the generated data", "", "| Field | Checked | Precision |", "|---|---|---|"]
        for k, v in hc.items():
            lines.append(f"| {k.replace('_', ' ')} | {v['checked']} | {v['precision'] * 100:.1f}% |")
        lines += ["", "The reviewed rows are in `handcheck.csv`.", ""]
    lines += [
        "## Generation prompt",
        "",
        "System:",
        "",
        "```",
        SYSTEM,
        "```",
        "",
        "User:",
        "",
        "```",
        TEMPLATE.replace("{{", "{").replace("}}", "}"),
        "```",
        "",
        "## Columns",
        "",
        "* default, xgen_test: `id`, `type`, `anchor`, `paraphrase`, `trap`",
        "* human_test: adds `author`, `yes`, `no` (check votes)",
        "* candidates: adds `generator`, `nli_para`, `nli_trap`, `overlap`, `kept`, `reason`",
        "* seeds: `seed_id`, `text`, `source`, `has_number`, `context`",
        "* passage_test: `query`, `original`, `twin` (the passage with one sentence flipped)",
        "",
        "## Sources and license",
        "",
        "Seed sentences come from [GooAQ](https://huggingface.co/datasets/sentence-transformers/gooaq) and "
        "[AllNLI](https://huggingface.co/datasets/sentence-transformers/all-nli). Released under CC BY SA 4.0 to match AllNLI.",
        "",
        "## Limitations",
        "",
        "* Generated rows are machine written and machine filtered, so some noise remains.",
        "* English only.",
        "",
        f"Pipeline code: {lk['code']}",
        "",
    ]
    return "\n".join(lines)
