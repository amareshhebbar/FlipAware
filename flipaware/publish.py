import html
import os
import shutil

import pandas as pd

from flipaware.common import hf_token, log, read_json, read_jsonl, run_dir, space_name, write_json
from flipaware.modelcard import dataset_card, model_card

COLS = ["id", "type", "anchor", "paraphrase", "trap"]
EXTRAS = (
    ("candidates", "candidates.jsonl", ["id", "type", "anchor", "paraphrase", "trap", "generator", "nli_para", "nli_trap", "overlap", "kept", "reason"]),
    ("seeds", "seeds.jsonl", ["seed_id", "text", "source", "has_number", "context"]),
    ("xgen_test", "xgen_test.jsonl", COLS),
    ("human_test", "human_test.jsonl", COLS + ["author", "yes", "no"]),
    ("passage_test", "passage_test.jsonl", ["id", "type", "query", "original", "twin", "anchor", "trap"]),
)


def api():
    if os.environ.get("FLIP_PUBLISH", "1") != "1":
        return None
    token = hf_token()
    if not token:
        log("no HuggingFace login, publishing disabled")
        return None
    from huggingface_hub import HfApi

    return HfApi(token=token)


def ensure_collection(cfg, hf):
    path = cfg["paths"]["results"] / "collection.json"
    if path.exists():
        return read_json(path)["slug"]
    proj, run = cfg["project"], cfg["run"]
    title = proj["collection_title"] + (" smoke" if run["repo_suffix"] else "")
    coll = hf.create_collection(title=title, namespace=proj["hf_user"], description=proj["collection_description"], private=bool(run["private"]), exists_ok=True)
    write_json(path, {"slug": coll.slug})
    log(f"collection https://huggingface.co/collections/{coll.slug}")
    return coll.slug


def add_item(hf, slug, item_id, kind, note):
    try:
        hf.add_collection_item(slug, item_id=item_id, item_type=kind, note=note[:500], exists_ok=True)
    except Exception as ex:
        log(f"collection add failed for {item_id}: {ex!r}")


def set_private(hf, repo, private, kind="model"):
    try:
        hf.update_repo_settings(repo_id=repo, private=private, repo_type=kind)
    except (AttributeError, TypeError):
        hf.update_repo_visibility(repo_id=repo, private=private, repo_type=kind)


def stage_dataset(cfg, ctx):
    data, res, pub = cfg["paths"]["data"], cfg["paths"]["results"], cfg["paths"]["publish"]
    root = pub / ctx["dataset"]
    shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True)
    counts = {}
    for split in ("train", "dev", "test"):
        rows = read_jsonl(data / f"{split}.jsonl")
        pd.DataFrame(rows, columns=COLS).to_parquet(root / f"{split}.parquet", index=False)
        counts[split] = len(rows)
    extras = {}
    for name, src, cols in EXTRAS:
        rows = read_jsonl(data / src)
        if rows:
            pd.DataFrame(rows).reindex(columns=cols).to_parquet(root / f"{name}.parquet", index=False)
            extras[name] = len(rows)
    for f in ("filter_stats.json", "filter_stats_xgen.json", "human_stats.json"):
        if (res / f).exists():
            shutil.copy(res / f, root / f)
    if (res / "handcheck_done.csv").exists():
        shutil.copy(res / "handcheck_done.csv", root / "handcheck.csv")
    (root / "README.md").write_text(dataset_card(cfg, ctx, counts, extras))
    return root, counts


def publish_dataset(cfg, hf, ctx):
    user = cfg["project"]["hf_user"]
    root, counts = stage_dataset(cfg, ctx)
    repo = f"{user}/{ctx['dataset']}"
    hf.create_repo(repo, repo_type="dataset", private=bool(cfg["run"]["private"]), exist_ok=True)
    hf.upload_folder(folder_path=str(root), repo_id=repo, repo_type="dataset", commit_message="Upload FlipAware data")
    log(f"dataset https://huggingface.co/datasets/{repo}")
    return repo, counts


def trim(r):
    r = dict(r)
    for k, v in list(r.items()):
        if isinstance(v, dict) and isinstance(v.get("dims"), dict):
            v = dict(v)
            v["dims"] = {d: {x: y for x, y in rec.items() if x not in ("correct", "margins")} for d, rec in v["dims"].items()}
            r[k] = v
    return r


ASSET_MAP = (
    ("headline.png", "headline.png"),
    ("retrieval.png", "retrieval.png"),
    ("benchmarks.png", "benchmarks.png"),
    ("dilution.png", "dilution.png"),
    ("scaling.png", "scaling.png"),
    ("blend.png", "blend.png"),
    ("leaderboard.png", "leaderboard.png"),
)


def stage_model(cfg, m, e, ctx, root, card_only=False):
    res, assets = cfg["paths"]["results"], cfg["paths"]["assets"]
    head = cfg["headline_variant"]
    rd = run_dir(cfg, m, head, e["best_seed"])
    shutil.rmtree(root, ignore_errors=True)
    if card_only:
        root.mkdir(parents=True)
        if (rd / "model" / "onnx").exists():
            shutil.copytree(rd / "model" / "onnx", root / "onnx")
    else:
        shutil.copytree(rd / "model", root)
    (root / "assets").mkdir(exist_ok=True)
    (root / "results").mkdir(exist_ok=True)
    present = set()
    per = (
        (f"types_{m}.png", "types.png"),
        (f"mrl_{m}.png", "mrl.png"),
        (f"curve_{m}.png", "curve.png"),
    )
    for src, dst in ASSET_MAP + per:
        if (assets / src).exists():
            shutil.copy(assets / src, root / "assets" / dst)
            present.add(dst)
    for f in sorted(res.glob(f"*_{m}.json")) + sorted(res.glob(f"*_{m}__*.json")):
        write_json(root / "results" / f.name, trim(read_json(f)))
    if not card_only:
        for s in cfg["run"]["seeds"]:
            tb = run_dir(cfg, m, head, s) / "tb"
            if tb.exists():
                shutil.copytree(tb, root / "runs" / f"seed{s}", dirs_exist_ok=True)
    (root / "README.md").write_text(model_card(cfg, m, e, ctx, present))


def publish_model(cfg, hf, m, e, ctx, private, card_only=False):
    repo = f"{cfg['project']['hf_user']}/{e['repo']}"
    root = cfg["paths"]["publish"] / e["repo"]
    stage_model(cfg, m, e, ctx, root, card_only=card_only)
    hf.create_repo(repo, private=private, exist_ok=True)
    hf.upload_folder(folder_path=str(root), repo_id=repo, commit_message="Update model card and results" if card_only else f"Upload {e['repo']}")
    set_private(hf, repo, private)
    log(f"{'updated' if card_only else 'uploaded'} https://huggingface.co/{repo}{' (private)' if private else ''}")
    return repo


def leaderboard_html(cfg, rows, ctx):
    name = cfg["project"]["name"]
    types = list(cfg["trap_types"])
    cols = [("model", "Model"), ("trap", "Trap"), ("trap_xgen", "Cross gen"), ("trap_human", "Human"), ("passage", "Passage"), ("nevir", "NevIR"), ("binary", "Binary")]
    cols += [(t, t.replace("_", " ")) for t in types]
    head = "".join(f'<th onclick="sortBy({i})">{html.escape(c[1])}</th>' for i, c in enumerate(cols))
    body = []
    for r in rows:
        tds = []
        for k, _ in cols:
            if k == "model":
                tag = f' <span class="tag">{html.escape(name)}</span>' if r["kind"] == name else ""
                tds.append(f'<td><a href="https://huggingface.co/{html.escape(r["model"])}">{html.escape(r["model"])}</a>{tag}</td>')
            else:
                v = r["by_type"].get(k) if k in types else r.get(k)
                tds.append(f'<td data-v="{-1 if v is None else v}">{"" if v is None else f"{v * 100:.1f}"}</td>')
        body.append(f'<tr class="{"ours" if r["kind"] == name else ""}">{"".join(tds)}</tr>')
    imgs = "".join(f'<img src="{p}" alt="{p}">' for p in ("leaderboard.png", "dilution.png", "benchmarks.png") if (cfg["paths"]["assets"] / p).exists())
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(name)} Leaderboard</title>
<style>
:root{{--bg:#fff;--fg:#0f172a;--mute:#64748b;--line:#e2e8f0;--hi:#fff7ed;--acc:#c2410c}}
@media (prefers-color-scheme:dark){{:root{{--bg:#0b1120;--fg:#e2e8f0;--mute:#94a3b8;--line:#1e293b;--hi:#2a1a10;--acc:#fb923c}}}}
body{{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif}}
main{{max-width:1200px;margin:0 auto;padding:24px 16px}}
h1{{margin:0 0 4px}} p{{color:var(--mute);max-width:820px}}
.wrap{{overflow-x:auto;border:1px solid var(--line);border-radius:8px}}
table{{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}}
th,td{{padding:6px 10px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}}
th{{cursor:pointer;position:sticky;top:0;background:var(--bg)}} th:first-child,td:first-child{{text-align:left}}
tr.ours{{background:var(--hi)}} a{{color:inherit}} .tag{{color:var(--acc);font-size:12px;font-weight:600}}
img{{max-width:100%;margin:16px 0;border:1px solid var(--line);border-radius:8px}}
</style></head><body><main>
<h1>{html.escape(name)} Leaderboard</h1>
<p>Can an embedding model tell apart sentences that look alike but mean different things? Each score is the share of test items where the true paraphrase
lands closer to the anchor than a look alike trap. Trap: {ctx['test_rows']} held out triples. Cross gen: {ctx.get('xgen_rows', 0)} triples written by a different model family.
Human: {ctx.get('human_rows', 0)} human written triples. Passage: one sentence flipped inside a real passage. NevIR: public negation retrieval benchmark. Binary: trap accuracy with 1 bit vectors.
Click a column to sort.</p>
<p><a href="https://huggingface.co/datasets/{cfg['project']['hf_user']}/{ctx['dataset']}">Dataset</a> | <a href="https://github.com/{cfg['project']['github']}">Code</a></p>
<div class="wrap"><table id="t"><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table></div>
{imgs}
</main>
<script>
let dir={{}};
function sortBy(i){{const tb=document.querySelector('#t tbody');const rows=[...tb.rows];dir[i]=!dir[i];
rows.sort((a,b)=>{{const x=a.cells[i],y=b.cells[i];if(i===0)return dir[i]?x.innerText.localeCompare(y.innerText):y.innerText.localeCompare(x.innerText);
return dir[i]?(+y.dataset.v)-(+x.dataset.v):(+x.dataset.v)-(+y.dataset.v)}});rows.forEach(r=>tb.appendChild(r))}}
</script></body></html>
"""


def publish_leaderboard(cfg, hf, rows, ctx):
    if not rows:
        return None
    user = cfg["project"]["hf_user"]
    repo = f"{user}/{space_name(cfg, 'Leaderboard')}"
    root = cfg["paths"]["publish"] / "leaderboard_space"
    shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True)
    (root / "index.html").write_text(leaderboard_html(cfg, rows, ctx))
    (root / "README.md").write_text(f"---\ntitle: {cfg['project']['name']} Leaderboard\ncolorFrom: gray\ncolorTo: red\nsdk: static\npinned: false\n---\n")
    for p in ("leaderboard.png", "dilution.png", "benchmarks.png"):
        if (cfg["paths"]["assets"] / p).exists():
            shutil.copy(cfg["paths"]["assets"] / p, root / p)
    for p in ("leaderboard.csv", "leaderboard.json"):
        if (cfg["paths"]["results"] / p).exists():
            shutil.copy(cfg["paths"]["results"] / p, root / p)
    hf.create_repo(repo, repo_type="space", space_sdk="static", private=bool(cfg["run"]["private"]), exist_ok=True)
    hf.upload_folder(folder_path=str(root), repo_id=repo, repo_type="space", commit_message="Update leaderboard")
    log(f"leaderboard https://huggingface.co/spaces/{repo}")
    return repo
