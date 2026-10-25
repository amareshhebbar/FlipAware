import os
import shutil

import pandas as pd
from huggingface_hub import HfApi

from flipaware.common import load_cfg, log, read_json, read_jsonl, run_dir, write_json
from flipaware.modelcard import dataset_card, model_card

COLS = ["id", "type", "anchor", "paraphrase", "trap"]


def trim(r):
    r = dict(r)
    for d in r.get("trap", {}).get("dims", {}).values():
        d.pop("correct", None)
        d.pop("margins", None)
    return r


def stage_dataset(cfg, summary, root):
    data = cfg["paths"]["data"]
    shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True)
    counts = {}
    for split in ("train", "dev", "test"):
        rows = read_jsonl(data / f"{split}.jsonl")
        pd.DataFrame(rows, columns=COLS).to_parquet(root / f"{split}.parquet", index=False)
        counts[split] = len(rows)
    (root / "README.md").write_text(dataset_card(cfg, summary, counts))
    return counts


def stage_model(cfg, mkey, e, summary, root):
    res, assets = cfg["paths"]["results"], cfg["paths"]["assets"]
    head = cfg["headline_variant"]
    rd = run_dir(cfg, mkey, head, e["best_seed"])
    shutil.rmtree(root, ignore_errors=True)
    shutil.copytree(rd / "model", root)
    (root / "assets").mkdir(exist_ok=True)
    (root / "results").mkdir(exist_ok=True)
    present = set()
    for src, dst in (
        ("headline.png", "headline.png"),
        ("retrieval.png", "retrieval.png"),
        (f"types_{mkey}.png", "types.png"),
        (f"mrl_{mkey}.png", "mrl.png"),
        (f"curve_{mkey}.png", "curve.png"),
    ):
        if (assets / src).exists():
            shutil.copy(assets / src, root / "assets" / dst)
            present.add(src)
    for f in [res / f"base_{mkey}.json"] + sorted(res.glob(f"run_{mkey}__*.json")):
        if f.exists():
            write_json(root / "results" / f.name, trim(read_json(f)))
    if (rd / "tb").exists():
        shutil.copytree(rd / "tb", root / "runs" / rd.name, dirs_exist_ok=True)
    (root / "README.md").write_text(model_card(cfg, mkey, e, summary, present))


def main():
    cfg = load_cfg()
    token = os.environ.get("HF_TOKEN")
    if not token:
        log("HF_TOKEN missing, skipping publish")
        return
    run, proj = cfg["run"], cfg["project"]
    res, pub = cfg["paths"]["results"], cfg["paths"]["publish"]
    user, private = proj["hf_user"], bool(run["private"])
    api = HfApi(token=token)
    summary = read_json(res / "summary.json")
    items = []

    ds_repo = f"{user}/{summary['dataset']}"
    counts = stage_dataset(cfg, summary, pub / summary["dataset"])
    api.create_repo(ds_repo, repo_type="dataset", private=private, exist_ok=True)
    api.upload_folder(folder_path=str(pub / summary["dataset"]), repo_id=ds_repo, repo_type="dataset", commit_message="Upload trap triples")
    items.append((ds_repo, "dataset", f"{sum(counts.values())} anchor, paraphrase, trap triples across {len(cfg['trap_types'])} trap types"))
    log(f"published {ds_repo}")

    for m in run["models"]:
        e = summary["models"].get(m)
        if not e or not e.get("tuned"):
            continue
        repo = f"{user}/{e['repo']}"
        root = pub / e["repo"]
        stage_model(cfg, m, e, summary, root)
        api.create_repo(repo, private=private, exist_ok=True)
        api.upload_folder(folder_path=str(root), repo_id=repo, commit_message=f"Upload {e['repo']}")
        note = (
            f"Base {e['id']}. Trap accuracy {e['base']['trap'] * 100:.1f} to {e['tuned']['trap']['mean'] * 100:.1f}, "
            f"NanoBEIR nDCG@10 {e['base']['ndcg'] * 100:.1f} to {e['tuned']['ndcg']['mean'] * 100:.1f}"
        )
        items.append((repo, "model", note))
        log(f"published {repo}")

    title = proj["collection_title"] + (" smoke" if run["repo_suffix"] else "")
    coll = api.create_collection(
        title=title, namespace=user, description=proj["collection_description"], private=private, exists_ok=True
    )
    for item_id, kind, note in items:
        api.add_collection_item(coll.slug, item_id=item_id, item_type=kind, note=note[:500], exists_ok=True)
    write_json(res / "publish.json", {"collection": coll.slug, "items": [i[0] for i in items]})
    log(f"collection https://huggingface.co/collections/{coll.slug}")


if __name__ == "__main__":
    main()
