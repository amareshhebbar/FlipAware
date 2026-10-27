import argparse
import json
import os
import random
import shutil
from pathlib import Path

from flipaware.common import REPO, edit_check, hf_token, load_cfg, log, read_jsonl, write_json, write_jsonl


def data_repo(cfg):
    return f"{cfg['project']['hf_user']}/{cfg['project']['name']}_HumanWrites{cfg['run']['repo_suffix']}"


def writer_space(cfg):
    return f"{cfg['project']['hf_user']}/{cfg['project']['name']}_Writer{cfg['run']['repo_suffix']}"


def deploy(cfg):
    from huggingface_hub import HfApi

    token = hf_token()
    if not token:
        raise SystemExit("log in to HuggingFace first")
    api = HfApi(token=token)
    ds, sp = data_repo(cfg), writer_space(cfg)
    api.create_repo(ds, repo_type="dataset", private=True, exist_ok=True)
    api.create_repo(sp, repo_type="space", space_sdk="gradio", private=False, exist_ok=True)
    stage = cfg["paths"]["publish"] / "writer_space"
    shutil.rmtree(stage, ignore_errors=True)
    shutil.copytree(REPO / "spaces" / "writer", stage)
    anchors = read_jsonl(cfg["paths"]["data"] / "human_anchors.jsonl")
    if not anchors:
        raise SystemExit("run stage s01_data first so human_anchors.jsonl exists")
    write_jsonl(stage / "anchors.jsonl", anchors)
    types = {t: {"rule": r, "example": cfg["trap_examples"][t]} for t, r in cfg["trap_types"].items()}
    (stage / "trap_types.json").write_text(json.dumps(types, indent=2))
    (stage / "README.md").write_text(
        "---\ntitle: FlipAware Writer\ncolorFrom: blue\ncolorTo: indigo\nsdk: gradio\napp_file: app.py\npinned: false\n---\n\n"
        "Write and check human test triples for the FlipAware embedding benchmark.\n"
    )
    api.upload_folder(folder_path=str(stage), repo_id=sp, repo_type="space", commit_message="Deploy writer")
    secret = os.environ.get("WRITER_TOKEN") or token
    api.add_space_secret(sp, "WRITE_TOKEN", secret)
    api.add_space_variable(sp, "DATA_REPO", ds)
    log(f"writer space https://huggingface.co/spaces/{sp}")
    log(f"submissions are saved to the private dataset https://huggingface.co/datasets/{ds}")


def pull(cfg):
    token = hf_token()
    data, res = cfg["paths"]["data"], cfg["paths"]["results"]
    if not token:
        log("no HuggingFace login, skip human set")
        return
    from huggingface_hub import snapshot_download

    try:
        d = Path(snapshot_download(data_repo(cfg), repo_type="dataset", token=token, allow_patterns=["data/*.jsonl"])) / "data"
    except Exception as ex:
        log(f"no human writes yet ({type(ex).__name__}); deploy the writer with: python -m flipaware.human deploy")
        return
    subs, checks = {}, []
    for p in d.glob("submissions-*.jsonl"):
        for r in read_jsonl(p):
            subs[r["id"]] = r
    for p in d.glob("checks-*.jsonl"):
        checks += read_jsonl(p)
    votes = {}
    for c in checks:
        s = subs.get(c["sub_id"])
        if not s or c["checker"] == s["author"]:
            continue
        v = votes.setdefault(c["sub_id"], {"yes": 0, "no": 0})
        v["yes" if c["paraphrase_ok"] and c["trap_ok"] else "no"] += 1
    train_anchors = {r["anchor"].lower() for r in read_jsonl(data / "train.jsonl")}
    rows, stats = [], {"submitted": len(subs), "checks": len(checks), "approved": 0, "rejected": 0, "unchecked": 0, "leaked": 0, "low_overlap": 0}
    for i, (sid, s) in enumerate(sorted(subs.items(), key=lambda x: x[1].get("ts", 0))):
        v = votes.get(sid)
        if not v:
            stats["unchecked"] += 1
            continue
        if v["yes"] <= v["no"]:
            stats["rejected"] += 1
            continue
        if s["anchor"].lower() in train_anchors:
            stats["leaked"] += 1
            continue
        if edit_check(s["anchor"], s["paraphrase"], s["trap"], cfg["filter"]):
            stats["low_overlap"] += 1
            continue
        stats["approved"] += 1
        rows.append(
            {
                "id": f"human:{sid[:12]}",
                "type": s["type"],
                "seed_id": f"h{i}",
                "anchor": s["anchor"],
                "paraphrase": s["paraphrase"],
                "trap": s["trap"],
                "author": s["author"],
                "anchor_source": s.get("anchor_source"),
                "yes": v["yes"],
                "no": v["no"],
            }
        )
    random.Random(3).shuffle(rows)
    old = read_jsonl(data / "human_test.jsonl")
    if rows and len(rows) != len(old):
        for p in res.glob("*.json"):
            if p.name.startswith(("base_", "run_")):
                r = json.loads(p.read_text())
                if "trap_human" in r:
                    r.pop("trap_human")
                    write_json(p, r)
    write_jsonl(data / "human_test.jsonl", rows)
    write_json(res / "human_stats.json", stats)
    log(f"human test set: {stats}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("action", nargs="?", default="pull", choices=["pull", "deploy"])
    a = ap.parse_args()
    cfg = load_cfg()
    deploy(cfg) if a.action == "deploy" else pull(cfg)


if __name__ == "__main__":
    main()
