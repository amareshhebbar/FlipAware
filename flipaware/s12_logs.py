import os
import re
import shutil
import sqlite3

from flipaware import tracking
from flipaware.common import REPO, load_cfg, log, read_json, space_name, write_json
from flipaware.publish import add_item, api, ensure_collection, set_private, trim

SECRET = re.compile(r"(hf_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9]{20,}|CLEARML_API_SECRET_KEY\S*)")
TEXT_MAX = 20 * 1024 * 1024


def redact(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(SECRET.sub("[redacted]", src.read_text(errors="replace")))


def runs_repo(cfg):
    return f"{cfg['project']['hf_user']}/{cfg['project']['name']}_Runs{cfg['run']['repo_suffix']}"


def mlflow_snapshot(src, dst):
    con = sqlite3.connect(str(src))
    out = sqlite3.connect(str(dst))
    with out:
        con.backup(out)
    con.close()
    out.close()


def build_export(cfg):
    work, res, runs = cfg["paths"]["work"], cfg["paths"]["results"], cfg["paths"]["runs"]
    t = tracking.tdir(cfg)
    root = cfg["paths"]["publish"] / "runs_export"
    shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True)
    have = []
    logs = work / "logs"
    if logs.exists():
        for p in sorted(logs.glob("*.log")):
            redact(p, root / "logs" / p.name)
        have.append("logs")
    for name in ("metrics.jsonl",):
        if (t / name).exists():
            shutil.copy(t / name, root / name)
            have.append(name)
    for p in sorted(res.glob("*.json")):
        try:
            write_json(root / "results" / p.name, trim(read_json(p)))
        except Exception as ex:
            log(f"skip {p.name}: {ex!r}")
    for p in ("summary.md", "leaderboard.csv", "REPORT.md", "handcheck_done.csv"):
        if (res / p).exists():
            shutil.copy(res / p, root / "results" / p)
    if (work / "env.json").exists():
        shutil.copy(work / "env.json", root / "env.json")
    tb = root / "tensorboard"
    for rd in sorted(runs.glob("*")):
        if (rd / "tb").exists():
            shutil.copytree(rd / "tb", tb / "train" / rd.name, dirs_exist_ok=True)
    if (t / "tb_eval").exists():
        shutil.copytree(t / "tb_eval", tb / "eval", dirs_exist_ok=True)
    if tb.exists():
        have.append("tensorboard")
    if (t / "mlflow.db").exists():
        (root / "mlflow").mkdir()
        mlflow_snapshot(t / "mlflow.db", root / "mlflow" / "mlflow.db")
        have.append("mlflow")
    if (t / "aim").exists() and any((t / "aim").iterdir()):
        shutil.copytree(t / "aim", root / "aim", dirs_exist_ok=True)
        have.append("aim")
    cm = t / "clearml_cache"
    zips = sorted(cm.rglob("*.zip")) if cm.exists() else []
    if zips:
        (root / "clearml").mkdir()
        for z in zips:
            shutil.copy(z, root / "clearml" / z.name)
        have.append("clearml")
    (root / "README.md").write_text(runs_readme(cfg, have))
    write_json(root / "index.json", {"profile": cfg["profile"], "contents": have, "trackers": tracking.enabled(cfg)})
    return root, have


def runs_readme(cfg, have):
    p = cfg["project"]["name"]
    u = cfg["project"]["hf_user"]
    sfx = cfg["run"]["repo_suffix"]
    lines = [
        "---",
        "license: apache-2.0",
        "tags:",
        "- training-logs",
        "- tensorboard",
        "- mlflow",
        "---",
        "",
        f"# {p} training and evaluation logs",
        "",
        f"Every log, metric, and result from the {p} pipeline ({cfg['profile']} profile). Code: https://github.com/{cfg['project']['github']}",
        "",
        "| Folder | What it holds | Open it |",
        "|---|---|---|",
    ]
    rows = {
        "logs": ("Full console output of every pipeline stage, secrets redacted", "any text viewer"),
        "metrics.jsonl": ("Every training step and evaluation result, one JSON line each", "`pandas.read_json(path, lines=True)`"),
        "tensorboard": ("Training curves and evaluation scalars", f"[TensorBoard Space](https://huggingface.co/spaces/{u}/{p}_TensorBoard{sfx}) or `tensorboard --logdir tensorboard`"),
        "mlflow": ("MLflow experiment database with every run, parameter, and metric", f"[MLflow Space](https://huggingface.co/spaces/{u}/{p}_MLflow{sfx}) or `mlflow ui --backend-store-uri sqlite:///mlflow/mlflow.db`"),
        "aim": ("Aim repository", f"[Aim Space](https://huggingface.co/spaces/{u}/{p}_Aim{sfx}) or `aim up --repo aim`"),
        "clearml": ("ClearML offline sessions", "`python -c \"from clearml import Task; Task.import_offline_session('clearml/<file>.zip')\"`"),
    }
    for k in have:
        if k in rows:
            lines.append(f"| `{k}` | {rows[k][0]} | {rows[k][1]} |")
    lines += ["| `results` | Raw evaluation JSON behind every number in the model cards, plus the summary and leaderboard | any JSON viewer |", ""]
    return "\n".join(lines)


SPACE_README = "---\ntitle: {title}\ncolorFrom: gray\ncolorTo: blue\nsdk: docker\napp_port: 7860\npinned: false\n---\n\n{body}\n"

DOCKER = {
    "TensorBoard": (
        "tensorboard",
        "FROM python:3.11-slim\nRUN pip install --no-cache-dir tensorboard=={ver}\nRUN useradd -m -u 1000 user\nUSER user\nWORKDIR /home/user/app\nCOPY --chown=user . .\n"
        'CMD ["tensorboard", "--logdir", "tensorboard", "--host", "0.0.0.0", "--port", "7860", "--reload_interval", "0"]\n',
    ),
    "MLflow": (
        "mlflow",
        "FROM python:3.11-slim\nRUN pip install --no-cache-dir mlflow=={ver}\nRUN useradd -m -u 1000 user\nUSER user\nWORKDIR /home/user/app\nCOPY --chown=user . .\n"
        'CMD ["mlflow", "server", "--backend-store-uri", "sqlite:///mlflow/mlflow.db", "--host", "0.0.0.0", "--port", "7860", "--workers", "1", "--allowed-hosts", "*", "--x-frame-options", "NONE"]\n',
    ),
    "Aim": (
        "aim",
        "FROM python:3.11-slim\nRUN pip install --no-cache-dir aim=={ver}\nRUN useradd -m -u 1000 user\nUSER user\nWORKDIR /home/user/app\nCOPY --chown=user . .\n"
        'CMD ["aim", "up", "--repo", "aim", "--host", "0.0.0.0", "--port", "7860"]\n',
    ),
}


def version(pkg):
    try:
        from importlib.metadata import version as v

        return v(pkg)
    except Exception:
        return None


def publish_dashboards(cfg, hf, export, have, slug):
    out = []
    private = bool(cfg["run"]["private"])
    for title, (folder, tpl) in DOCKER.items():
        key = folder
        if key not in have:
            continue
        ver = version(folder)
        if not ver:
            continue
        name = space_name(cfg, title)
        repo = f"{cfg['project']['hf_user']}/{name}"
        root = cfg["paths"]["publish"] / f"space_{folder}"
        shutil.rmtree(root, ignore_errors=True)
        root.mkdir(parents=True)
        shutil.copytree(export / folder, root / folder)
        (root / "Dockerfile").write_text(tpl.format(ver=ver))
        body = f"Live {title} view of the {cfg['project']['name']} training and evaluation runs. Raw files: https://huggingface.co/datasets/{runs_repo(cfg)}"
        (root / "README.md").write_text(SPACE_README.format(title=f"{cfg['project']['name']} {title}", body=body))
        try:
            hf.create_repo(repo, repo_type="space", space_sdk="docker", private=private, exist_ok=True)
            hf.upload_folder(folder_path=str(root), repo_id=repo, repo_type="space", commit_message=f"Update {title} dashboard")
            set_private(hf, repo, private, kind="space")
            if slug:
                add_item(hf, slug, repo, "space", f"Live {title} dashboard for every run")
            log(f"{title} dashboard https://huggingface.co/spaces/{repo}")
            out.append(repo)
        except Exception as ex:
            log(f"{title} dashboard failed: {ex!r}")
    return out


def copy_to_github(cfg, export):
    dst = REPO / "reports" / cfg["profile"]
    shutil.rmtree(dst, ignore_errors=True)
    dst.mkdir(parents=True)
    n = 0
    for p in export.rglob("*"):
        if not p.is_file():
            continue
        rel = p.relative_to(export)
        if rel.parts[0] in ("tensorboard", "aim", "clearml", "mlflow") or p.stat().st_size > TEXT_MAX:
            continue
        (dst / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(p, dst / rel)
        n += 1
    for img in ("headline.png", "leaderboard.png", "dilution.png", "benchmarks.png", "scaling.png", "blend.png", "retrieval.png"):
        src = cfg["paths"]["assets"] / img
        if src.exists():
            (dst / "assets").mkdir(exist_ok=True)
            shutil.copy(src, dst / "assets" / img)
            n += 1
    log(f"copied {n} files into {dst} for GitHub")
    return dst


def main():
    cfg = load_cfg()
    export, have = build_export(cfg)
    log(f"export ready at {export}: {', '.join(have) or 'nothing'}")
    if os.environ.get("FLIP_GITHUB_REPORTS", "1") == "1":
        copy_to_github(cfg, export)
    hf = api()
    if not hf:
        return
    slug = ensure_collection(cfg, hf)
    repo = runs_repo(cfg)
    hf.create_repo(repo, repo_type="dataset", private=bool(cfg["run"]["private"]), exist_ok=True)
    hf.upload_folder(folder_path=str(export), repo_id=repo, repo_type="dataset", commit_message="Update training and evaluation logs")
    add_item(hf, slug, repo, "dataset", "Every training log, metric, and evaluation result")
    log(f"logs https://huggingface.co/datasets/{repo}")
    if os.environ.get("FLIP_DASHBOARDS", "1") == "1":
        publish_dashboards(cfg, hf, export, have, slug)


if __name__ == "__main__":
    main()
