import importlib.util
import json
import os
import time
from pathlib import Path

from flipaware.common import append_jsonl, log

ALL = ("tensorboard", "mlflow", "aim", "clearml")
HF_NATIVE = ("tensorboard", "mlflow", "clearml")
_warned = set()


def enabled(cfg):
    raw = os.environ.get("FLIP_TRACKERS") or ",".join(cfg.get("tracking", {}).get("default", ["tensorboard", "mlflow"]))
    out = []
    for t in (x.strip().lower() for x in raw.split(",") if x.strip()):
        if t not in ALL:
            continue
        if importlib.util.find_spec(t) is None:
            if t not in _warned:
                _warned.add(t)
                log(f"tracker {t} is not installed, skipping it (install with FLIP_TRACKERS={raw} bash setup.sh)")
            continue
        out.append(t)
    return out


def tdir(cfg):
    p = cfg["paths"]["work"] / "tracking"
    p.mkdir(parents=True, exist_ok=True)
    return p


def experiment(cfg):
    return f"{cfg['project']['name']}_{cfg['profile']}"


def mlflow_uri(cfg):
    return os.environ.get("FLIP_MLFLOW_URI") or f"sqlite:///{tdir(cfg) / 'mlflow.db'}"


def clearml_online():
    return bool(os.environ.get("CLEARML_API_ACCESS_KEY")) or Path("~/clearml.conf").expanduser().exists()


def setup_train(cfg, run_name, tags):
    on = enabled(cfg)
    d = tdir(cfg)
    if "mlflow" in on:
        os.environ["MLFLOW_TRACKING_URI"] = mlflow_uri(cfg)
        os.environ["MLFLOW_EXPERIMENT_NAME"] = experiment(cfg)
        os.environ["MLFLOW_RUN_NAME"] = run_name
        os.environ["MLFLOW_TAGS"] = json.dumps({**tags, "stage": "train"})
        os.environ["MLFLOW_FLATTEN_PARAMS"] = "TRUE"
        os.environ.setdefault("HF_MLFLOW_LOG_ARTIFACTS", "FALSE")
    if "clearml" in on:
        os.environ["CLEARML_PROJECT"] = experiment(cfg)
        os.environ["CLEARML_TASK"] = run_name
        os.environ.setdefault("CLEARML_LOG_MODEL", "FALSE")
        if not clearml_online():
            os.environ["CLEARML_CACHE_DIR"] = str(d / "clearml_cache")
            from clearml import Task

            Task.set_offline(True)
    report_to = [t for t in on if t in HF_NATIVE] or ["none"]
    callbacks = []
    if "aim" in on:
        try:
            from aim import Repo
            from aim.hugging_face import AimCallback

            path = d / "aim"
            path.mkdir(exist_ok=True)
            Repo.from_path(str(path), init=True)
            cb = AimCallback(repo=str(path), experiment=experiment(cfg))
            callbacks.append(cb)
        except Exception as ex:
            log(f"aim callback unavailable: {ex!r}")
    return report_to, callbacks


def finish_train(cfg, run_name, tags, history, callbacks):
    rows = [{"ts": time.time(), "stage": "train", "run": run_name, **tags, **{k: v for k, v in h.items() if isinstance(v, (int, float, str))}} for h in history]
    append_jsonl(tdir(cfg) / "metrics.jsonl", rows)
    for cb in callbacks:
        try:
            r = getattr(cb, "experiment", None)
            if r is not None:
                for k, v in tags.items():
                    r[k] = v
                r.close()
        except Exception:
            pass
    if "clearml" in enabled(cfg):
        try:
            from clearml import Task

            t = Task.current_task()
            if t is not None:
                t.close()
        except Exception:
            pass


def flat_metrics(r):
    out = {}

    def top(x):
        return x["dims"][str(x["full"])]

    if r.get("trap"):
        t = top(r["trap"])
        out["trap/accuracy"] = t["accuracy"]
        out["trap/margin"] = t["margin"]
        for ty, v in t["by_type"].items():
            out[f"trap_type/{ty}"] = v
        for dim, rec in r["trap"]["dims"].items():
            out[f"trap_dim/{dim}"] = rec["accuracy"]
        for q, v in (r["trap"].get("quant") or {}).items():
            out[f"trap_quant/{q}"] = v
    for k in ("trap_xgen", "trap_human"):
        if r.get(k):
            out[f"{k}/accuracy"] = top(r[k])["accuracy"]
    if r.get("passage"):
        out["passage/pair_acc"] = r["passage"]["pair_acc"]
        out["passage/recall1"] = r["passage"]["recall1"]
        out["passage/mrr"] = r["passage"]["mrr"]
    if r.get("nevir"):
        out["nevir/pair_acc"] = r["nevir"]["pair_acc"]
    if r.get("nanobeir"):
        out["nanobeir/ndcg10"] = r["nanobeir"]["mean"]
        for k, v in r["nanobeir"].get("detail", {}).items():
            out[f"nanobeir_detail/{k.replace('@', '_at_')}"] = v
    return out


def safe_key(k):
    return "".join(c if c.isalnum() or c in "_/.-" else "_" for c in k)


def log_eval(cfg, kind, m, v, s, r):
    on = enabled(cfg)
    metrics = flat_metrics(r)
    if not metrics:
        return
    name = f"eval_{kind}_{m}" + (f"__{v}__s{s}" if kind == "run" else "")
    tags = {"model": m, "base_id": cfg["models"][m]["id"], "kind": kind, "variant": v or "base", "seed": "" if s is None else str(s), "stage": "eval"}
    d = tdir(cfg)
    append_jsonl(d / "metrics.jsonl", [{"ts": time.time(), "stage": "eval", "run": name, **tags, **metrics}])
    if "tensorboard" in on:
        try:
            from torch.utils.tensorboard import SummaryWriter

            w = SummaryWriter(str(d / "tb_eval" / name))
            for k, val in metrics.items():
                w.add_scalar(k, val, 0)
            w.add_text("tags", json.dumps(tags), 0)
            w.close()
        except Exception as ex:
            log(f"tensorboard eval log failed: {ex!r}")
    if "mlflow" in on:
        try:
            import mlflow

            mlflow.set_tracking_uri(mlflow_uri(cfg))
            mlflow.set_experiment(experiment(cfg))
            with mlflow.start_run(run_name=name, tags=tags):
                mlflow.log_metrics({safe_key(k): float(val) for k, val in metrics.items()})
        except Exception as ex:
            log(f"mlflow eval log failed: {ex!r}")
    if "aim" in on:
        try:
            from aim import Repo, Run

            path = d / "aim"
            path.mkdir(exist_ok=True)
            Repo.from_path(str(path), init=True)
            run = Run(repo=str(path), experiment=experiment(cfg))
            run.name = name
            for k, val in tags.items():
                run[k] = val
            for k, val in metrics.items():
                run.track(float(val), name=safe_key(k), context={"stage": "eval"})
            run.close()
        except Exception as ex:
            log(f"aim eval log failed: {ex!r}")
    if "clearml" in on:
        try:
            from clearml import Task

            if not clearml_online():
                os.environ["CLEARML_CACHE_DIR"] = str(d / "clearml_cache")
                Task.set_offline(True)
            t = Task.init(project_name=experiment(cfg), task_name=name, task_type=Task.TaskTypes.testing, reuse_last_task_id=False, auto_connect_frameworks=False, auto_resource_monitoring=False)
            t.connect(tags, name="tags")
            lg = t.get_logger()
            for k, val in metrics.items():
                lg.report_single_value(safe_key(k), float(val))
            t.close()
        except Exception as ex:
            log(f"clearml eval log failed: {ex!r}")
