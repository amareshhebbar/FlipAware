import importlib.util
import os

from flipaware.common import done_seeds, load_cfg, log, run_dir, run_module, runs_for, write_json


def has(mod):
    return importlib.util.find_spec(mod) is not None


def main():
    cfg = load_cfg()
    run = cfg["run"]
    res = cfg["paths"]["results"]
    fails = []

    def step(label, *args):
        rc = run_module(*args)
        if rc != 0:
            fails.append(label)
            log(f"failed: {label}")

    log("refreshing evaluations for trained models")
    for m in run["models"]:
        if not done_seeds(cfg, m):
            continue
        step(f"eval base {m}", "flipaware.evaluate", "base", m)
        for v, s in runs_for(cfg, m):
            if (run_dir(cfg, m, v, s) / "done.json").exists():
                step(f"eval {m} {v} s{s}", "flipaware.evaluate", "run", m, v, s)

    lb = [m for m in run.get("leaderboard") or [] if m not in run["models"]]
    log(f"leaderboard: {len(lb)} extra open embedders")
    for m in lb:
        step(f"leaderboard {m}", "flipaware.evaluate", "base", m)

    for m in run.get("blend_models") or []:
        if done_seeds(cfg, m):
            log(f"weight blending {m}")
            step(f"blend {m}", "flipaware.blend", m)

    if os.environ.get("FLIP_ONNX", "1") == "1" and has("optimum"):
        for m in run.get("onnx_models") or []:
            if done_seeds(cfg, m):
                log(f"onnx export {m}")
                step(f"onnx {m}", "flipaware.onnx_export", m)
    else:
        log("onnx export skipped (install with FLIP_ONNX=1 bash setup.sh)")

    if run.get("mteb_tasks") and os.environ.get("FLIP_MTEB", "0") == "1" and has("mteb"):
        for m in run["models"]:
            if done_seeds(cfg, m):
                log(f"mteb {m}")
                step(f"mteb base {m}", "flipaware.mteb_eval", "base", m)
                step(f"mteb tuned {m}", "flipaware.mteb_eval", "run", m)
    else:
        log("mteb skipped (enable with FLIP_MTEB=1)")

    write_json(res / "analysis_failures.json", fails)
    log(f"analysis finished, {len(fails)} failed")


if __name__ == "__main__":
    main()
