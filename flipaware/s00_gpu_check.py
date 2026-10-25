import importlib
import os
import sys
import time

import torch

from flipaware.common import gpu_info, hf_token, load_cfg, log, write_json


def versions():
    out = {"python": sys.version.split()[0], "torch": torch.__version__, "cuda_runtime": torch.version.cuda}
    for pkg in ("transformers", "sentence_transformers", "datasets", "peft", "huggingface_hub", "accelerate"):
        try:
            out[pkg] = importlib.import_module(pkg).__version__
        except Exception as ex:
            out[pkg] = f"missing: {ex}"
    return out


def gpu_smoke(info):
    dtype = torch.bfloat16 if info["bf16"] else torch.float16
    x = torch.randn(2048, 2048, device="cuda", dtype=dtype, requires_grad=True)
    w = torch.randn(2048, 2048, device="cuda", dtype=dtype, requires_grad=True)
    torch.cuda.synchronize()
    t0 = time.time()
    y = (x @ w).float().pow(2).mean()
    y.backward()
    torch.cuda.synchronize()
    ok = bool(torch.isfinite(x.grad).all().item() and torch.isfinite(w.grad).all().item())
    free, total = torch.cuda.mem_get_info()
    return {"matmul_ok": ok, "matmul_ms": round((time.time() - t0) * 1000, 2), "free_gb": round(free / 1024**3, 1)}


def hf_checks(cfg):
    out = {}
    token = hf_token()
    if not token:
        out["hf_token"] = False
        return out
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    try:
        out["hf_user"] = api.whoami()["name"]
        out["hf_token"] = True
    except Exception as ex:
        out["hf_token"] = False
        out["hf_error"] = repr(ex)
        return out
    gated = {}
    for m in cfg["run"]["models"]:
        mcfg = cfg["models"][m]
        if not mcfg.get("gated"):
            continue
        try:
            api.auth_check(mcfg["id"])
            gated[mcfg["id"]] = "ok"
        except Exception as ex:
            gated[mcfg["id"]] = f"no access: {type(ex).__name__}"
    out["gated"] = gated
    return out


def main():
    cfg = load_cfg()
    info = gpu_info()
    info.update(platform=cfg["platform"], profile=cfg["profile"], work=str(cfg["paths"]["work"]), **versions())
    if not info["cuda"]:
        log("no CUDA GPU found")
        if os.environ.get("FLIP_ALLOW_CPU") != "1":
            sys.exit(1)
    else:
        info.update(gpu_smoke(info))
        if not info["matmul_ok"]:
            log("GPU matmul produced non finite values")
            sys.exit(1)
    info.update(hf_checks(cfg))
    write_json(cfg["paths"]["work"] / "env.json", info)
    for k, v in info.items():
        log(f"{k}: {v}")
    bad = [k for k, v in (info.get("gated") or {}).items() if v != "ok"]
    if bad:
        log("gated models without access:", ", ".join(bad))
        sys.exit(1)
    if not info.get("hf_token"):
        log("no HuggingFace login found; gated downloads and publishing will fail")
        if os.environ.get("FLIP_PUBLISH", "1") == "1":
            sys.exit(1)
    log(f"models this run: {', '.join(cfg['run']['models'])}")
    log("gpu check passed")


if __name__ == "__main__":
    main()