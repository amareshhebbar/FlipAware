import shutil
import sys

from flipaware.common import best_seed, load_cfg, log, prompts, read_jsonl, run_dir, set_seq, trap_eval, write_json


def main():
    cfg = load_cfg()
    m = sys.argv[1]
    res = cfg["paths"]["results"]
    out = res / f"onnx_{m}.json"
    if out.exists():
        return
    mcfg = cfg["models"][m]
    best = best_seed(cfg, m)
    if best is None:
        return
    from sentence_transformers import SentenceTransformer, export_dynamic_quantized_onnx_model

    path = run_dir(cfg, m, cfg["headline_variant"], best) / "model"
    rc = bool(mcfg.get("trust_remote_code"))
    model = SentenceTransformer(str(path), backend="onnx", device="cpu", trust_remote_code=rc)
    model.save_pretrained(str(path))
    export_dynamic_quantized_onnx_model(model, "avx512_vnni", str(path))
    onnx_dir = path / "onnx"
    q = sorted(onnx_dir.glob("model_qint8*.onnx"))
    if q:
        shutil.copy(q[0], onnx_dir / "model_quantized.onnx")
    test = read_jsonl(cfg["paths"]["data"] / "test.jsonl")
    _, _, s = prompts(mcfg)
    r = {"files": sorted(p.name for p in onnx_dir.glob("*.onnx"))}
    for label, kw in (("onnx_fp32", {}), ("onnx_int8", {"file_name": f"onnx/{q[0].name}"} if q else None)):
        if kw is None:
            continue
        mm = SentenceTransformer(str(path), backend="onnx", device="cpu", trust_remote_code=rc, model_kwargs=kw)
        set_seq(mm, cfg["run"]["eval_seq_len"])
        t = trap_eval(mm, test, s)
        r[label] = t["dims"][str(t["full"])]["accuracy"]
    write_json(out, r)
    log(f"onnx {m}: {r}")


if __name__ == "__main__":
    main()
