import dataclasses
import os
import random
import shutil
import sys
import time

import torch
from sentence_transformers.evaluation import SentenceEvaluator

from flipaware import tracking
from flipaware.common import (
    emb_dim,
    encode,
    gpu_info,
    load_cfg,
    load_st,
    log,
    mrl_dims,
    prompts,
    read_jsonl,
    run_dir,
    seed_all,
    set_seq,
    train_precision,
    trap_eval,
    write_json,
)


class TrapEval(SentenceEvaluator):
    def __init__(self, rows, prompt):
        super().__init__()
        self.rows = rows
        self.prompt = prompt
        self.name = "trapdev"
        self.primary_metric = "accuracy"
        self.greater_is_better = True

    def __call__(self, model, output_path=None, epoch=-1, steps=-1):
        r = trap_eval(model, self.rows, self.prompt)
        top = r["dims"][str(r["full"])]
        metrics = self.prefix_name_to_metrics({"accuracy": top["accuracy"], "margin": top["margin"]}, self.name)
        try:
            self.store_metrics_in_model_card_data(model, metrics, epoch, steps)
        except Exception:
            pass
        return metrics


class Weighted(torch.nn.Module):
    def __init__(self, inner, weight):
        super().__init__()
        self.inner = inner
        self.weight = weight
        self.model = inner.model

    def forward(self, features, labels):
        return self.inner(features, labels) * self.weight

    def get_config_dict(self):
        return {"weight": self.weight}


def clean_prompts(pr):
    out = {}
    for ds, cols in pr.items():
        c = {k: v for k, v in cols.items() if v}
        if c:
            out[ds] = c
    return out or None


def latest_ckpt(d):
    if not d.exists():
        return None
    cs = sorted(d.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]))
    return cs[-1] if cs else None


def lora_attr(model):
    mod = model[0]
    return mod, ("auto_model" if hasattr(mod, "auto_model") else "model")


def run_one(cfg, mkey, vname, seed):
    from datasets import Dataset
    from sentence_transformers import SentenceTransformerTrainer, SentenceTransformerTrainingArguments, losses
    from sentence_transformers.training_args import BatchSamplers, MultiDatasetBatchSamplers

    run, tr = cfg["run"], cfg["train"]
    mcfg, var = cfg["models"][mkey], cfg["variants"][vname]
    data = cfg["paths"]["data"]
    out = run_dir(cfg, mkey, vname, seed)
    out.mkdir(parents=True, exist_ok=True)
    seed_all(seed)
    prec = train_precision(mcfg)
    log(f"train {mkey} {vname} seed {seed} precision {prec}")

    model = load_st(mcfg)
    set_seq(model, run["max_seq_len"])
    q, d, s = prompts(mcfg)
    dims = mrl_dims(cfg, emb_dim(model))

    train_rows = read_jsonl(data / "train.jsonl")
    dev_rows = read_jsonl(data / "dev.jsonl")[: run["dev_max"]]
    ret_rows = read_jsonl(data / "retrieval.jsonl")
    mined_info = None
    if var.get("limit"):
        order = list(range(len(train_rows)))
        random.Random(1234).shuffle(order)
        train_rows = [train_rows[i] for i in order[: var["limit"]]]
        log(f"training on {len(train_rows)} trap triples")
    if var.get("mined"):
        model.eval()
        r = trap_eval(model, train_rows, s)
        margins = r["dims"][str(r["full"])]["margins"]
        hard = [x for x, mg in zip(train_rows, margins) if mg < tr["mine_margin"]]
        easy = [x for x, mg in zip(train_rows, margins) if mg >= tr["mine_margin"]]
        rng = random.Random(seed)
        easy = rng.sample(easy, int(len(easy) * tr["mine_easy_frac"]))
        if len(hard) + len(easy) >= run["batch_size"]:
            train_rows = hard + easy
        rng.shuffle(train_rows)
        mined_info = {"hard": len(hard), "easy_kept": len(easy)}
        log(f"mined: {len(hard)} hard triples, {len(easy)} easy kept")

    sets = {
        "trap": Dataset.from_dict(
            {
                "anchor": [r["anchor"] for r in train_rows],
                "positive": [r["paraphrase"] for r in train_rows],
                "negative": [r["trap"] for r in train_rows],
            }
        ),
        "retrieval": Dataset.from_dict(
            {"anchor": [r["anchor"] for r in ret_rows], "positive": [r["positive"] for r in ret_rows]}
        ),
    }

    def mnrl():
        inner = losses.CachedMultipleNegativesRankingLoss(model, mini_batch_size=run["mini_batch"])
        return losses.MatryoshkaLoss(model, inner, matryoshka_dims=dims)

    loss = {"trap": mnrl(), "retrieval": mnrl()}
    pr = {"trap": {"anchor": s, "positive": s, "negative": s}, "retrieval": {"anchor": q, "positive": d}}

    if var["guard"]:
        g = read_jsonl(data / "guard_src.jsonl")
        model.eval()
        e1 = encode(model, [x["s1"] for x in g], s)
        e2 = encode(model, [x["s2"] for x in g], s)
        labels = torch.nn.functional.cosine_similarity(e1, e2).cpu().tolist()
        sets["guard"] = Dataset.from_dict(
            {"sentence1": [x["s1"] for x in g], "sentence2": [x["s2"] for x in g], "score": labels}
        )
        gl = losses.CosineSimilarityLoss(model)
        loss["guard"] = gl if tr["guard_weight"] == 1 else Weighted(gl, tr["guard_weight"])
        pr["guard"] = {"sentence1": s, "sentence2": s}

    if var["lora"]:
        from peft import LoraConfig, get_peft_model

        mod, attr = lora_attr(model)
        pcfg = LoraConfig(
            r=tr["lora_r"],
            lora_alpha=tr["lora_alpha"],
            lora_dropout=tr["lora_dropout"],
            target_modules="all-linear",
            bias="none",
        )
        peft_model = get_peft_model(getattr(mod, attr), pcfg)
        try:
            setattr(mod, attr, peft_model)
        except AttributeError:
            pass
        if run.get("gradient_checkpointing"):
            peft_model.enable_input_require_grads()
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        log(f"LoRA trainable parameters: {trainable}")

    bs = run["batch_size"]
    steps = max(1, sum(len(ds) // bs for ds in sets.values()) * run["epochs"])
    lr = tr["lr_lora"] if var["lora"] else mcfg.get("lr", tr["lr_full"])
    os.environ["TENSORBOARD_LOGGING_DIR"] = str(out / "tb")
    tags = {"model": mkey, "base_id": mcfg["id"], "variant": vname, "seed": str(seed), "lora": str(var["lora"]), "guard": str(var["guard"]), "profile": cfg["profile"]}
    report_to, callbacks = tracking.setup_train(cfg, out.name, tags)
    log(f"tracking: {', '.join(report_to + (['aim'] if callbacks else []))}")
    fields ={f.name for f in dataclasses.fields(SentenceTransformerTrainingArguments)}
    kw = dict(
        output_dir=str(out / "ckpt"),
        num_train_epochs=run["epochs"],
        per_device_train_batch_size=bs,
        learning_rate=lr,
        warmup_steps=max(1, int(steps * tr["warmup_ratio"])),
        weight_decay=tr["weight_decay"],
        bf16=prec == "bf16",
        fp16=prec == "fp16",
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        multi_dataset_batch_sampler=MultiDatasetBatchSamplers.PROPORTIONAL,
        eval_strategy="steps",
        eval_steps=run["eval_steps"],
        save_strategy="no" if var["lora"] else "steps",
        save_steps=run["eval_steps"],
        save_total_limit=1,
        logging_steps=max(5, run["eval_steps"] // 4),
        logging_dir=str(out / "tb"),
        report_to=report_to,
        seed=seed,
        data_seed=seed,
        prompts=clean_prompts(pr),
        dataloader_num_workers=2,
        run_name=out.name,
        disable_tqdm=cfg["platform"] == "runpod",
        gradient_checkpointing=bool(run.get("gradient_checkpointing")),
    )
    if "eval_strategy" not in fields:
        kw["evaluation_strategy"] = kw.pop("eval_strategy")
    args = SentenceTransformerTrainingArguments(**{k: v for k, v in kw.items() if k in fields})

    t0 = time.time()
    trainer = SentenceTransformerTrainer(
        model=model, args=args, train_dataset=sets, loss=loss, evaluator=TrapEval(dev_rows, s), callbacks=callbacks or None
    )
    last = None if var["lora"] else latest_ckpt(out / "ckpt")
    if last:
        log(f"resuming from {last.name}")
    trainer.train(resume_from_checkpoint=str(last) if last else None)

    if var["lora"]:
        merged = peft_model.merge_and_unload()
        try:
            setattr(mod, attr, merged)
        except AttributeError:
            pass
    model.save(str(out / "model"))
    write_json(out / "train_log.json", trainer.state.log_history)
    tracking.finish_train(cfg, out.name, tags, trainer.state.log_history, callbacks)
    write_json(
        out / "done.json",
        {
            "model": mkey,
            "base_id": mcfg["id"],
            "variant": vname,
            "seed": seed,
            "precision": prec,
            "seconds": round(time.time() - t0, 1),
            "gpu": gpu_info().get("name"),
            "steps": trainer.state.global_step,
            "dims": dims,
            "lr": lr,
            "batch_size": bs,
            "mini_batch": run["mini_batch"],
            "epochs": run["epochs"],
            "max_seq_len": run["max_seq_len"],
            "lora": var["lora"],
            "guard": var["guard"],
            "rows": {k: len(v) for k, v in sets.items()},
            "limit": var.get("limit"),
            "mined": mined_info,
        },
    )
    shutil.rmtree(out / "ckpt", ignore_errors=True)
    log(f"saved {out / 'model'}")


def main():
    cfg = load_cfg()
    run_one(cfg, sys.argv[1], sys.argv[2], int(sys.argv[3]))


if __name__ == "__main__":
    main()
