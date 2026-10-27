import argparse
import json
import random
import re
import shutil

from flipaware.common import append_jsonl, edit_check, file_sig, gpu_info, load_cfg, log, read_jsonl, stamp_ok, text_sig, write_stamp

SYSTEM = "You create test data for embedding models. You always reply with a single JSON object and nothing else."

TEMPLATE = """Trap type: {name}
Rule: {rule}

Example
Source: {ex_anchor}
{{"paraphrase": "{ex_para}", "trap": "{ex_trap}"}}

Now the real task.
Source: {seed}

paraphrase: the same meaning as the source, with a few words changed. Every fact stays true.
trap: copy the source and change only one to three words, following the rule, so the meaning becomes different.

If the rule cannot apply to this source, reply {{"skip": true}}
Otherwise reply with the JSON object only, in the same form as the example."""

PLACEHOLDER = re.compile(r"^\W*$|\.\.\.|<[^>]*>|your (paraphrase|trap)", re.I)
OBJ = re.compile(r"\{[^{}]*\}", re.S)


def parse(text):
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    skip = False
    for m in reversed(OBJ.findall(text)):
        try:
            obj = json.loads(m)
        except ValueError:
            continue
        if not isinstance(obj, dict):
            continue
        if obj.get("skip") is True:
            skip = True
            continue
        p, t = obj.get("paraphrase"), obj.get("trap")
        if isinstance(p, str) and isinstance(t, str):
            p, t = " ".join(p.split()), " ".join(t.split())
            if p and t and not PLACEHOLDER.search(p) and not PLACEHOLDER.search(t):
                return {"paraphrase": p, "trap": t}
    return {"skip": True} if skip else None


def pick(pool, ty, k, rng):
    if ty == "numbers":
        num = [s for s in pool if s["has_number"]]
        rest = [s for s in pool if not s["has_number"]]
        rng.shuffle(num)
        rng.shuffle(rest)
        return (num + rest)[:k]
    return rng.sample(pool, min(k, len(pool)))


def build_jobs(cfg, which):
    run, data = cfg["run"], cfg["paths"]["data"]
    seeds = read_jsonl(data / "seeds.jsonl")
    if which == "xgen":
        test_ids = {r["seed_id"] for r in read_jsonl(data / "test.jsonl")}
        pool = [s for s in seeds if s["seed_id"] in test_ids]
        random.Random(11).shuffle(pool)
        pool = pool[: run["xgen_seeds_max"]]
        k, rng = len(pool), random.Random(11)
    else:
        pool, k, rng = seeds, run["seeds_per_type"], random.Random(7)
    jobs = []
    for ty, rule in cfg["trap_types"].items():
        for s in pick(pool, ty, k, rng):
            jobs.append({"id": f"{ty}:{s['seed_id']}", "type": ty, "seed_id": s["seed_id"], "anchor": s["text"], "rule": rule})
    return jobs


def messages(cfg, job):
    ex = cfg["trap_examples"][job["type"]]
    q = lambda s: s.replace('"', "'")
    return [
        {"role": "system", "content": SYSTEM},
        {
            "role": "user",
            "content": TEMPLATE.format(
                name=job["type"].replace("_", " "),
                rule=job["rule"],
                ex_anchor=ex[0],
                ex_para=q(ex[1]),
                ex_trap=q(ex[2]),
                seed=job["anchor"],
            ),
        },
    ]


def pick_backend(run):
    want = run.get("gen_backend", "auto")
    if want in ("vllm", "auto"):
        try:
            import vllm

            if gpu_info()["cc_major"] >= 8:
                return "vllm"
        except ImportError:
            pass
        if want == "vllm":
            log("vllm unavailable on this machine, using transformers")
    return "transformers"


class HFGen:
    def __init__(self, name):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(name, padding_side="left")
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        dtype = torch.bfloat16 if gpu_info()["bf16"] else torch.float16
        self.model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=dtype, device_map="cuda").eval()

    def __call__(self, texts, max_new, temp, n):
        enc = self.tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(self.model.device)
        with self.torch.no_grad():
            out = self.model.generate(
                **enc,
                max_new_tokens=max_new,
                do_sample=True,
                temperature=temp,
                top_p=0.9,
                num_return_sequences=n,
                pad_token_id=self.tok.pad_token_id,
            )
        dec = self.tok.batch_decode(out[:, enc["input_ids"].shape[1] :], skip_special_tokens=True)
        return [dec[i * n : (i + 1) * n] for i in range(len(texts))]


class VGen:
    def __init__(self, name):
        from vllm import LLM, SamplingParams

        self.sp = SamplingParams
        self.llm = LLM(model=name, dtype="bfloat16", max_model_len=2048, gpu_memory_utilization=0.88, seed=13)
        self.tok = self.llm.get_tokenizer()

    def __call__(self, texts, max_new, temp, n):
        outs = self.llm.generate(texts, self.sp(temperature=temp, top_p=0.9, max_tokens=max_new, n=n), use_tqdm=False)
        return [[c.text for c in o.outputs] for o in outs]


def chat_text(tok, msgs):
    try:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def choose(anchor, outs, flt):
    first = None
    skips = 0
    for i, o in enumerate(outs, 1):
        r = parse(o)
        if r is None:
            continue
        if r.get("skip"):
            skips += 1
            continue
        if first is None:
            first = (r, i)
        if edit_check(anchor, r["paraphrase"], r["trap"], flt) is None:
            return {**r, "attempts": i}
    if first:
        return {**first[0], "attempts": len(outs)}
    if skips:
        return {"skip": True, "attempts": len(outs)}
    return {"error": True, "attempts": len(outs), "sample": (outs[0] if outs else "")[:300]}


def generate(cfg, which):
    run, data = cfg["run"], cfg["paths"]["data"]
    raw = data / ("raw.jsonl" if which == "main" else "raw_xgen.jsonl")
    name = run["generator"] if which == "main" else run["xgen_generator"]
    n = int(run.get("gen_samples", 3))
    sig = text_sig(name, SYSTEM, TEMPLATE, cfg["trap_types"], cfg["trap_examples"], run["seeds_per_type"], n, file_sig(data / "seeds.jsonl"), file_sig(data / "test.jsonl") if which == "xgen" else None)
    stamp = f"gen_{which}"
    if raw.exists() and not stamp_ok(cfg, stamp, sig):
        shutil.move(str(raw), str(raw) + ".old")
        log(f"generator settings changed, old generations moved to {raw.name}.old and generation restarts")
    write_stamp(cfg, stamp, sig)
    done = {r["id"] for r in read_jsonl(raw)}
    jobs = [j for j in build_jobs(cfg, which) if j["id"] not in done]
    if not jobs:
        log(f"generation complete ({which})")
        return
    backend = pick_backend(run)
    log(f"generating {len(jobs)} items with {name} on {backend}, {n} samples each ({which})")
    gen = VGen(name) if backend == "vllm" else HFGen(name)
    chunk = run["gen_batch"]
    total_ok = total = 0
    for i in range(0, len(jobs), chunk):
        part = jobs[i : i + chunk]
        outs = gen([chat_text(gen.tok, messages(cfg, j)) for j in part], 160, 0.8, n)
        recs = []
        for j, o in zip(part, outs):
            r = choose(j["anchor"], o, cfg["filter"])
            recs.append({"id": j["id"], "type": j["type"], "seed_id": j["seed_id"], "anchor": j["anchor"], "generator": name, **r})
        append_jsonl(raw, recs)
        ok = sum("paraphrase" in r and edit_check(r["anchor"], r["paraphrase"], r["trap"], cfg["filter"]) is None for r in recs)
        total_ok += ok
        total += len(recs)
        log(f"{min(i + chunk, len(jobs))}/{len(jobs)} done, {ok}/{len(recs)} pass the edit check in this chunk ({total_ok}/{total} so far)")
        if i == 0 and len(recs) >= 8 and ok == 0:
            bad = recs[0]
            log(f"warning: no usable output in the first chunk, example: {json.dumps(bad)[:400]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="main", choices=["main", "xgen"])
    generate(load_cfg(), ap.parse_args().set)


if __name__ == "__main__":
    main()
