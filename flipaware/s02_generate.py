import json
import random
import re

from flipaware.common import append_jsonl, gpu_info, load_cfg, log, read_jsonl

SYSTEM = "You create test data for embedding models. Reply with one JSON object and nothing else."

TEMPLATE = """Trap type: {name}
Rule: {rule}

Source sentence: {seed}

Write two new sentences.
paraphrase: the same meaning as the source. Change only a few words (synonyms, word order, contractions). Every fact must stay true.
trap: keep almost every word of the source but apply the rule so the meaning becomes different. Change as few words as possible.

If the rule cannot apply to this sentence, reply {{"skip": true}}
Otherwise reply {{"paraphrase": "...", "trap": "..."}}"""


def parse(text):
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    for pat in (r"\{.*\}", r"\{.*?\}"):
        m = re.search(pat, text, re.S)
        if not m:
            continue
        try:
            obj = json.loads(m.group(0))
        except ValueError:
            continue
        if not isinstance(obj, dict):
            continue
        if obj.get("skip"):
            return {"skip": True}
        p, t = obj.get("paraphrase"), obj.get("trap")
        if isinstance(p, str) and isinstance(t, str) and p.strip() and t.strip():
            return {"paraphrase": " ".join(p.split()), "trap": " ".join(t.split())}
    return None


def build_jobs(cfg):
    run = cfg["run"]
    seeds = read_jsonl(cfg["paths"]["data"] / "seeds.jsonl")
    rng = random.Random(7)
    k = run["seeds_per_type"]
    jobs = []
    for ty, rule in cfg["trap_types"].items():
        if ty == "numbers":
            num = [s for s in seeds if s["has_number"]]
            rest = [s for s in seeds if not s["has_number"]]
            rng.shuffle(num)
            rng.shuffle(rest)
            picked = (num + rest)[:k]
        else:
            picked = rng.sample(seeds, min(k, len(seeds)))
        for s in picked:
            jobs.append({"id": f"{ty}:{s['seed_id']}", "type": ty, "seed_id": s["seed_id"], "anchor": s["text"], "rule": rule})
    return jobs


def messages(job):
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": TEMPLATE.format(name=job["type"].replace("_", " "), rule=job["rule"], seed=job["anchor"])},
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
        dtype = torch.bfloat16 if gpu_info()["bf16"] else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=dtype, device_map="cuda").eval()

    def __call__(self, texts, max_new, temp):
        enc = self.tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(self.model.device)
        with self.torch.no_grad():
            out = self.model.generate(
                **enc,
                max_new_tokens=max_new,
                do_sample=True,
                temperature=temp,
                top_p=0.9,
                pad_token_id=self.tok.pad_token_id,
            )
        return self.tok.batch_decode(out[:, enc["input_ids"].shape[1] :], skip_special_tokens=True)


class VGen:
    def __init__(self, name):
        from vllm import LLM, SamplingParams

        self.sp = SamplingParams
        self.llm = LLM(model=name, dtype="bfloat16", max_model_len=2048, gpu_memory_utilization=0.88, seed=13)
        self.tok = self.llm.get_tokenizer()

    def __call__(self, texts, max_new, temp):
        outs = self.llm.generate(texts, self.sp(temperature=temp, top_p=0.9, max_tokens=max_new), use_tqdm=False)
        return [o.outputs[0].text for o in outs]


def main():
    cfg = load_cfg()
    run = cfg["run"]
    raw = cfg["paths"]["data"] / "raw.jsonl"
    done = {r["id"] for r in read_jsonl(raw)}
    jobs = [j for j in build_jobs(cfg) if j["id"] not in done]
    if not jobs:
        log("generation complete")
        return
    backend = pick_backend(run)
    log(f"generating {len(jobs)} items with {run['generator']} on {backend}")
    gen = VGen(run["generator"]) if backend == "vllm" else HFGen(run["generator"])
    chunk = run["gen_batch"]
    for i in range(0, len(jobs), chunk):
        part = jobs[i : i + chunk]
        texts = [
            gen.tok.apply_chat_template(messages(j), tokenize=False, add_generation_prompt=True, enable_thinking=False)
            for j in part
        ]
        outs = gen(texts, 200, 0.7)
        recs = []
        for j, o in zip(part, outs):
            parsed = parse(o) or {"error": True}
            recs.append({"id": j["id"], "type": j["type"], "seed_id": j["seed_id"], "anchor": j["anchor"], **parsed})
        append_jsonl(raw, recs)
        ok = sum("paraphrase" in r for r in recs)
        log(f"{min(i + chunk, len(jobs))}/{len(jobs)} done, {ok}/{len(recs)} usable in this chunk")


if __name__ == "__main__":
    main()
