import json
import os
import random
import re
import time
import uuid
from pathlib import Path

import gradio as gr
from huggingface_hub import CommitScheduler, snapshot_download

DATA_REPO = os.environ["DATA_REPO"]
TOKEN = os.environ.get("WRITE_TOKEN") or os.environ.get("HF_TOKEN")
HERE = Path(__file__).parent
BUF = HERE / "buffer"
BUF.mkdir(exist_ok=True)
SID = uuid.uuid4().hex[:8]
SUB = BUF / f"submissions-{SID}.jsonl"
CHK = BUF / f"checks-{SID}.jsonl"
scheduler = CommitScheduler(repo_id=DATA_REPO, repo_type="dataset", folder_path=BUF, path_in_repo="data", every=2, token=TOKEN)

TYPES = json.loads((HERE / "trap_types.json").read_text())
ANCHORS = [json.loads(l)["text"] for l in (HERE / "anchors.jsonl").read_text().splitlines() if l.strip()]
WORD = re.compile(r"[a-z0-9']+")
CACHE = {"t": 0.0, "rows": []}


def overlap(a, b):
    x, y = set(WORD.findall(a.lower())), set(WORD.findall(b.lower()))
    return len(x & y) / max(1, len(x | y))


def write_row(path, row):
    with scheduler.lock:
        with path.open("a") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_rows(prefix):
    rows = []
    if time.time() - CACHE["t"] > 60:
        try:
            d = snapshot_download(DATA_REPO, repo_type="dataset", token=TOKEN, allow_patterns=["data/*.jsonl"])
            CACHE["dir"] = Path(d) / "data"
        except Exception:
            CACHE["dir"] = None
        CACHE["t"] = time.time()
    dirs = [BUF] + ([CACHE["dir"]] if CACHE.get("dir") else [])
    seen = set()
    for d in dirs:
        for p in d.glob(f"{prefix}-*.jsonl"):
            for line in p.read_text().splitlines():
                if not line.strip():
                    continue
                r = json.loads(line)
                if r["id"] not in seen:
                    seen.add(r["id"])
                    rows.append(r)
    return rows


def rule_text(ty):
    t = TYPES[ty]
    a, p, x = t["example"]
    return f"**Rule:** {t['rule']}\n\n**Example**\n\nAnchor: {a}\n\nParaphrase (same meaning): {p}\n\nTrap (looks alike, different meaning): {x}"


def new_anchor():
    return random.choice(ANCHORS), "pool"


def mark_own(text, src, last):
    return "own" if text.strip() != last.strip() else src


def submit(name, ty, anchor, src, para, trap):
    name, anchor, para, trap = name.strip(), anchor.strip(), para.strip(), trap.strip()
    if not name:
        return "Enter your name first.", para, trap
    if not (anchor and para and trap):
        return "Fill anchor, paraphrase, and trap.", para, trap
    if len({anchor.lower(), para.lower(), trap.lower()}) < 3:
        return "Paraphrase and trap must both differ from the anchor.", para, trap
    if overlap(anchor, trap) < 0.5:
        return "The trap must reuse most of the anchor's words. Change as few words as possible.", para, trap
    row = {"id": uuid.uuid4().hex, "author": name, "type": ty, "anchor": anchor, "anchor_source": src, "paraphrase": para, "trap": trap, "ts": time.time()}
    write_row(SUB, row)
    mine = sum(1 for r in load_rows("submissions") if r["author"] == name)
    return f"Saved. You have written {mine} triples.", "", ""


def next_check(name):
    name = name.strip()
    if not name:
        return None, "Enter your name first.", "", "", "", ""
    subs = load_rows("submissions")
    done = {(c["sub_id"]) for c in load_rows("checks") if c["checker"] == name}
    counts = {}
    for c in load_rows("checks"):
        counts[c["sub_id"]] = counts.get(c["sub_id"], 0) + 1
    pool = [r for r in subs if r["author"] != name and r["id"] not in done]
    if not pool:
        return None, "Nothing left to check right now.", "", "", "", ""
    pool.sort(key=lambda r: counts.get(r["id"], 0))
    r = random.choice(pool[: max(1, len(pool) // 3)])
    return r["id"], rule_text(r["type"]), r["anchor"], r["paraphrase"], r["trap"], r["type"]


def save_check(name, sub_id, para_ok, trap_ok):
    if not sub_id:
        return "Load a triple first."
    if para_ok is None or trap_ok is None:
        return "Answer both questions."
    write_row(CHK, {"id": uuid.uuid4().hex, "sub_id": sub_id, "checker": name.strip(), "paraphrase_ok": para_ok == "Yes", "trap_ok": trap_ok == "Yes", "ts": time.time()})
    return "Saved. Load the next one."


def stats():
    subs = load_rows("submissions")
    checks = load_rows("checks")
    by = {t: 0 for t in TYPES}
    for r in subs:
        by[r["type"]] = by.get(r["type"], 0) + 1
    authors = {}
    for r in subs:
        authors[r["author"]] = authors.get(r["author"], 0) + 1
    lines = [f"**{len(subs)}** triples written, **{len(checks)}** checks", "", "| Trap type | Written |", "|---|---|"]
    lines += [f"| {t.replace('_', ' ')} | {n} |" for t, n in by.items()]
    lines += ["", "| Writer | Triples |", "|---|---|"]
    lines += [f"| {a} | {n} |" for a, n in sorted(authors.items(), key=lambda x: -x[1])]
    return "\n".join(lines)


with gr.Blocks(title="FlipAware Writer") as demo:
    gr.Markdown(
        "# FlipAware Writer\n"
        "Help build a human written test set for embedding models. For each sentence, write a **paraphrase** with the same meaning "
        "and a **trap** that reuses almost every word but means something different."
    )
    name = gr.Textbox(label="Your name", max_lines=1)
    with gr.Tab("Write"):
        first = list(TYPES)[0]
        ty = gr.Dropdown(list(TYPES), value=first, label="Trap type")
        rule = gr.Markdown(rule_text(first))
        a0, s0 = new_anchor()
        anchor = gr.Textbox(value=a0, label="Anchor sentence (edit it or write your own)", lines=2)
        last = gr.State(a0)
        src = gr.State(s0)
        gr.Button("New sentence").click(new_anchor, None, [anchor, src]).then(lambda a: a, anchor, last)
        para = gr.Textbox(label="Paraphrase: same meaning, a few words changed", lines=2)
        trap = gr.Textbox(label="Trap: almost the same words, different meaning", lines=2)
        status = gr.Markdown()
        ty.change(rule_text, ty, rule)
        anchor.change(mark_own, [anchor, src, last], src)
        gr.Button("Submit", variant="primary").click(submit, [name, ty, anchor, src, para, trap], [status, para, trap])
    with gr.Tab("Check"):
        sub_id = gr.State(None)
        crule = gr.Markdown()
        ca = gr.Textbox(label="Anchor", interactive=False)
        cp = gr.Textbox(label="Paraphrase", interactive=False)
        ct = gr.Textbox(label="Trap", interactive=False)
        cty = gr.Textbox(label="Trap type", interactive=False)
        q1 = gr.Radio(["Yes", "No"], label="Does the paraphrase mean exactly the same as the anchor?")
        q2 = gr.Radio(["Yes", "No"], label="Does the trap mean something different from the anchor?")
        cstatus = gr.Markdown()
        gr.Button("Load a triple").click(next_check, name, [sub_id, crule, ca, cp, ct, cty])
        gr.Button("Save check", variant="primary").click(save_check, [name, sub_id, q1, q2], cstatus)
    with gr.Tab("Progress"):
        board = gr.Markdown()
        gr.Button("Refresh").click(stats, None, board)

demo.launch()
