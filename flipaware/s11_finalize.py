from flipaware.common import load_cfg, log, read_json
from flipaware.publish import add_item, api, ensure_collection, publish_dataset, publish_leaderboard, publish_model
from flipaware.report import build_summary, table
from flipaware.writeup import report


def main():
    cfg = load_cfg()
    res = cfg["paths"]["results"]
    summary = build_summary(cfg)
    (res / "REPORT.md").write_text(report(cfg, summary))
    log(f"write up draft: {res / 'REPORT.md'}")
    hf = api()
    if hf:
        slug = ensure_collection(cfg, hf)
        try:
            publish_dataset(cfg, hf, summary)
        except Exception as ex:
            log(f"dataset refresh failed: {ex!r}")
        for m, e in summary["models"].items():
            if not e.get("tuned"):
                continue
            state = res / f"model_{m}.json"
            private = read_json(state).get("private", True) if state.exists() else True
            try:
                publish_model(cfg, hf, m, e, summary, private=private, card_only=True)
            except Exception as ex:
                log(f"card refresh failed for {m}: {ex!r}")
        try:
            sp = publish_leaderboard(cfg, hf, summary["leaderboard"], summary)
            if sp:
                add_item(hf, slug, sp, "space", f"Trap accuracy for {len(summary['leaderboard'])} embedders")
        except Exception as ex:
            log(f"leaderboard publish failed: {ex!r}")
    print("\n" + table(list(summary["models"].values())) + "\n", flush=True)
    good = [e["tail"] for e in summary["models"].values() if e.get("verdict", {}).get("good")]
    bad = [e["tail"] for e in summary["models"].values() if e.get("verdict") and not e["verdict"]["good"]]
    log(f"GOOD: {', '.join(good) or 'none'}")
    log(f"BAD: {', '.join(bad) or 'none'}")
    if (res / "collection.json").exists():
        log(f"collection https://huggingface.co/collections/{read_json(res / 'collection.json')['slug']}")


if __name__ == "__main__":
    main()
