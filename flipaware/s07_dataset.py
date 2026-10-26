from flipaware.common import load_cfg, log
from flipaware.publish import add_item, api, ensure_collection, publish_dataset
from flipaware.report import context


def main():
    cfg = load_cfg()
    hf = api()
    if hf is None:
        log("publishing off, skip dataset upload")
        return
    ctx = context(cfg)
    repo, counts = publish_dataset(cfg, hf, ctx)
    slug = ensure_collection(cfg, hf)
    note = f"{sum(counts.values())} anchor, paraphrase, trap triples across {len(cfg['trap_types'])} trap types, plus cross generator, passage, and human test sets"
    add_item(hf, slug, repo, "dataset", note)


if __name__ == "__main__":
    main()
