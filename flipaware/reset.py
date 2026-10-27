import argparse

from flipaware.common import RESET, load_cfg, reset_from


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", type=int, help=f"delete outputs of this stage and every later one; stages with outputs: {sorted(RESET)}")
    ap.add_argument("--yes", action="store_true")
    a = ap.parse_args()
    cfg = load_cfg()
    if not a.yes:
        print(f"this deletes outputs of stage {a.stage} and later in {cfg['paths']['work']}; add --yes to confirm")
        return
    for p in reset_from(cfg, a.stage):
        print("removed", p)


if __name__ == "__main__":
    main()
