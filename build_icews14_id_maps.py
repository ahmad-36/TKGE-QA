
from pathlib import Path
import pandas as pd


TRAIN_PATH = Path("icews_2014_train.txt")
VALID_PATH = Path("icews_2014_valid.txt")
TEST_PATH  = Path("icews_2014_test.txt")

OUT_DIR = Path("icews14")

def _read_if_exists(path: Path) -> pd.DataFrame | None:
    if path.exists():
        return pd.read_csv(path, sep="\t", header=None, dtype=str)
    return None

def _write_map(items: list[str], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for idx, item in enumerate(items):
            f.write(f"{item}\t{idx}\n")

def main():
    if not TRAIN_PATH.exists():
        raise FileNotFoundError(f"Missing {TRAIN_PATH}. Put icews_2014_train.txt in the project root.")

    train = _read_if_exists(TRAIN_PATH)
    valid = _read_if_exists(VALID_PATH)
    test  = _read_if_exists(TEST_PATH)

    dfs = [train]
    used = [str(TRAIN_PATH)]
    if valid is not None:
        dfs.append(valid); used.append(str(VALID_PATH))
    if test is not None:
        dfs.append(test);  used.append(str(TEST_PATH))

    all_df = pd.concat(dfs, ignore_index=True)

    # columns: head, relation, tail, time
    entities = sorted(set(all_df[0].tolist() + all_df[2].tolist()))
    relations = sorted(set(all_df[1].tolist()))
    times = sorted(set(all_df[3].tolist()))  # stable (sorted) time IDs

    _write_map(entities, OUT_DIR / "entity2id.txt")
    _write_map(relations, OUT_DIR / "relation2id.txt")
    _write_map(times, OUT_DIR / "time2id.txt")

    print("Built deterministic ID maps from:")
    for p in used:
        print(f"  - {p}")
    print("Wrote:")
    print(f"  entities:  {len(entities)} -> {OUT_DIR/'entity2id.txt'}")
    print(f"  relations: {len(relations)} -> {OUT_DIR/'relation2id.txt'}")
    print(f"  times:     {len(times)} -> {OUT_DIR/'time2id.txt'}")

if __name__ == "__main__":
    main()
