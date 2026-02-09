
from __future__ import annotations
from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import Dataset

class ICEWSData(Dataset):

    def __init__(
        self,
        split: str,
        fourteen: bool = True,
        base_dir: str = ".",
        valid_ratio: float = 0.05,
        seed: int = 13,
    ):
        assert split in {"train", "valid", "test"}
        if not fourteen:
            raise ValueError("This loader is configured for ICEWS14 only.")

        self.base_dir = Path(base_dir)
        self.maps_dir = self.base_dir / "icews14"

        self.entity2id = self._load_map(self.maps_dir / "entity2id.txt")
        self.relation2id = self._load_map(self.maps_dir / "relation2id.txt")
        self.time2id = self._load_map(self.maps_dir / "time2id.txt")

        train_path = self.base_dir / "icews_2014_train.txt"
        valid_path = self.base_dir / "icews_2014_valid.txt"
        test_path  = self.base_dir / "icews_2014_test.txt"

        if not train_path.exists():
            raise FileNotFoundError(f"Missing {train_path}. Put icews_2014_train.txt in the project root.")

        if split == "train":
            df = pd.read_csv(train_path, sep="\t", header=None, dtype=str)

            # If no external valid file exists, carve out a deterministic validation split
            if not valid_path.exists() and valid_ratio > 0.0:
                df = self._split_train(df, split="train", valid_ratio=valid_ratio, seed=seed)

        elif split == "valid":
            if valid_path.exists():
                df = pd.read_csv(valid_path, sep="\t", header=None, dtype=str)
            else:
                # deterministic holdout from train
                df_all = pd.read_csv(train_path, sep="\t", header=None, dtype=str)
                df = self._split_train(df_all, split="valid", valid_ratio=valid_ratio, seed=seed)

        else:  # test
            if test_path.exists():
                df = pd.read_csv(test_path, sep="\t", header=None, dtype=str)
            else:
                # If no official test exists, fall back to "valid" split behavior (still deterministic)
                df_all = pd.read_csv(train_path, sep="\t", header=None, dtype=str)
                df = self._split_train(df_all, split="valid", valid_ratio=valid_ratio, seed=seed)

        # Map to IDs
        h = df[0].map(self.entity2id)
        r = df[1].map(self.relation2id)
        t = df[2].map(self.entity2id)
        ti = df[3].map(self.time2id)

        #detect, debug early
        if h.isna().any() or r.isna().any() or t.isna().any() or ti.isna().any():

            bad = df[h.isna() | r.isna() | t.isna() | ti.isna()].head(5)
            raise ValueError(
                "Found unmapped items. Ensure you ran build_icews14_id_maps.py first "
                "and that maps were built from the same files used here. "
                f"Examples:\n{bad}"
            )

        self.h = h.astype("int64").tolist()
        self.r = r.astype("int64").tolist()
        self.t = t.astype("int64").tolist()
        self.time = ti.astype("int64").tolist()

    @staticmethod
    def _split_train(df: pd.DataFrame, split: str, valid_ratio: float, seed: int) -> pd.DataFrame:
        # deterministic shuffle/split
        df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        n = len(df)
        n_valid = max(1, int(n * valid_ratio))
        valid_df = df.iloc[:n_valid]
        train_df = df.iloc[n_valid:]
        return train_df if split == "train" else valid_df

    @staticmethod
    def _load_map(path: Path) -> dict[str, int]:
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path}. Run build_icews14_id_maps.py to generate id maps."
            )
        m: dict[str, int] = {}
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                item, idx = line.rstrip("\n").split("\t")
                m[item] = int(idx)
        return m

    def __len__(self) -> int:
        return len(self.h)

    def __getitem__(self, idx: int):
        return (
            torch.tensor(self.h[idx], dtype=torch.long),
            torch.tensor(self.r[idx], dtype=torch.long),
            torch.tensor(self.t[idx], dtype=torch.long),
            torch.tensor(self.time[idx], dtype=torch.long),
        )
