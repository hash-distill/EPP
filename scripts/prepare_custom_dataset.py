import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def load_pairs(path: Path):
    pairs = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if "," not in line:
                raise ValueError(f"Invalid index line (expected 'ag_id,ab_id'): {line}")
            ag_id, ab_id = [part.strip() for part in line.split(",", 1)]
            pairs.append((ag_id, ab_id))
    return pairs


def label_to_array(label_str: str, seq_len: int, name: str):
    label_str = str(label_str).strip()
    if not set(label_str) <= {"0", "1"}:
        raise ValueError(f"{name} label has non-binary chars: {label_str[:30]}...")
    arr = np.fromiter(label_str, dtype=np.int8)
    if arr.shape[0] != seq_len:
        raise ValueError(f"{name} label length {arr.shape[0]} != seq len {seq_len}")
    return arr


def select_by_pairs(df: pd.DataFrame, pairs):
    df_key = df.set_index(["antigen_id", "antibody_id"], drop=False)
    rows = []
    missing = []
    for pair in pairs:
        if pair not in df_key.index:
            missing.append(pair)
            continue
        row = df_key.loc[pair]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        rows.append(row)
    return pd.DataFrame(rows), missing


def build_labels(df: pd.DataFrame):
    label_mats = []
    for _, row in df.iterrows():
        ag_seq = str(row["antigen_seq"])
        ab_seq = str(row["antibody_seq"])
        ag_label = label_to_array(row["antigen_label"], len(ag_seq), "antigen")
        ab_label = label_to_array(row["antibody_label"], len(ab_seq), "antibody")
        label_mats.append(torch.from_numpy(np.outer(ag_label, ab_label)).float())

    if not label_mats:
        raise ValueError("No samples after filtering; cannot build label tensor.")

    max_rows = max(m.shape[0] for m in label_mats)
    max_cols = max(m.shape[1] for m in label_mats)
    padded = [
        torch.nn.functional.pad(m, (0, max_cols - m.shape[1], 0, max_rows - m.shape[0]))
        for m in label_mats
    ]
    return torch.stack(padded)


def write_split(df_split: pd.DataFrame, output_dir: Path, csv_name: str, label_name: str):
    output_dir.mkdir(parents=True, exist_ok=True)

    train_csv = pd.DataFrame({
        "pdbid": df_split["antigen_id"].astype(str) + "_" + df_split["antibody_id"].astype(str),
        "abseq": df_split["antibody_seq"].astype(str),
        "agseq": df_split["antigen_seq"].astype(str),
    })
    train_csv.to_csv(output_dir / csv_name, index=False)

    label_y = build_labels(df_split)
    torch.save(label_y.cpu(), output_dir / label_name)


def main():
    parser = argparse.ArgumentParser(description="Prepare custom dataset splits without overwriting paper data.")
    parser.add_argument("--tsv", default="data/custom/fasta_label4.5.tsv", help="Input TSV with labels.")
    parser.add_argument("--train-index", default="data/custom/train_c30_10p.txt", help="Train index file.")
    parser.add_argument("--test-index", default="data/custom/test_c30_10p.txt", help="Test index file.")
    parser.add_argument("--output-dir", default="data/custom", help="Output directory.")
    parser.add_argument("--train-csv", default="train.csv", help="Output train CSV name.")
    parser.add_argument("--train-label", default="label_y.pt", help="Output train label tensor name.")
    parser.add_argument("--test-csv", default="test.csv", help="Output test CSV name.")
    parser.add_argument("--test-label", default="label_y_test.pt", help="Output test label tensor name.")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    tsv_path = (repo_root / args.tsv).resolve()
    output_dir = (repo_root / args.output_dir).resolve()

    df = pd.read_csv(tsv_path, sep="\t")
    required = {
        "antigen_id",
        "antigen_seq",
        "antigen_label",
        "antibody_id",
        "antibody_seq",
        "antibody_label",
    }
    missing_cols = required - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing columns in TSV: {sorted(missing_cols)}")

    train_pairs = load_pairs((repo_root / args.train_index).resolve())
    test_pairs = load_pairs((repo_root / args.test_index).resolve())

    train_df, train_missing = select_by_pairs(df, train_pairs)
    test_df, test_missing = select_by_pairs(df, test_pairs)

    if train_missing:
        print(f"Train missing pairs: {len(train_missing)}")
    if test_missing:
        print(f"Test missing pairs: {len(test_missing)}")

    print(f"Train samples: {len(train_df)}")
    print(f"Test samples: {len(test_df)}")

    write_split(train_df, output_dir, args.train_csv, args.train_label)
    write_split(test_df, output_dir, args.test_csv, args.test_label)

    print("Done.")


if __name__ == "__main__":
    main()
