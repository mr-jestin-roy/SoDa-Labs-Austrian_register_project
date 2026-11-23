"""
sample_comma_patterns.py
─────────────────────────
Randomly samples 500 unique entries per comma-count bucket (0, 1, 2, 3, 4+)
from RTB1's transcribed_text column, plus population-wide comma stats.

Usage:
    python sample_comma_patterns.py
"""

import pandas as pd

SRC = "master_RTB_1_Aach-Berlichingen_clean.xlsx"
OUT = "RTB1_comma_samples.xlsx"
SHEET = "Reichstelefonbuch"
PER_BUCKET = 500
SEED = 42

BUCKETS = [0, 1, 2, 3, "4+"]


def comma_bucket(text: str) -> "str | int":
    n = text.count(",")
    return n if n < 4 else "4+"


def main() -> None:
    df = pd.read_excel(SRC, sheet_name=SHEET)
    df = df.dropna(subset=["transcribed_text"]).reset_index(drop=True)

    df["comma_type"] = df["transcribed_text"].apply(comma_bucket)

    # Population-wide stats (over all eligible rows, no sampling involved).
    counts = df["comma_type"].value_counts().reindex(BUCKETS, fill_value=0)
    summary = pd.DataFrame(
        {
            "comma_type": BUCKETS,
            "row_count": counts.values,
            "percent": (counts / len(df) * 100).round(1).values,
        }
    )

    # Random draw: shuffle once with pandas' PRNG, then bucket-fill in that
    # random order so every bucket gets exactly PER_BUCKET unique rows.
    shuffled = df.sample(frac=1, random_state=SEED).reset_index(drop=True)

    picked = {b: [] for b in BUCKETS}
    for _, row in shuffled.iterrows():
        b = row["comma_type"]
        if len(picked[b]) < PER_BUCKET:
            picked[b].append(row)
        if all(len(v) >= PER_BUCKET for v in picked.values()):
            break

    for b in BUCKETS:
        if len(picked[b]) < PER_BUCKET:
            raise RuntimeError(f"Bucket {b!r} only has {len(picked[b])} rows, need {PER_BUCKET}")

    samples = pd.concat([pd.DataFrame(rows) for rows in picked.values()], ignore_index=True)

    with pd.ExcelWriter(OUT, engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="Summary", index=False)
        samples.to_excel(writer, sheet_name="Samples", index=False)

    print(summary.to_string(index=False))
    print(f"\nWrote {len(samples)} sampled rows ({PER_BUCKET} per bucket) to {OUT}")


if __name__ == "__main__":
    main()
