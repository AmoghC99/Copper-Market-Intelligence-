from pathlib import Path

import pandas as pd


def main() -> None:
    p = Path("data/raw/fred")
    files = sorted(
        p.glob("fred_PCOPPUSDM_*.csv"), key=lambda x: x.stat().st_mtime, reverse=True
    )
    if not files:
        print("No FRED files found")
        return
    f = files[0]
    print("File:", f, f.stat().st_size, "bytes")
    df = pd.read_csv(f)
    print("Columns:", list(df.columns))
    date_col = None
    for c in df.columns:
        if "date" in c.lower():
            date_col = c
            break
    if date_col is None:
        date_col = df.columns[0]
    print("Date column:", date_col)
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    print("Parsed dates - missing:", int(df[date_col].isna().sum()))
    num_cols = [c for c in df.columns if c != date_col]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        print(
            "Col %s: numeric missing=%s, dtype=%s"
            % (c, int(df[c].isna().sum()), df[c].dtype)
        )
    print("Duplicate dates count:", int(df[date_col].duplicated().sum()))
    print("Earliest:", df[date_col].min())
    print("Latest:", df[date_col].max())
    print("\nFirst 5 rows:")
    print(df.head().to_string(index=False))


if __name__ == "__main__":
    main()
