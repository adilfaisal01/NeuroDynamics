import pandas as pd
import os
from itertools import combinations

datasets_dir = "datasets"
files = sorted(os.listdir(datasets_dir))

print("=== Dataset Analysis: Double Pendulum Configurations ===\n")

all_configs = {}
config_cols = ["config_id", "mass pendulum 1", "mass pendulum 2", "length pendulum 1", "length pendulum 2"]
key_cols = ["mass pendulum 1", "mass pendulum 2", "length pendulum 1", "length pendulum 2"]

for f in files:
    path = os.path.join(datasets_dir, f)
    df = pd.read_parquet(path)
    unique_configs = df[config_cols].drop_duplicates().sort_values("config_id")

    print(f"--- {f} ---")
    print(f"  Rows: {len(df)}, Unique configs: {len(unique_configs)}")
    print(f"  Config IDs: {sorted(unique_configs['config_id'].unique())}")
    for _, row in unique_configs.iterrows():
        print(f"    config_id={int(row['config_id']):3d}  m1={row['mass pendulum 1']:.6f}  m2={row['mass pendulum 2']:.6f}  l1={row['length pendulum 1']:.6f}  l2={row['length pendulum 2']:.6f}")
    print()

    all_configs[f] = unique_configs

print("=== Cross-Dataset Overlap Analysis ===\n")
for (f1, df1), (f2, df2) in combinations(all_configs.items(), 2):
    merged = pd.merge(df1[key_cols].round(10), df2[key_cols].round(10), on=key_cols, how="inner")
    if len(merged) > 0:
        print(f"  OVERLAP: {f1} <-> {f2}: {len(merged)} shared config(s)")
        for _, row in merged.iterrows():
            print(f"    m1={row['mass pendulum 1']:.6f}  m2={row['mass pendulum 2']:.6f}  l1={row['length pendulum 1']:.6f}  l2={row['length pendulum 2']:.6f}")
    else:
        print(f"  NO OVERLAP: {f1} <-> {f2}")

print("\n=== Intra-Dataset Duplicate Check ===\n")
for f, df in all_configs.items():
    dupes = df[key_cols].round(10).duplicated()
    if dupes.any():
        print(f"  DUPLICATES in {f}: {dupes.sum()} duplicate config(s)")
    else:
        print(f"  CLEAN: {f} - no internal duplicates")

print("\n=== Summary ===")
print(f"  Total datasets: {len(files)}")
print(f"  Total unique configs across all datasets: {sum(len(df) for df in all_configs.values())}")
