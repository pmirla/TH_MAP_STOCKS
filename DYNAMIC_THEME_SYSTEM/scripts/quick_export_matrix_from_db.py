#!/usr/bin/env python3
# quick_export_matrix_from_db.py
import pandas as pd, numpy as np, lancedb

DB_PATH = "./data/lancedb"
OUT = "reports/stock_theme_matrix_from_db.csv"
TOPK_OUT = "reports/stock_theme_matrix_from_db_top5.csv"
TOPK = 5  # optional masked preview

db = lancedb.connect(DB_PATH)

rels = db.open_table("stock_theme_relationships").to_pandas()
stocks = db.open_table("enhanced_stocks_dynamic").to_pandas()[["symbol"]].drop_duplicates()
themes = db.open_table("themes").to_pandas()[["name"]].drop_duplicates()

# Pivot whatever is in relationships
pivot = rels.pivot(index="symbol", columns="theme_name", values="normalized_score")

# Reindex to ALL stocks & ALL themes so nothing is silently dropped
pivot = pivot.reindex(
    index=stocks["symbol"].sort_values().tolist(),
    columns=themes["name"].sort_values().tolist()
).fillna(0.0)

pivot.to_csv(OUT)
print(f"Saved {OUT} {pivot.shape}")

# Optional: top-K masked (blank out everything except top-K per stock)
masked = pd.DataFrame('', index=pivot.index, columns=pivot.columns)
for sym, row in pivot.iterrows():
    top = row.nlargest(TOPK)
    masked.loc[sym, top.index] = top.round(3).astype(str)
masked.to_csv(TOPK_OUT)
print(f"Saved {TOPK_OUT} {masked.shape}")
