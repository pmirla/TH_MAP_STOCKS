#!/usr/bin/env python3
# diagnostic_full_cosine_matrix.py
import pandas as pd, numpy as np, lancedb, ast

DB_PATH = "./data/lancedb"
OUT = "reports/stock_theme_matrix_cosine.csv"

def to_np(x):
    if isinstance(x, list):
        return np.asarray(x, dtype=np.float32)
    if isinstance(x, str):
        return np.asarray(ast.literal_eval(x), dtype=np.float32)
    return np.asarray(x, dtype=np.float32)

db = lancedb.connect(DB_PATH)
stocks = db.open_table("enhanced_stocks_dynamic").to_pandas()[["symbol","weighted_embedding"]]
themes = db.open_table("themes").to_pandas()[["name","embedding"]]

A = np.stack(stocks["weighted_embedding"].apply(to_np))
T = np.stack(themes["embedding"].apply(to_np))

# cosine
A = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
T = T / (np.linalg.norm(T, axis=1, keepdims=True) + 1e-12)
S = A @ T.T  # shape: n_stocks x n_themes

matrix = pd.DataFrame(S, index=stocks["symbol"], columns=themes["name"])
matrix.to_csv(OUT)
print(f"Saved {OUT} {matrix.shape}")
