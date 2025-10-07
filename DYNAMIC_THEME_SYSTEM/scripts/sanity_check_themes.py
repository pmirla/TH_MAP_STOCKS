# #!/usr/bin/env python3
# """
# Sanity Check: Stocks × Themes
# =============================

# What it does
# ------------
# 1) Builds a **raw cosine** stock×theme matrix using:
#    - stocks.weighted_embedding  (fallback to 'embedding' if needed)
#    - themes.embedding
#    - L2-row normalization before dot product.

# 2) Prints **theme→theme nearest neighbors** (to catch overlapping themes).

# 3) Shows **side-by-side top-K** themes per ticker:
#    - Cosine top-K (from the raw matrix)
#    - Aggregated top-K (from LanceDB 'stock_theme_relationships')

# Optional exports:
# - Side-by-side CSV (one row per (symbol, rank))
# - Cosine max-theme per symbol CSV
# - Theme→theme neighbors CSV

# Usage examples
# --------------
# # Basic: inspect a few tickers and a theme
# python scripts/diagnostics/sanity_check_themes.py \
#   --symbols ROKU ZM PLTR CMCSA ABBV LLY \
#   --theme space_technology \
#   --top-k 5

# # Export side-by-side results & cosine tops
# python scripts/diagnostics/sanity_check_themes.py \
#   --symbols TSLA AAPL META \
#   --theme communications_networks \
#   --top-k 5 \
#   --out-prefix reports/sanity_checks

# # Use a different LanceDB path
# LANCEDB_PATH=./data/lancedb python scripts/diagnostics/sanity_check_themes.py --symbols NVDA
# """

# import os
# import sys
# import ast
# import argparse
# from pathlib import Path
# import numpy as np
# import pandas as pd
# import lancedb

# # -------------------- helpers --------------------

# def to_np(x):
#     """Robust list/str→np.array converter (float32)."""
#     if isinstance(x, np.ndarray):
#         return x.astype(np.float32)
#     if isinstance(x, list):
#         return np.asarray(x, dtype=np.float32)
#     if isinstance(x, str):
#         try:
#             v = ast.literal_eval(x)
#         except Exception:
#             v = eval(x)  # last resort (avoid in prod if possible)
#         return np.asarray(v, dtype=np.float32)
#     return np.asarray(x, dtype=np.float32)

# def row_normalize(mat: np.ndarray) -> np.ndarray:
#     eps = 1e-12
#     norms = np.linalg.norm(mat, axis=1, keepdims=True) + eps
#     return mat / norms

# def load_tables(db_path: str):
#     db = lancedb.connect(db_path)
#     stocks = db.open_table("enhanced_stocks_dynamic").to_pandas()
#     themes = db.open_table("themes").to_pandas()
#     try:
#         rels = db.open_table("stock_theme_relationships").to_pandas()
#     except Exception:
#         rels = pd.DataFrame(columns=["symbol","theme_name","normalized_score"])
#     return stocks, themes, rels

# def pick_stock_embedding_column(stocks: pd.DataFrame) -> str:
#     for c in ["weighted_embedding", "embedding"]:
#         if c in stocks.columns:
#             return c
#     raise ValueError("No stock embedding column found (expected 'weighted_embedding' or 'embedding').")

# # -------------------- core computations --------------------

# def build_cosine_matrix(stocks: pd.DataFrame, themes: pd.DataFrame):
#     """Return (S, stocks, themes) where S is [n_stocks x n_themes] cosine sims."""
#     stock_emb_col = pick_stock_embedding_column(stocks)
#     A = np.stack(stocks[stock_emb_col].apply(to_np).to_list())
#     T = np.stack(themes["embedding"].apply(to_np).to_list())

#     A = row_normalize(A)
#     T = row_normalize(T)

#     S = A @ T.T           # cosine similarity
#     return S, stocks.reset_index(drop=True), themes.reset_index(drop=True)

# def theme_neighbors(themes: pd.DataFrame, top_k: int = 5, theme_name: str | None = None):
#     """Compute theme→theme nearest neighbors (cosine)."""
#     T = np.stack(themes["embedding"].apply(to_np).to_list())
#     T = row_normalize(T)
#     TT = T @ T.T

#     names = themes["name"].tolist()
#     if theme_name and theme_name not in names:
#         print(f"⚠️ theme '{theme_name}' not found. Available: {names[:10]} ...")
#         theme_name = None

#     out = []
#     if theme_name:
#         j = names.index(theme_name)
#         sims = [(names[i], TT[j, i]) for i in range(len(names)) if i != j]
#         sims.sort(key=lambda x: x[1], reverse=True)
#         sims = sims[:top_k]
#         print(f"\n📎 Nearest themes to '{theme_name}':")
#         for name, val in sims:
#             print(f"   • {name:30s} {val:.3f}")
#         out = [(theme_name, name, float(val)) for name, val in sims]
#     else:
#         # Print/return neighbors for all themes (brief print for first few)
#         for j, tname in enumerate(names):
#             sims = [(names[i], TT[j, i]) for i in range(len(names)) if i != j]
#             sims.sort(key=lambda x: x[1], reverse=True)
#             sims = sims[:top_k]
#             if j < 5:
#                 print(f"\n📎 Nearest themes to '{tname}':")
#                 for name, val in sims:
#                     print(f"   • {name:30s} {val:.3f}")
#             out.extend([(tname, name, float(val)) for name, val in sims])
#     return pd.DataFrame(out, columns=["theme","neighbor","cosine"])

# def topk_from_cosine(symbol: str, k: int, S: np.ndarray, stocks: pd.DataFrame, themes: pd.DataFrame):
#     if symbol not in set(stocks["symbol"]):
#         return [], []
#     i = stocks.index[stocks["symbol"] == symbol][0]
#     row = pd.Series(S[i], index=themes["name"]).sort_values(ascending=False).head(k)
#     return row.index.tolist(), row.values.tolist()

# def topk_from_db(symbol: str, k: int, rels: pd.DataFrame):
#     g = rels[rels["symbol"] == symbol].copy()
#     if g.empty:
#         return [], []
#     g = g.sort_values("normalized_score", ascending=False).head(k)
#     return g["theme_name"].tolist(), g["normalized_score"].astype(float).tolist()

# # -------------------- printing & export --------------------

# def print_side_by_side(symbols: list[str], k: int, S: np.ndarray, stocks: pd.DataFrame, themes: pd.DataFrame, rels: pd.DataFrame):
#     print("\n🔍 Side-by-side (cosine vs aggregated):")
#     for sym in symbols:
#         cos_themes, cos_scores = topk_from_cosine(sym, k, S, stocks, themes)
#         agg_themes, agg_scores = topk_from_db(sym, k, rels)
#         cos_view = [f"{t}:{s:.3f}" for t, s in zip(cos_themes, cos_scores)]
#         agg_view = [f"{t}:{s:.3f}" for t, s in zip(agg_themes, agg_scores)]
#         print(f"  {sym}:")
#         print(f"    cosine: {cos_view}")
#         print(f"      agg: {agg_view}")

# def build_side_by_side_df(symbols: list[str], k: int, S: np.ndarray, stocks: pd.DataFrame, themes: pd.DataFrame, rels: pd.DataFrame):
#     rows = []
#     for sym in symbols:
#         cos_t, cos_s = topk_from_cosine(sym, k, S, stocks, themes)
#         agg_t, agg_s = topk_from_db(sym, k, rels)
#         max_len = max(len(cos_t), len(agg_t), k)
#         for r in range(max_len):
#             rows.append({
#                 "symbol": sym,
#                 "rank": r+1,
#                 "cosine_theme": cos_t[r] if r < len(cos_t) else "",
#                 "cosine_score": float(cos_s[r]) if r < len(cos_s) else np.nan,
#                 "agg_theme": agg_t[r] if r < len(agg_t) else "",
#                 "agg_score": float(agg_s[r]) if r < len(agg_s) else np.nan,
#             })
#     return pd.DataFrame(rows)

# def export_cosine_tops(S: np.ndarray, stocks: pd.DataFrame, themes: pd.DataFrame, path: str, k: int = 1):
#     rows = []
#     for i, sym in enumerate(stocks["symbol"]):
#         row = pd.Series(S[i], index=themes["name"]).sort_values(ascending=False).head(k)
#         for rank, (tname, score) in enumerate(row.items(), start=1):
#             rows.append({"symbol": sym, "rank": rank, "theme": tname, "cosine": float(score)})
#     df = pd.DataFrame(rows)
#     Path(path).parent.mkdir(parents=True, exist_ok=True)
#     df.to_csv(path, index=False)
#     return df

# # -------------------- CLI --------------------

# def parse_args():
#     p = argparse.ArgumentParser(description="Sanity checks for stock×theme embeddings.")
#     p.add_argument("--db", default=os.environ.get("LANCEDB_PATH", "./data/lancedb"), help="LanceDB path")
#     p.add_argument("--symbols", nargs="*", default=None, help="Symbols to inspect (default: a small sample)")
#     p.add_argument("--theme", default=None, help="Theme to show neighbors for (default: show a few themes)")
#     p.add_argument("--top-k", type=int, default=5, help="Top-K to display/export")
#     p.add_argument("--out-prefix", default=None, help="If set, write CSVs with this prefix (e.g., reports/sanity)")
#     return p.parse_args()

# def main():
#     args = parse_args()

#     print(f"🔗 LanceDB: {args.db}")
#     stocks, themes, rels = load_tables(args.db)

#     S, stocks, themes = build_cosine_matrix(stocks, themes)
#     print(f"📐 Cosine matrix: {S.shape[0]} stocks × {S.shape[1]} themes")

#     # Theme→theme neighbors
#     nn_df = theme_neighbors(themes, top_k=args.top_k, theme_name=args.theme)

#     # Which symbols to inspect
#     if args.symbols:
#         symbols = [s.upper() for s in args.symbols]
#     else:
#         # sensible small default set
#         sample = ["ROKU","ZM","PLTR","CMCSA","ABBV","LLY","AAPL","META","TSLA","NVDA"]
#         have = set(stocks["symbol"])
#         symbols = [s for s in sample if s in have][:6] or stocks["symbol"].head(6).tolist()

#     # Side-by-side comparison
#     print_side_by_side(symbols, args.top_k, S, stocks, themes, rels)
#     sbs_df = build_side_by_side_df(symbols, args.top_k, S, stocks, themes, rels)

#     # Optional exports
#     if args.out_prefix:
#         ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
#         base = f"{args.out_prefix}_{ts}"
#         Path(base).parent.mkdir(parents=True, exist_ok=True)

#         # 1) Side-by-side CSV
#         sbs_path = f"{base}_side_by_side.csv"
#         sbs_df.to_csv(sbs_path, index=False)
#         print(f"💾 Wrote {sbs_path}")

#         # 2) Cosine tops per symbol
#         cos_path = f"{base}_cosine_tops.csv"
#         export_cosine_tops(S, stocks, themes, cos_path, k=1)
#         print(f"💾 Wrote {cos_path}")

#         # 3) Theme→theme neighbors
#         if not nn_df.empty:
#             nn_path = f"{base}_theme_neighbors.csv"
#             nn_df.to_csv(nn_path, index=False)
#             print(f"💾 Wrote {nn_path}")

#     print("✅ Done.")

# if __name__ == "__main__":
#     main()


#!/usr/bin/env python3
import os, sys, argparse, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import lancedb

# Project path for config if needed
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

def to_np(x):
    return np.asarray(x, dtype=np.float32)

def row_norm(mat):
    eps = 1e-12
    return mat / (np.linalg.norm(mat, axis=1, keepdims=True) + eps)

def load_tables(db_path="./data/lancedb"):
    db = lancedb.connect(db_path)
    stocks = db.open_table("enhanced_stocks_dynamic").to_pandas()
    themes = db.open_table("themes").to_pandas()
    # stock_theme_relationships is optional (may not exist yet)
    try:
        rels = db.open_table("stock_theme_relationships").to_pandas()
    except Exception:
        rels = pd.DataFrame(columns=[
            "symbol","theme_name","normalized_score","article_count",
            "confidence","latest_article_date","article_titles","article_sources","created_at"
        ])
    return db, stocks, themes, rels

def build_cosine(stocks, themes):
    A = np.stack(stocks["weighted_embedding"].apply(to_np))
    T = np.stack(themes["embedding"].apply(to_np))
    A = row_norm(A); T = row_norm(T)
    S = A @ T.T  # stocks x themes
    return S

def nearest_themes(themes, T, anchor_name, k=5):
    names = themes["name"].tolist()
    if anchor_name not in names:
        return []
    j = names.index(anchor_name)
    sims = T @ T[j]
    order = np.argsort(-sims)
    out = []
    for idx in order:
        if idx == j: 
            continue
        out.append((names[idx], float(sims[idx])))
        if len(out) >= k:
            break
    return out

def topk_for_stock_cosine(symbol, stocks, themes, S, k=5):
    if symbol not in stocks["symbol"].values:
        return []
    i = stocks.index[stocks["symbol"] == symbol][0]
    row = pd.Series(S[i], index=themes["name"]).sort_values(ascending=False).head(k)
    return [(t, float(v)) for t, v in row.items()]

def topk_for_stock_agg(symbol, rels, k=5, min_score=0.0):
    g = rels[rels["symbol"] == symbol]
    if min_score:
        g = g[g["normalized_score"] >= float(min_score)]
    if g.empty:
        return []
    g = g.sort_values("normalized_score", ascending=False).head(k)
    return [(r["theme_name"], float(r["normalized_score"])) for _, r in g.iterrows()]

def parse_args():
    p = argparse.ArgumentParser(description="Theme sanity checks: theme↔theme overlap and per-stock previews.")
    p.add_argument("--db", default="./data/lancedb", help="LanceDB path")
    p.add_argument("--symbols", nargs="*", default=None, help="Stocks to inspect")
    p.add_argument("--theme", nargs="*", default=[], help="Theme(s) to show nearest neighbors for")
    p.add_argument("--top-k", type=int, default=5, help="Top-K items to display")
    p.add_argument("--min-agg-score", type=float, default=0.0, help="Min normalized_score for aggregated per-stock themes")
    return p.parse_args()

def main():
    args = parse_args()

    print(f"🔗 LanceDB: {args.db}")
    db, stocks, themes, rels = load_tables(args.db)

    # Prepare cosine matrices
    S = build_cosine(stocks, themes)  # stocks x themes
    T = row_norm(np.stack(themes["embedding"].apply(to_np)))
    print(f"📐 Cosine matrix: {S.shape[0]} stocks × {S.shape[1]} themes\n")

    # Show nearest themes for requested anchors
    for anchor in args.theme:
        pairs = nearest_themes(themes, T, anchor, k=args.top_k)
        print(f"📎 Nearest themes to '{anchor}':")
        for name, sc in pairs:
            print(f"   • {name:<30} {sc:0.3f}")
        print()

    # Per-stock cosine vs aggregated
    if args.symbols:
        print("📚 Per-stock themes (cosine vs aggregated):")
        for sym in args.symbols:
            cos = topk_for_stock_cosine(sym, stocks, themes, S, k=args.top_k)
            agg = topk_for_stock_agg(sym, rels, k=args.top_k, min_score=args.min_agg_score)

            cos_str = [f"{t}:{v:0.3f}" for t, v in cos] if cos else []
            agg_str = [f"{t}:{v:0.3f}" for t, v in agg] if agg else []

            print(f"  {sym}:")
            print(f"    cosine: {cos_str if cos_str else '[]'}")
            print(f"      agg: {agg_str if agg_str else '[]'}")
        print("✅ Done.")
    else:
        print("✅ Done.")

if __name__ == "__main__":
    main()
