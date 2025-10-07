#!/usr/bin/env python3
"""
Article→Theme Scoring (Vectorized) + Selective Export + Top-K Masked Matrix
=========================================================================

- Reads LanceDB tables: stock_articles, themes, enhanced_stocks_dynamic
- Optional SYMBOLS / SYMBOL_FILE env filters and CLI --symbols / --symbols-file
- Vectorized cosine similarity; threshold keeps article→theme links
- Aggregates to per-stock normalized scores
- Safe upserts into:
    • article_theme_links         (KEY: ['symbol','article_url','theme_name'] with fallbacks)
    • stock_theme_relationships   (KEY: ['symbol','theme_name'])
- Exports CSV/JSON + two matrices:
    • Full dense matrix
    • Top-K masked matrix (only top-K per stock; others blank)

Env overrides (optional):
  LANCEDB_PATH=./data/lancedb
  AT_SIM_THRESHOLD=0.25
  TOP_N=5
  SYMBOLS="TSLA,AAPL"
  SYMBOL_FILE=/path/to.txt
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import lancedb

# Project path for config if needed
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
try:
    from core.config import get_settings
except Exception:
    get_settings = None


# ------------------------------ helpers ------------------------------

def _load_symbols_from_env() -> list[str] | None:
    syms_env = os.environ.get("SYMBOLS")
    syms_file = os.environ.get("SYMBOL_FILE")
    symbols = None
    if syms_env:
        symbols = [s.strip().upper() for s in syms_env.split(',') if s.strip()]
    elif syms_file and os.path.exists(syms_file):
        symbols = _read_symbols_file(syms_file)
    return symbols


def _read_symbols_file(path: str) -> list[str]:
    path = str(path)
    try:
        if path.lower().endswith(('.csv', '.tsv')):
            sep = ',' if path.lower().endswith('.csv') else '\t'
            df = pd.read_csv(path, sep=sep)
            cand = [c for c in df.columns if c.lower() in {"symbol","ticker","tickers"}]
            if cand:
                return [str(x).strip().upper() for x in df[cand[0]].dropna().tolist()]
            return [str(x).strip().upper() for x in df.iloc[:,0].dropna().tolist()]
        else:
            with open(path, 'r') as f:
                return [line.strip().upper() for line in f if line.strip()]
    except Exception:
        return []


def _np_normalize(mat: np.ndarray) -> np.ndarray:
    eps = 1e-12
    norms = np.linalg.norm(mat, axis=1, keepdims=True) + eps
    return mat / norms


def _to_np_embedding(x, dim: int = 384) -> np.ndarray:
    """Robustly coerce a list-like embedding to np.array; fallback to zeros."""
    try:
        arr = np.asarray(x, dtype=np.float32)
        if arr.ndim == 1 and arr.size > 0:
            return arr
    except Exception:
        pass
    return np.zeros((dim,), dtype=np.float32)


def _safe_article_id(row: pd.Series, fallback: str) -> str:
    """
    Provide a stable identifier for an article to use in logs/preview, NOT as upsert key.
    Priority: article_id → content_hash → url → fallback
    """
    for k in ("article_id", "content_hash", "url"):
        v = row.get(k)
        if v is not None and str(v).strip():
            return str(v)
    return fallback


def apply_result_filters(
    stock_theme_df: pd.DataFrame,
    include_themes: list[str] | None = None,
    exclude_themes: list[str] | None = None,
    min_score: float | None = None,
    top_k: int | None = None,
) -> pd.DataFrame:
    """Filter & trim stock-level aggregates before saving/exporting."""
    df = stock_theme_df.copy()

    if include_themes:
        df = df[df["theme_name"].isin(include_themes)]
    if exclude_themes:
        df = df[~df["theme_name"].isin(exclude_themes)]
    if min_score is not None:
        df = df[df["normalized_score"] >= float(min_score)]

    if top_k:
        df = (
            df.sort_values(["symbol", "normalized_score"], ascending=[True, False])
              .groupby("symbol", as_index=False, sort=False)
              .head(int(top_k))
        )

    return df.reset_index(drop=True)


def export_filtered_results(
    df: pd.DataFrame,
    out_prefix: str,
    export_mode: str = "csv",
) -> list[str]:
    """Export filtered stock-theme rows to CSV/JSON."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = f"{out_prefix}_{ts}"
    Path(out_prefix).parent.mkdir(parents=True, exist_ok=True)

    exports = []
    if export_mode in ("csv", "both"):
        csv_path = f"{base}.csv"
        df.to_csv(csv_path, index=False)
        exports.append(csv_path)
    if export_mode in ("json", "both"):
        json_path = f"{base}.json"
        df.to_json(json_path, orient="records", indent=2, date_format="iso")
        exports.append(json_path)
    return exports


# ------------------------------ core class ------------------------------

class ArticleThemeScorer:
    def __init__(self,
                 lancedb_path: str | None = None,
                 at_sim_threshold: float | None = None,
                 top_n: int | None = None):
        settings = get_settings() if get_settings else None
        self.db_path = lancedb_path or (settings.lancedb_path if settings and hasattr(settings, 'lancedb_path') else './data/lancedb')
        self.db = lancedb.connect(self.db_path)
        self.threshold = at_sim_threshold if at_sim_threshold is not None else float(os.environ.get('AT_SIM_THRESHOLD', 0.25))
        self.top_n = top_n if top_n is not None else int(os.environ.get('TOP_N', 5))
        self.symbol_filter = set(_load_symbols_from_env() or [])

        print("🔬 Article→Theme Scorer Initialized")
        print(f"🔗 LanceDB: {self.db_path}")
        print(f"🎚️ Threshold: {self.threshold}")
        if self.symbol_filter:
            print(f"🔎 Symbol filter: {sorted(self.symbol_filter)}")
        print("-"*80)

    # ---------- data ----------
    def load(self) -> bool:
        try:
            self.articles = self.db.open_table('stock_articles').to_pandas()
            self.themes   = self.db.open_table('themes').to_pandas()
            self.stocks   = self.db.open_table('enhanced_stocks_dynamic').to_pandas()
        except Exception as e:
            print(f"❌ Failed to load tables: {e}")
            return False

        # Normalize symbols early
        for df in (self.articles, self.stocks):
            if 'symbol' in df.columns:
                df['symbol'] = df['symbol'].astype(str).str.strip().str.upper()

        # Optional filter
        if self.symbol_filter:
            before = len(self.articles)
            self.articles = self.articles[self.articles['symbol'].isin(self.symbol_filter)].reset_index(drop=True)
            print(f"🧹 Filtered articles by symbols: {before} → {len(self.articles)}")

        # Convert list embeddings → np arrays (robust)
        self.articles['embedding'] = self.articles['embedding'].apply(_to_np_embedding)
        self.themes['embedding']   = self.themes['embedding'].apply(_to_np_embedding)

        print(f"📄 Articles: {len(self.articles)} | 🎯 Themes: {len(self.themes)} | 📈 Stocks: {len(self.stocks)}")
        return True

    # ---------- scoring ----------
    def score_article_theme(self) -> pd.DataFrame:
        if len(self.articles) == 0 or len(self.themes) == 0:
            return pd.DataFrame()

        A = np.stack(self.articles['embedding'].to_list())
        T = np.stack(self.themes['embedding'].to_list())
        A = _np_normalize(A)
        T = _np_normalize(T)

        # cosine sim matrix (n_articles x n_themes)
        S = A @ T.T

        # threshold mask
        mask = S >= self.threshold
        ai, ti = np.where(mask)
        if ai.size == 0:
            print(f"ℹ️ No links above threshold {self.threshold}")
            return pd.DataFrame()

        # build rows
        rows = []
        for i, j in zip(ai, ti):
            art = self.articles.iloc[i]
            th  = self.themes.iloc[j]
            rows.append({
                # For logs/uniqs only; NOT used as upsert key:
                'article_id': _safe_article_id(art, f"art_{i}"),
                'symbol': art['symbol'],
                'article_title': art.get('title'),
                'article_url': art.get('url'),
                'source_domain': art.get('source_domain'),
                'retrieved_at': art.get('retrieved_at'),
                'content_hash': art.get('content_hash'),
                'theme_name': th['name'],
                'similarity_score': float(S[i, j]),
                'article_score': float(art.get('score', 1.0)),
            })
        df = pd.DataFrame(rows)
        # Count uniq using the safe field we set above
        uniq_articles = df['article_id'].nunique(dropna=True)
        uniq_stocks   = df['symbol'].nunique(dropna=True)
        print(f"✅ Links kept: {len(df)} | uniq articles: {uniq_articles} | uniq stocks: {uniq_stocks}")
        return df

    # ---------- aggregate ----------
    @staticmethod
    def _aggregate_stock_theme(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame()
        out = []
        for (sym, theme), g in df.groupby(['symbol','theme_name'], sort=False):
            article_count = len(g)
            avg_sim = g['similarity_score'].mean()
            max_sim = g['similarity_score'].max()
            w_sim = (g['similarity_score'] * g['article_score']).sum() / max(g['article_score'].sum(), 1e-9)
            confidence = min(1.0, article_count / 3.0)
            normalized = 0.4*avg_sim + 0.3*max_sim + 0.2*w_sim + 0.1*confidence
            out.append({
                'symbol': sym,
                'theme_name': theme,
                'normalized_score': float(normalized),
                'avg_similarity': float(avg_sim),
                'max_similarity': float(max_sim),
                'weighted_similarity': float(w_sim),
                'article_count': int(article_count),
                'confidence': float(confidence),
                'latest_article_date': pd.to_datetime(g['retrieved_at'], errors='coerce').max(),
                'article_titles': g['article_title'].dropna().tolist(),
                'article_sources': g['source_domain'].dropna().unique().tolist(),
            })
        return pd.DataFrame(out)

    # ---------- persistence ----------
    def _upsert_table(self, name: str, new_df: pd.DataFrame, key_cols: list[str]):
        """Merge by key_cols: keep existing rows where keys don't match, replace where they do."""
        if new_df is None or new_df.empty:
            return 0
        try:
            try:
                existing = self.db.open_table(name).to_pandas()
            except Exception:
                existing = pd.DataFrame(columns=new_df.columns)

            # Ensure key columns exist
            for c in key_cols:
                if c not in new_df.columns:
                    new_df[c] = None
                if c not in existing.columns:
                    existing[c] = None

            # Merge: keep existing rows not matched by new keys
            if not existing.empty:
                keep = existing.merge(new_df[key_cols].drop_duplicates(), on=key_cols, how='left', indicator=True)
                keep = keep[keep['_merge'] == 'left_only'].drop(columns=['_merge'])
                combined = pd.concat([keep, new_df], ignore_index=True)
            else:
                combined = new_df

            # Atomic replace
            try:
                self.db.drop_table(name)
            except Exception:
                pass
            self.db.create_table(name, combined)
            return len(new_df)
        except Exception as e:
            print(f"❌ Upsert error for {name}: {e}")
            return 0

    # ---------- matrices ----------
    @staticmethod
    def _make_dense_matrix(agg_df: pd.DataFrame) -> pd.DataFrame:
        """Stock (rows) x Theme (cols) matrix of normalized_score, zeros where absent."""
        if agg_df.empty:
            return pd.DataFrame()
        return agg_df.pivot(index='symbol', columns='theme_name', values='normalized_score').fillna(0.0)

    @staticmethod
    def _make_topk_masked_matrix(agg_df: pd.DataFrame, k: int) -> pd.DataFrame:
        """
        Keep only top-K themes per stock, others blank (''), for nicer human scanning.
        Stored as strings to allow blanks in CSV.
        """
        if agg_df.empty:
            return pd.DataFrame()
        topk = (
            agg_df.sort_values(['symbol','normalized_score'], ascending=[True, False])
                  .groupby('symbol', as_index=False, sort=False)
                  .head(k)
        )
        dense = agg_df.pivot(index='symbol', columns='theme_name', values='normalized_score')
        masked = pd.DataFrame('', index=dense.index, columns=dense.columns)
        for _, r in topk.iterrows():
            masked.at[r['symbol'], r['theme_name']] = f"{r['normalized_score']:.3f}"
        return masked

    # ---------- run ----------
    def run(self,
            include_themes: list[str] | None = None,
            exclude_themes: list[str] | None = None,
            min_score: float | None = None,
            top_k: int | None = None,
            do_upsert: bool = True,
            export_mode: str = "none",
            out_prefix: str = "reports/article_theme_filtered"):
        if not self.load():
            return

        links = self.score_article_theme()
        if links.empty:
            print("❌ No article→theme links above threshold; aborting")
            return

        # Aggregate to stock-level
        agg = self._aggregate_stock_theme(links)
        if agg.empty:
            print("❌ No stock-level aggregates produced")
            return

        agg = agg.sort_values(['symbol','normalized_score'], ascending=[True, False]).reset_index(drop=True)

        # Apply user filters (include/exclude/min_score/top_k for *saved/exported* view)
        filtered = apply_result_filters(
            agg,
            include_themes=include_themes,
            exclude_themes=exclude_themes,
            min_score=min_score,
            top_k=top_k,
        )

        # Persist (filtered set) unless told not to
        now = datetime.now().isoformat()
        if not filtered.empty and do_upsert:
            # Filter links to only those stocks/themes that survived filtering
            f_links = links[links['theme_name'].isin(filtered['theme_name'].unique()) &
                            links['symbol'].isin(filtered['symbol'].unique())].copy()
            f_links['created_at'] = now
            filtered['created_at'] = now

            # Robust link keys: prefer article_url; fallback to content_hash; last resort include article_id
            if 'article_url' not in f_links.columns:
                f_links['article_url'] = None
            if 'content_hash' not in f_links.columns:
                f_links['content_hash'] = None

            # Split rows with/without URLs so we have a deterministic key
            has_url = f_links['article_url'].notna() & (f_links['article_url'].astype(str).str.len() > 0)
            links_with_url = f_links[has_url].copy()
            links_no_url   = f_links[~has_url].copy()

            n_links = 0
            if not links_with_url.empty:
                n_links += self._upsert_table('article_theme_links', links_with_url,
                                              key_cols=['symbol','article_url','theme_name'])
            if not links_no_url.empty:
                n_links += self._upsert_table('article_theme_links', links_no_url,
                                              key_cols=['symbol','content_hash','theme_name'])

            n_aggs  = self._upsert_table('stock_theme_relationships', filtered,
                                         key_cols=['symbol','theme_name'])
            print(f"💾 Upserted: {n_links} links, {n_aggs} stock-theme rows")
        else:
            print("ℹ️ Skipping LanceDB upsert (no rows after filters or --no-upsert).")

        # Reports
        Path('reports').mkdir(exist_ok=True, parents=True)
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')

        dense_matrix = self._make_dense_matrix(filtered)
        masked_matrix = self._make_topk_masked_matrix(filtered, k=(top_k or self.top_n))

        if not dense_matrix.empty:
            dense_path = f"reports/stock_theme_matrix_dense_{ts}.csv"
            dense_matrix.to_csv(dense_path)
            print(f"💾 Saved dense matrix: {dense_path}")

        if not masked_matrix.empty:
            masked_path = f"reports/stock_theme_matrix_top{top_k or self.top_n}_{ts}.csv"
            masked_matrix.to_csv(masked_path)
            print(f"💾 Saved top-{top_k or self.top_n} masked matrix: {masked_path}")

        # Optional export of the filtered long-form rows
        if export_mode != "none" and not filtered.empty:
            paths = export_filtered_results(filtered, out_prefix, export_mode)
            for p in paths:
                print(f"💾 Exported: {p}")

        # Console preview (from filtered)
        topn_k = top_k or self.top_n
        topn = {}
        for sym, g in filtered.groupby('symbol'):
            g2 = g.nlargest(topn_k, 'normalized_score')
            topn[sym] = [{
                'theme': r['theme_name'],
                'score': r['normalized_score'],
                'confidence': r['confidence'],
                'article_count': r['article_count'],
                'sources': r['article_sources'],
            } for _, r in g2.iterrows()]

        # Console preview (honor --preview; -1 => all)
        if topn:
            keys = list(topn.keys())
            n = getattr(self, "_preview_n", 5) if hasattr(self, "_preview_n") else 5
            if n < 0 or n >= len(keys):
                n = len(keys)
            print(f"\n🔍 Preview (first {n} stocks):")
            for sym in keys[:n]:
                preview = [f"{t['theme']}:{t['score']:.3f}" for t in topn[sym]]
                print(f"  {sym}: {preview}")
        else:
            print("ℹ️ No rows to preview after filtering.")



def parse_args():
    parser = argparse.ArgumentParser(description="Article→Theme scoring with selective export/preview and top-K masked matrix.")
    parser.add_argument("--symbols", nargs="*", default=None, help="Limit scoring/eval to these tickers (overrides env SYMBOLS/SYMBOL_FILE)")
    parser.add_argument("--symbols-file", default=None, help="File with one ticker per line, or CSV/TSV with a symbol/ticker column")
    parser.add_argument("--min-score", type=float, default=None, help="Filter themes with normalized_score below this")
    parser.add_argument("--top-k", type=int, default=None, help="Keep top-K themes per stock after filtering (also used for masked matrix)")
    parser.add_argument("--themes-include", nargs="*", default=None, help="Only include these theme names")
    parser.add_argument("--themes-exclude", nargs="*", default=None, help="Exclude these theme names")
    parser.add_argument("--preview", type=int, default=5, help="How many stocks to preview in console (from filtered view)")
    parser.add_argument("--export", choices=["csv","json","both","none"], default="none", help="Write filtered stock-theme rows to file(s)")
    parser.add_argument("--out-prefix", default="reports/article_theme_filtered", help="Path prefix for export files")
    parser.add_argument("--no-upsert", action="store_true", help="Skip upserting to LanceDB (preview/export only)")
    parser.add_argument("--threshold", type=float, default=None, help="Article→theme similarity threshold used in scoring step (overrides env)")
    return parser.parse_args()


def main():
    args = parse_args()

    # Instantiate with optional threshold
    scorer = ArticleThemeScorer(at_sim_threshold=args.threshold)
    scorer._preview_n = args.preview  # <- add this line
    # CLI symbols override env
    if args.symbols_file:
        scorer.symbol_filter = set(_read_symbols_file(args.symbols_file))
    if args.symbols:
        scorer.symbol_filter = set([s.upper() for s in args.symbols])

    print("🚀 Starting Article→Theme Scoring & Stock Aggregation")

    scorer.run(
        include_themes=args.themes_include,
        exclude_themes=args.themes_exclude,
        min_score=args.min_score,
        top_k=args.top_k,
        do_upsert=(not args.no_upsert),
        export_mode=args.export,
        out_prefix=args.out_prefix,
    )

    print("\n✅ Done.")


if __name__ == '__main__':
    main()
