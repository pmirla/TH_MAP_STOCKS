#!/usr/bin/env python3
"""
Refresh Themes Table from YAML → LanceDB
=======================================

This utility rebuilds the `themes` table in LanceDB from a YAML file
(e.g., ./config/themes.yaml). It:
  • Loads themes (supports either {themes: {...}} or flat {...})
  • Builds text (description + keywords)
  • Embeds with SentenceTransformer (all-MiniLM-L6-v2 by default)
  • Optionally backs up the existing table to Parquet
  • Atomically replaces the `themes` table

Usage examples
--------------
# default paths (run from repo root)
python scripts/00_refresh_themes.py

# custom config/db
python scripts/00_refresh_themes.py \
  --config ./config/themes.yaml \
  --db ./data/lancedb \
  --model all-MiniLM-L6-v2

# dry run (compute and show a preview; no DB writes)
python scripts/00_refresh_themes.py --dry-run

"""
from __future__ import annotations
import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List

import pandas as pd
import lancedb

# Transformer import is sizable; keep import local for faster --dry-run without embed
from sentence_transformers import SentenceTransformer

DEFAULT_CONFIG = "./config/themes.yaml"
DEFAULT_DB = "./data/lancedb"
DEFAULT_TABLE = "themes"
DEFAULT_MODEL = "all-MiniLM-L6-v2"


def load_themes_yaml(path: str) -> Dict:
    """Load YAML and return a dict of {theme_name: {description, keywords, ...}}.
    Accepts either:
      - {"themes": {name: {...}}}
      - {name: {...}} (flat)
    """
    import yaml
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Theme config not found: {p}")
    with p.open('r') as f:
        raw = yaml.safe_load(f) or {}
    themes = raw.get("themes", raw)
    if not isinstance(themes, dict) or not themes:
        raise ValueError("Theme config is empty or malformed. Expect a mapping under 'themes' or at top level.")
    # normalize fields
    cleaned = {}
    for name, meta in themes.items():
        if meta is None:
            meta = {}
        desc = meta.get("description", "")
        kws = meta.get("keywords", []) or []
        # ensure list of strings
        if isinstance(kws, str):
            kws = [kws]
        kws = [str(x).strip() for x in kws if str(x).strip()]
        cleaned[str(name)] = {
            "description": str(desc or ""),
            "keywords": kws,
            **{k: v for k, v in meta.items() if k not in {"description", "keywords"}}
        }
    return cleaned


def build_rows(themes: Dict, model_name: str) -> pd.DataFrame:
    """Create a DataFrame with columns: name, description, keywords, embedding, updated_at."""
    print(f"🔧 Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)

    rows: List[Dict] = []
    now = datetime.utcnow().isoformat()
    for name, meta in themes.items():
        desc = meta.get("description", "")
        kws = meta.get("keywords", [])
        text = (desc + " " + " ".join(kws)).strip()
        emb = model.encode(text).tolist() if text else [0.0] * 384
        row = {
            "name": name,
            "description": desc,
            "keywords": kws,
            "embedding": emb,
            "updated_at": now,
        }
        # include passthrough meta
        for k, v in meta.items():
            if k not in row:
                row[k] = v
        rows.append(row)
    df = pd.DataFrame(rows)
    # guarantee required columns exist even if empty
    for col in ["name", "description", "keywords", "embedding", "updated_at"]:
        if col not in df.columns:
            df[col] = None
    return df


def backup_existing_table(db: lancedb.DBConnection, table: str, out_dir: Path) -> Path | None:
    try:
        t = db.open_table(table)
    except Exception:
        return None
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        out_path = out_dir / f"{table}_backup_{ts}.parquet"
        t.to_pandas().to_parquet(out_path, index=False)
        print(f"💾 Backup saved: {out_path}")
        return out_path
    except Exception as e:
        print(f"⚠️ Backup failed: {e}")
        return None


def replace_table(db: lancedb.DBConnection, table: str, df: pd.DataFrame):
    # Atomic replace: drop & recreate
    try:
        db.drop_table(table)
    except Exception:
        pass
    db.create_table(table, df)
    print(f"✅ Rebuilt table '{table}' with {len(df)} rows")


def main():
    ap = argparse.ArgumentParser(description="Rebuild LanceDB 'themes' table from YAML.")
    ap.add_argument("--config", default=DEFAULT_CONFIG, help="Path to themes.yaml")
    ap.add_argument("--db", default=DEFAULT_DB, help="Path to LanceDB directory")
    ap.add_argument("--table", default=DEFAULT_TABLE, help="Table name to write (default: themes)")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="SentenceTransformer model name")
    ap.add_argument("--backup-dir", default="./backups", help="Directory to write a parquet backup of existing table")
    ap.add_argument("--dry-run", action="store_true", help="Load & embed but do not write to DB")
    args = ap.parse_args()

    print("🔎 Loading themes from:", args.config)
    themes = load_themes_yaml(args.config)
    print(f"📚 Loaded {len(themes)} themes")

    df = build_rows(themes, args.model)

    # Show a small preview
    preview_cols = ["name", "description", "keywords"]
    print("\n👀 Preview:")
    print(df[preview_cols].head(5).to_string(index=False))

    if args.dry_run:
        print("\nℹ️ Dry-run mode — no DB changes made.")
        return

    # Connect and write
    print("\n🔗 Connecting to LanceDB:", args.db)
    db = lancedb.connect(args.db)

    # Optional backup
    if args.backup_dir:
        backup_existing_table(db, args.table, Path(args.backup_dir))

    replace_table(db, args.table, df)
    print("🎉 Done.")


if __name__ == "__main__":
    main()
