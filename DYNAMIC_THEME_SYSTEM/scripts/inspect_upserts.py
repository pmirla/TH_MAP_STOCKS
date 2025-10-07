#!/usr/bin/env python3
import argparse, pandas as pd, lancedb
from datetime import datetime, timedelta

def recent(df, since_latest, since_iso, last_minutes):
    if df.empty:
        return df
    if 'created_at' not in df.columns:
        return df
    df = df.copy()
    df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')

    if since_latest:
        ts = df['created_at'].max()
        return df[df['created_at'] == ts]
    if since_iso:
        ts = pd.to_datetime(since_iso, errors='coerce')
        return df[df['created_at'] >= ts]
    if last_minutes:
        cutoff = pd.Timestamp.utcnow() - pd.Timedelta(minutes=last_minutes)
        return df[df['created_at'] >= cutoff]
    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--db', default='./data/lancedb')
    ap.add_argument('--symbols', nargs='*', default=['TSLA','AAPL','META'])
    ap.add_argument('--since-latest', action='store_true', default=True)
    ap.add_argument('--since-iso', default=None, help='e.g. 2025-09-21T20:35:00')
    ap.add_argument('--last-minutes', type=int, default=None)
    ap.add_argument('--limit', type=int, default=50)
    args = ap.parse_args()

    db = lancedb.connect(args.db)
    links = db.open_table('article_theme_links').to_pandas()
    agg   = db.open_table('stock_theme_relationships').to_pandas()

    agg_f   = recent(agg,   args.since_latest, args.since_iso, args.last_minutes)
    links_f = recent(links, args.since_latest, args.since_iso, args.last_minutes)

    if args.symbols:
        agg_f   = agg_f[agg_f['symbol'].isin(args.symbols)]
        links_f = links_f[links_f['symbol'].isin(args.symbols)]

    agg_f   = agg_f.sort_values(['symbol','normalized_score'], ascending=[True, False])
    links_f = links_f.sort_values(['symbol','similarity_score'], ascending=[True, False])

    print("\n=== stock_theme_relationships ===")
    if agg_f.empty:
        print("(no rows)")
    else:
        cols = ['symbol','theme_name','normalized_score','article_count','confidence','created_at']
        print(agg_f[cols].head(args.limit).to_string(index=False))

    print("\n=== article_theme_links ===")
    if links_f.empty:
        print("(no rows)")
    else:
        cols = ['symbol','theme_name','similarity_score','article_title','source_domain','created_at']
        print(links_f[cols].head(args.limit).to_string(index=False))

if __name__ == '__main__':
    main()
