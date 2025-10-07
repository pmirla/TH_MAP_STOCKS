# scripts/debug_articles_symbols.py
import lancedb, pandas as pd
db = lancedb.connect("./data/lancedb")
df = db.open_table("stock_articles").to_pandas()
print("Rows:", len(df))
print("Columns:", df.columns.tolist())

# show counts by raw symbol
print("\nTop 50 symbols (raw):")
print(df["symbol"].value_counts().head(50))

# show counts by normalized symbol (strip + upper)
df["_norm_symbol"] = df["symbol"].astype(str).str.strip().str.upper()
print("\nTop 50 symbols (normalized):")
print(df["_norm_symbol"].value_counts().head(50))

# find dirty symbols
dirty = df[df["symbol"] != df["symbol"].astype(str).str.strip()]
print("\nDirty (has leading/trailing spaces):", len(dirty))
print(dirty[["symbol"]].drop_duplicates().head(20))

# check a few of your new ones
check = ["MNPR","FEIM","GFI","PL","HOOD","HUT","IAG","INBX","INDV","RMBS","RKLB",
         "RGTI","QUBT","KTOS","LASR","QBTS","LIF","MDXH","METC","PRPO","NBIS",
         "NBTX","NEPH","PLTR","NGD","OKLO","SUPX","SYM","OUST","XERS","APEI",
         "ATRA","AU","TATT","USAS","UI","BKSY","CDTX","UFG","VRNA","WDC","ALLT",
         "ALDX","XNET","COEP","ADPT","CYD","TGEN","XGN","XMTR"]
missing = set(check) - set(df["_norm_symbol"].unique())
print("\nMissing from normalized set:", sorted(missing)[:50])
