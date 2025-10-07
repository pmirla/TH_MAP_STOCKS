#!/usr/bin/env bash
# Smoke test for DYNAMIC_THEME_SYSTEM
# Runs: config checks → ingestion → theme/table checks → scoring/upsert → inspection → sanity checks → matrix export

set -euo pipefail
[[ "${DEBUG:-0}" == "1" ]] && set -x

# ---------- pretty printing ----------
RED=$(printf '\033[31m'); GRN=$(printf '\033[32m'); YLW=$(printf '\033[33m'); BLU=$(printf '\033[34m'); BLD=$(printf '\033[1m'); RST=$(printf '\033[0m')
step()   { echo -e "\n${BLD}${BLU}▶ $*${RST}"; }
ok()     { echo -e "${GRN}✔ $*${RST}"; }
warn()   { echo -e "${YLW}⚠ $*${RST}"; }
fail()   { echo -e "${RED}✖ $*${RST}"; exit 1; }

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
cd "$ROOT"

# ---------- defaults & args ----------
LANCEDB_PATH="${LANCEDB_PATH:-./data/lancedb}"
MIN_SCORE="${MIN_SCORE:-0.30}"
TOP_K="${TOP_K:-5}"
THRESHOLD="${THRESHOLD:-0.25}"
EXPORT_MODE="${EXPORT_MODE:-both}"
OUT_PREFIX="${OUT_PREFIX:-reports/smoke_ats}"
THEME_FOCUS="${THEME_FOCUS:-space_technology}"

CLI_SYMBOLS=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --symbols)
      shift
      # collect non-flag tokens as symbols
      while [[ $# -gt 0 && "$1" != --* ]]; do
        CLI_SYMBOLS+="${1} "
        shift
      done
      ;;
    --min-score)   MIN_SCORE="${2:-$MIN_SCORE}"; shift 2;;
    --top-k)       TOP_K="${2:-$TOP_K}"; shift 2;;
    --threshold)   THRESHOLD="${2:-$THRESHOLD}"; shift 2;;
    --export)      EXPORT_MODE="${2:-$EXPORT_MODE}"; shift 2;;
    --out-prefix)  OUT_PREFIX="${2:-$OUT_PREFIX}"; shift 2;;
    --theme-focus) THEME_FOCUS="${2:-$THEME_FOCUS}"; shift 2;;
    -h|--help)
      cat <<EOF
Usage: scripts/smoke_test.sh [--symbols "AAPL META TSLA"] [--min-score 0.30] [--top-k 5] [--threshold 0.25]
                            [--export csv|json|both|none] [--out-prefix reports/whatever] [--theme-focus NAME]
Also respects env: SYMBOLS="AAPL,TSLA", LANCEDB_PATH, MIN_SCORE, TOP_K, THRESHOLD, EXPORT_MODE, OUT_PREFIX, THEME_FOCUS
EOF
      exit 0;;
    *) warn "Unknown arg: $1"; shift;;
  esac
done

if [[ -n "$CLI_SYMBOLS" ]]; then
  SYMBOLS_SP=$(echo "$CLI_SYMBOLS" | tr ',' ' ' | xargs)
elif [[ -n "${SYMBOLS:-}" ]]; then
  SYMBOLS_SP=$(echo "$SYMBOLS" | tr ',' ' ' | xargs)
else
  SYMBOLS_SP="AAPL META TSLA"
fi
read -r -a SYMBOL_ARR <<<"$SYMBOLS_SP"

# Make LANCEDB_PATH visible to all Python calls
export LANCEDB_PATH

# ---------- step 0: quick env info ----------
step "Environment"
echo "Repo:               $ROOT"
echo "Python:             $(command -v python || true)"
echo "LanceDB path:       $LANCEDB_PATH"
echo "Symbols:            ${SYMBOL_ARR[*]}"
echo "Threshold:          $THRESHOLD"
echo "Min score:          $MIN_SCORE"
echo "Top-K:              $TOP_K"
echo "Export mode:        $EXPORT_MODE"
echo "Out prefix:         $OUT_PREFIX"
echo "Theme focus:        $THEME_FOCUS"
mkdir -p "$LANCEDB_PATH" reports logs

# ---------- step 1: config presence + counts ----------
step "Check config files & counts"
[[ -f config/tickers.yaml ]] || fail "Missing config/tickers.yaml"
[[ -f config/themes.yaml  ]] || warn "Missing config/themes.yaml (theme sync will be skipped if 01_ingester doesn't handle it)"

python - <<'PY' || fail "Config parse failed"
import yaml, pathlib
tick=pathlib.Path("config/tickers.yaml")
thm =pathlib.Path("config/themes.yaml")
with open(tick) as f: t=yaml.safe_load(f)
print(f"tickers.yaml: {len(t.get('tickers', []))} tickers")
if thm.exists():
    with open(thm) as f: y=yaml.safe_load(f) or {}
    themes = (y.get('themes') or y) if isinstance(y, dict) else {}
    print(f"themes.yaml:  {len(themes)} themes")
else:
    print("themes.yaml:  (absent)")
PY
ok "Config OK"

# ---------- step 2: run incremental ingestion (stocks + articles + theme sync) ----------
step "Run 01_incremental_data_ingestion.py"
python scripts/01_incremental_data_ingestion.py || fail "Incremental ingestion failed"
ok "Ingestion completed"

# ---------- step 3: verify LanceDB core tables ----------
step "Verify LanceDB tables exist (enhanced_stocks_dynamic, stock_articles, themes)"
python - <<'PY' || fail "LanceDB verification failed"
import os, lancedb, pandas as pd
db=lancedb.connect(os.environ.get("LANCEDB_PATH","./data/lancedb"))
names=set(db.table_names())
need={"enhanced_stocks_dynamic","stock_articles"}
missing=[n for n in need if n not in names]
if missing: raise SystemExit(f"Missing tables: {missing}")
print("Tables present:", ", ".join(sorted(names)))
for t in sorted(need|({"themes"} if "themes" in names else set())):
    try:
        df = db.open_table(t).to_pandas()
        print(f"{t}: {len(df)} rows")
    except Exception as e:
        print(f"{t}: error {e}")
PY
ok "Tables verified"

# ---------- step 4: scoring + upsert + exports ----------
step "Run 02_article_theme_scoring.py (upsert + exports)"
SCORER="scripts/02_article_theme_scoring.py"

python "$SCORER" \
  --symbols "${SYMBOL_ARR[@]}" \
  --threshold "$THRESHOLD" \
  --min-score "$MIN_SCORE" \
  --top-k "$TOP_K" \
  --export "$EXPORT_MODE" \
  --out-prefix "$OUT_PREFIX" || fail "Scoring failed"
ok "Scoring & upsert completed"

# ---------- step 5: inspect upserts ----------
step "Inspect upserts for selected symbols"
python scripts/inspect_upserts.py --symbols "${SYMBOL_ARR[@]}" || warn "inspect_upserts.py not found or failed"
ok "Upsert inspection printed above"

# ---------- step 6: sanity checks (cosine vs agg) ----------
step "Sanity check: cosine vs aggregated (theme focus: $THEME_FOCUS)"
if [[ -f scripts/sanity_check_themes.py ]]; then
  python scripts/sanity_check_themes.py --symbols "${SYMBOL_ARR[@]}" --theme "$THEME_FOCUS" --top-k "$TOP_K" || warn "Sanity check failed"
  ok "Sanity check printed above"
else
  warn "scripts/sanity_check_themes.py not found; skipping sanity checks"
fi

# ---------- step 7: export full matrix from DB ----------
step "Export full stock_theme matrix from DB (latest snapshot)"
if [[ -f scripts/quick_export_matrix_from_db.py ]]; then
  python scripts/quick_export_matrix_from_db.py || warn "quick_export_matrix_from_db.py failed"
else
  # inline tiny exporter
  python - <<'PY' || warn "inline matrix export failed"
import os, pandas as pd, lancedb, pathlib
pathlib.Path("reports").mkdir(exist_ok=True)
db = lancedb.connect(os.environ.get("LANCEDB_PATH","./data/lancedb"))
try:
    df = db.open_table("stock_theme_relationships").to_pandas()
    if df.empty:
        print("stock_theme_relationships is empty; nothing to export")
    else:
        mat = df.pivot(index="symbol", columns="theme_name", values="normalized_score").fillna(0.0)
        out = "reports/stock_theme_matrix_from_db.csv"
        mat.to_csv(out)
        print("Saved", out, mat.shape)
except Exception as e:
    print("Export error:", e)
PY
fi
ok "Matrix export attempted"

# ---------- wrap ----------
step "Smoke test finished"
echo "Artifacts (look under ./reports):"
ls -1 reports 2>/dev/null | sed 's/^/  • /' || true
ok "All done 🎉"
