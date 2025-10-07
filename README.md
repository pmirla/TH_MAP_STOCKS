# TH_MAP_STOCKS




tickers , yaml file edit 


python scripts/01_incremental_data_ingestion.py  


 export SYMBOLS="NVDA,TSLA,ZS,PLTR,ABBV"  



python scripts/02_article_theme_scoring.py \
  --threshold 0.25 \
  --export none


python scripts/diagnostic_full_cosine_matrix.py     




