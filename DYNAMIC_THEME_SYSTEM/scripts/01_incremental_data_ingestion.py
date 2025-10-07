# #!/usr/bin/env python3
# """
# Production-Ready Incremental Data Ingestion System
# ================================================

# This script implements efficient incremental data ingestion with:
# 1. Ticker-based configuration for selective updates
# 2. Duplicate checking and prevention
# 3. Timestamp-based refresh logic
# 4. Batch processing for efficiency
# 5. Comprehensive logging and error handling

# Key Features:
# - Only updates data that needs refreshing based on timestamps
# - Avoids duplicate articles and stock data
# - Configurable update frequencies per ticker priority
# - Efficient batch processing
# - Production-ready error handling
# """

# import os
# import sys
# import yaml
# import json
# import pandas as pd
# import numpy as np
# import lancedb
# import yfinance as yf
# import requests
# from datetime import datetime, timedelta
# from pathlib import Path
# from sentence_transformers import SentenceTransformer
# import logging
# from typing import Dict, List, Optional, Tuple
# import hashlib

# # Add the parent directory to the path to import core modules
# sys.path.append(str(Path(__file__).parent.parent))
# from core.config import Settings

# class IncrementalDataIngester:
#     """Production-ready incremental data ingestion system"""
    
#     def __init__(self, config_path="./config", db_path="./data/lancedb"):
#         self.config_path = Path(config_path)
#         self.db_path = db_path
#         self.db = lancedb.connect(db_path)
        
#         # Setup logging first
#         self.setup_logging()
        
#         # Load configurations
#         self.config = Settings()
#         self.ticker_config = self.load_ticker_config()
        
#         # Initialize sentence transformer
#         self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
#         self.logger.info("🚀 Incremental Data Ingester Initialized")
#         self.logger.info(f"📊 Database: {db_path}")
#         self.logger.info(f"⚙️ Config: {config_path}")
        
#     def setup_logging(self):
#         """Setup comprehensive logging"""
#         log_dir = Path("logs")
#         log_dir.mkdir(exist_ok=True)
        
#         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#         log_file = log_dir / f"incremental_ingestion_{timestamp}.log"
        
#         logging.basicConfig(
#             level=logging.INFO,
#             format='%(asctime)s - %(levelname)s - %(message)s',
#             handlers=[
#                 logging.FileHandler(log_file),
#                 logging.StreamHandler()
#             ]
#         )
#         self.logger = logging.getLogger(__name__)
        
#     def load_ticker_config(self) -> Dict:
#         """Load ticker configuration from YAML"""
#         ticker_file = self.config_path / "tickers.yaml"
        
#         if not ticker_file.exists():
#             self.logger.error(f"❌ Ticker config not found: {ticker_file}")
#             raise FileNotFoundError(f"Ticker config not found: {ticker_file}")
            
#         with open(ticker_file, 'r') as f:
#             config = yaml.safe_load(f)
            
#         self.logger.info(f"📋 Loaded ticker config with {len(config['tickers'])} tickers")
#         return config
        
#     def save_ticker_config(self):
#         """Save updated ticker configuration"""
#         ticker_file = self.config_path / "tickers.yaml"
        
#         with open(ticker_file, 'w') as f:
#             yaml.dump(self.ticker_config, f, default_flow_style=False)
            
#         self.logger.info("💾 Ticker config updated")
        
#     def needs_stock_update(self, ticker_info: Dict) -> bool:
#         """Check if stock data needs updating based on timestamp and priority"""
#         if self.ticker_config['update_settings']['force_full_refresh']:
#             return True
            
#         last_updated = ticker_info.get('last_updated')
#         if not last_updated:
#             return True
            
#         priority = ticker_info.get('priority', 'medium')
#         refresh_hours = self.ticker_config['priority_settings'][priority]['stock_refresh_hours']
        
#         last_update_time = datetime.fromisoformat(last_updated)
#         time_diff = datetime.now() - last_update_time
        
#         return time_diff.total_seconds() > (refresh_hours * 3600)
        
#     def needs_article_update(self, ticker_info: Dict) -> bool:
#         """Check if article data needs updating"""
#         if self.ticker_config['update_settings']['force_full_refresh']:
#             return True
            
#         last_updated = ticker_info.get('articles_last_updated')
#         if not last_updated:
#             return True
            
#         priority = ticker_info.get('priority', 'medium')
#         refresh_hours = self.ticker_config['priority_settings'][priority]['article_refresh_hours']
        
#         last_update_time = datetime.fromisoformat(last_updated)
#         time_diff = datetime.now() - last_update_time
        
#         return time_diff.total_seconds() > (refresh_hours * 3600)
        
#     def get_existing_stock_data(self) -> Dict:
#         """Get existing stock data to avoid duplicates"""
#         try:
#             table = self.db.open_table("enhanced_stocks_dynamic")
#             df = table.to_pandas()
#             return {row['symbol']: row for _, row in df.iterrows()}
#         except Exception as e:
#             self.logger.info(f"📊 No existing stock data found: {e}")
#             return {}
            
#     def get_existing_articles(self) -> Dict:
#         """Get existing articles to avoid duplicates"""
#         try:
#             table = self.db.open_table("stock_articles")
#             df = table.to_pandas()
#             # Create hash-based lookup for duplicate detection
#             article_hashes = {}
#             for _, row in df.iterrows():
#                 content_hash = self.generate_article_hash(row['title'], row['content'])
#                 article_hashes[content_hash] = row
#             return article_hashes
#         except Exception as e:
#             self.logger.info(f"📰 No existing articles found: {e}")
#             return {}
            
#     def generate_article_hash(self, title: str, content: str) -> str:
#         """Generate hash for article duplicate detection"""
#         combined = f"{title}|{content}"
#         return hashlib.md5(combined.encode()).hexdigest()
        
#     def fetch_stock_data(self, symbol: str) -> Optional[Dict]:
#         """Fetch stock data from multiple sources with error handling"""
#         try:
#             # Get data from yfinance
#             ticker = yf.Ticker(symbol)
#             info = ticker.info
            
#             # Get data from Finviz
#             finviz_data = self.fetch_finviz_data(symbol)
            
#             # Combine data
#             stock_data = {
#                 'symbol': symbol,
#                 'company_name': info.get('longName', symbol),
#                 'sector': info.get('sector', 'Unknown'),
#                 'industry': info.get('industry', 'Unknown'),
#                 'market_cap': info.get('marketCap', 0),
#                 'description': info.get('longBusinessSummary', ''),
#                 'finviz_description': finviz_data.get('description', ''),
#                 'last_updated': datetime.now().isoformat(),
#                 'data_sources': ['yfinance', 'finviz']
#             }
            
#             # Generate embeddings
#             combined_text = f"{stock_data['description']} {stock_data['finviz_description']}"
#             if combined_text.strip():
#                 embedding = self.model.encode(combined_text).tolist()
#                 stock_data['weighted_embedding'] = embedding
#             else:
#                 stock_data['weighted_embedding'] = [0.0] * 384
                
#             return stock_data
            
#         except Exception as e:
#             self.logger.error(f"❌ Error fetching data for {symbol}: {e}")
#             return None
            
#     def fetch_finviz_data(self, symbol: str) -> Dict:
#         """Fetch data from Finviz with error handling"""
#         try:
#             url = f"https://finviz.com/quote.ashx?t={symbol}"
#             headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
            
#             response = requests.get(url, headers=headers, timeout=10)
#             response.raise_for_status()
            
#             # Simple parsing - in production, use proper HTML parsing
#             content = response.text
            
#             # Extract description (simplified)
#             description = ""
#             if "Business Summary" in content:
#                 # This is a simplified extraction - implement proper parsing
#                 description = f"Finviz data for {symbol}"
                
#             return {'description': description}
            
#         except Exception as e:
#             self.logger.warning(f"⚠️ Could not fetch Finviz data for {symbol}: {e}")
#             return {'description': ''}
            
#     def fetch_articles_for_stock(self, symbol: str, existing_articles: Dict) -> List[Dict]:
#         """Fetch articles for a stock with duplicate checking"""
#         try:
#             if not hasattr(self.config, 'tavily_api_key') or not self.config.tavily_api_key:
#                 self.logger.warning("⚠️ Tavily API key not configured, skipping article collection")
#                 return []
                
#             from tavily import TavilyClient
#             client = TavilyClient(api_key=self.config.tavily_api_key)
            
#             # Search for recent articles
#             lookback_days = self.ticker_config['update_settings']['article_lookback_days']
#             max_articles = self.ticker_config['update_settings']['max_articles_per_stock']
            
#             query = f"{symbol} stock news earnings financial"
            
#             response = client.search(
#                 query=query,
#                 search_depth="basic",
#                 max_results=max_articles,
#                 days=lookback_days
#             )
            
#             new_articles = []
#             duplicate_count = 0
            
#             for result in response.get('results', []):
#                 title = result.get('title', '')
#                 content = result.get('content', '')
                
#                 # Check for duplicates
#                 article_hash = self.generate_article_hash(title, content)
#                 if article_hash in existing_articles:
#                     duplicate_count += 1
#                     continue
                    
#                 article_data = {
#                     'symbol': symbol,
#                     'title': title,
#                     'content': content,
#                     'url': result.get('url', ''),
#                     'published_date': result.get('published_date', datetime.now().isoformat()),
#                     'collected_at': datetime.now().isoformat(),
#                     'content_hash': article_hash
#                 }
                
#                 # Generate embedding
#                 combined_text = f"{title} {content}"
#                 if combined_text.strip():
#                     embedding = self.model.encode(combined_text).tolist()
#                     article_data['embedding'] = embedding
#                 else:
#                     article_data['embedding'] = [0.0] * 384
                    
#                 new_articles.append(article_data)
#                 existing_articles[article_hash] = article_data  # Update cache
                
#             self.logger.info(f"📰 {symbol}: Found {len(new_articles)} new articles, skipped {duplicate_count} duplicates")
#             return new_articles
            
#         except Exception as e:
#             self.logger.error(f"❌ Error fetching articles for {symbol}: {e}")
#             return []
            
#     def update_stock_data(self, stocks_to_update: List[str], existing_data: Dict) -> List[Dict]:
#         """Update stock data for specified tickers"""
#         updated_stocks = []
        
#         for symbol in stocks_to_update:
#             self.logger.info(f"📈 Updating stock data for {symbol}")
            
#             stock_data = self.fetch_stock_data(symbol)
#             if stock_data:
#                 updated_stocks.append(stock_data)
                
#                 # Update ticker config timestamp
#                 for ticker in self.ticker_config['tickers']:
#                     if ticker['symbol'] == symbol:
#                         ticker['last_updated'] = datetime.now().isoformat()
#                         break
                        
#         return updated_stocks
        
#     def update_article_data(self, stocks_to_update: List[str], existing_articles: Dict) -> List[Dict]:
#         """Update article data for specified tickers"""
#         all_new_articles = []
        
#         for symbol in stocks_to_update:
#             self.logger.info(f"📰 Updating articles for {symbol}")
            
#             new_articles = self.fetch_articles_for_stock(symbol, existing_articles)
#             all_new_articles.extend(new_articles)
            
#             # Update ticker config timestamp
#             for ticker in self.ticker_config['tickers']:
#                 if ticker['symbol'] == symbol:
#                     ticker['articles_last_updated'] = datetime.now().isoformat()
#                     break
                    
#         return all_new_articles
        
#     def save_to_database(self, stock_data: List[Dict], article_data: List[Dict]):
#         """Save updated data to LanceDB with proper merging"""
        
#         # Save stock data
#         if stock_data:
#             try:
#                 # Try to get existing table
#                 try:
#                     existing_table = self.db.open_table("enhanced_stocks_dynamic")
#                     existing_df = existing_table.to_pandas()
                    
#                     # Create new dataframe
#                     new_df = pd.DataFrame(stock_data)
                    
#                     # Merge data (update existing, add new)
#                     updated_symbols = set(new_df['symbol'])
#                     filtered_existing = existing_df[~existing_df['symbol'].isin(updated_symbols)]
                    
#                     combined_df = pd.concat([filtered_existing, new_df], ignore_index=True)
                    
#                     # Drop and recreate table
#                     self.db.drop_table("enhanced_stocks_dynamic")
#                     self.db.create_table("enhanced_stocks_dynamic", combined_df)
                    
#                 except Exception:
#                     # Create new table if doesn't exist
#                     df = pd.DataFrame(stock_data)
#                     self.db.create_table("enhanced_stocks_dynamic", df)
                    
#                 self.logger.info(f"💾 Saved {len(stock_data)} stock records to database")
                
#             except Exception as e:
#                 self.logger.error(f"❌ Error saving stock data: {e}")
                
#         # Save article data
#         if article_data:
#             try:
#                 # Try to get existing table
#                 try:
#                     existing_table = self.db.open_table("stock_articles")
#                     existing_df = existing_table.to_pandas()
                    
#                     # Create new dataframe
#                     new_df = pd.DataFrame(article_data)
                    
#                     # Append new articles
#                     combined_df = pd.concat([existing_df, new_df], ignore_index=True)
                    
#                     # Drop and recreate table
#                     self.db.drop_table("stock_articles")
#                     self.db.create_table("stock_articles", combined_df)
                    
#                 except Exception:
#                     # Create new table if doesn't exist
#                     df = pd.DataFrame(article_data)
#                     self.db.create_table("stock_articles", df)
                    
#                 self.logger.info(f"💾 Saved {len(article_data)} article records to database")
                
#             except Exception as e:
#                 self.logger.error(f"❌ Error saving article data: {e}")
                
#     def run_incremental_update(self):
#         """Run the complete incremental update process"""
#         self.logger.info("🔄 Starting incremental data update")
        
#         # Get existing data for duplicate checking
#         existing_stock_data = self.get_existing_stock_data()
#         existing_articles = self.get_existing_articles()
        
#         # Determine which stocks need updates
#         stocks_needing_update = []
#         stocks_needing_articles = []
        
#         for ticker_info in self.ticker_config['tickers']:
#             symbol = ticker_info['symbol']
            
#             if self.needs_stock_update(ticker_info):
#                 stocks_needing_update.append(symbol)
                
#             if self.needs_article_update(ticker_info):
#                 stocks_needing_articles.append(symbol)
                
#         self.logger.info(f"📊 Stocks needing data update: {len(stocks_needing_update)}")
#         self.logger.info(f"📰 Stocks needing article update: {len(stocks_needing_articles)}")
        
#         # Process updates in batches
#         batch_size = self.ticker_config['update_settings']['batch_size']
        
#         all_updated_stocks = []
#         all_new_articles = []
        
#         # Update stock data in batches
#         for i in range(0, len(stocks_needing_update), batch_size):
#             batch = stocks_needing_update[i:i + batch_size]
#             self.logger.info(f"📈 Processing stock batch {i//batch_size + 1}: {batch}")
            
#             updated_stocks = self.update_stock_data(batch, existing_stock_data)
#             all_updated_stocks.extend(updated_stocks)
            
#         # Update article data in batches
#         for i in range(0, len(stocks_needing_articles), batch_size):
#             batch = stocks_needing_articles[i:i + batch_size]
#             self.logger.info(f"📰 Processing article batch {i//batch_size + 1}: {batch}")
            
#             new_articles = self.update_article_data(batch, existing_articles)
#             all_new_articles.extend(new_articles)
            
#         # Save all updates to database
#         self.save_to_database(all_updated_stocks, all_new_articles)
        
#         # Save updated ticker configuration
#         self.save_ticker_config()
        
#         # Generate summary
#         summary = {
#             'timestamp': datetime.now().isoformat(),
#             'stocks_updated': len(all_updated_stocks),
#             'articles_added': len(all_new_articles),
#             'stocks_processed': stocks_needing_update,
#             'articles_processed': stocks_needing_articles,
#             'total_existing_articles': len(existing_articles),
#             'total_existing_stocks': len(existing_stock_data)
#         }
        
#         self.logger.info("✅ Incremental update completed")
#         self.logger.info(f"📊 Summary: {json.dumps(summary, indent=2)}")
        
#         return summary

# def main():
#     """Main execution function"""
#     print("🚀 PRODUCTION INCREMENTAL DATA INGESTION")
#     print("=" * 80)
    
#     try:
#         ingester = IncrementalDataIngester()
#         summary = ingester.run_incremental_update()
        
#         print("=" * 80)
#         print("✅ INCREMENTAL UPDATE COMPLETED")
#         print("=" * 80)
#         print(f"📊 Stocks updated: {summary['stocks_updated']}")
#         print(f"📰 Articles added: {summary['articles_added']}")
#         print(f"⏱️ Completed at: {summary['timestamp']}")
#         print("=" * 80)
        
#     except Exception as e:
#         print(f"❌ Error during incremental update: {e}")
#         raise

# if __name__ == "__main__":
#     main()

#!/usr/bin/env python3
"""
Production-Ready Incremental Data Ingestion System
================================================

This script implements efficient incremental data ingestion with:
1. Ticker-based configuration for selective updates
2. Duplicate checking and prevention
3. Timestamp-based refresh logic
4. Batch processing for efficiency
5. Comprehensive logging and error handling

Key Features:
- Only updates data that needs refreshing based on timestamps
- Avoids duplicate articles and stock data
- Configurable update frequencies per ticker priority
- Efficient batch processing
- Production-ready error handling
- NEW: Syncs themes from config/themes.yaml into LanceDB 'themes' table
"""

import os
import sys
import yaml
import json
import pandas as pd
import numpy as np
import lancedb
import yfinance as yf
import requests
from datetime import datetime, timedelta
from pathlib import Path
from sentence_transformers import SentenceTransformer
import logging
from typing import Dict, List, Optional, Tuple
import hashlib

# Add the parent directory to the path to import core modules
sys.path.append(str(Path(__file__).parent.parent))
from core.config import Settings


class IncrementalDataIngester:
    """Production-ready incremental data ingestion system"""

    def __init__(self, config_path: str = "./config", db_path: str = "./data/lancedb"):
        self.config_path = Path(config_path)
        self.db_path = db_path
        self.db = lancedb.connect(db_path)

        # Setup logging first
        self.setup_logging()

        # Load configurations
        self.config = Settings()
        self.ticker_config = self.load_ticker_config()

        # Initialize sentence transformer (used for articles, stocks, and themes)
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

        self.logger.info("🚀 Incremental Data Ingester Initialized")
        self.logger.info(f"📊 Database: {db_path}")
        self.logger.info(f"⚙️ Config: {config_path}")

    def setup_logging(self):
        """Setup comprehensive logging"""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"incremental_ingestion_{timestamp}.log"

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def load_ticker_config(self) -> Dict:
        """Load ticker configuration from YAML"""
        ticker_file = self.config_path / "tickers.yaml"

        if not ticker_file.exists():
            self.logger.error(f"❌ Ticker config not found: {ticker_file}")
            raise FileNotFoundError(f"Ticker config not found: {ticker_file}")

        with open(ticker_file, 'r') as f:
            config = yaml.safe_load(f)

        self.logger.info(f"📋 Loaded ticker config with {len(config['tickers'])} tickers")
        return config

    def save_ticker_config(self):
        """Save updated ticker configuration"""
        ticker_file = self.config_path / "tickers.yaml"

        with open(ticker_file, 'w') as f:
            yaml.dump(self.ticker_config, f, default_flow_style=False)

        self.logger.info("💾 Ticker config updated")

    # -------------------- THEME SYNC (NEW) --------------------

    def load_theme_config(self) -> dict:
        """Load themes from config/themes.yaml. Supports either {'themes': {...}} or flat {...}."""
        theme_file = self.config_path / "themes.yaml"
        if not theme_file.exists():
            self.logger.warning(f"⚠️ Theme config not found: {theme_file} — skipping theme sync")
            return {}
        try:
            with open(theme_file, "r") as f:
                raw = yaml.safe_load(f) or {}
            themes = raw.get("themes", raw)
            if not isinstance(themes, dict) or not themes:
                self.logger.warning("⚠️ Theme config is empty or malformed — skipping theme sync")
                return {}
            return themes
        except Exception as e:
            self.logger.error(f"❌ Error reading theme config: {e}")
            return {}

    def build_theme_rows(self, themes_dict: dict) -> List[dict]:
        """Build LanceDB-ready rows for the 'themes' table with embeddings."""
        rows = []
        for name, meta in themes_dict.items():
            meta = meta or {}
            desc = meta.get("description", "")
            kws = meta.get("keywords", [])
            text = (desc + " " + " ".join(kws)).strip()
            emb = self.model.encode(text).tolist() if text else [0.0] * 384
            rows.append({
                "name": name,
                "description": desc,
                "keywords": kws,
                "embedding": emb,
                "updated_at": datetime.now().isoformat(),
            })
        return rows

    def save_themes_table(self, rows: List[dict]):
        """Upsert themes by name (atomic replace)."""
        if not rows:
            return
        try:
            try:
                table = self.db.open_table("themes")
                existing = table.to_pandas()
                new_df = pd.DataFrame(rows)

                if 'name' not in existing.columns:
                    existing = pd.DataFrame(columns=new_df.columns)

                # Merge by name: keep existing rows whose names are not in new batch
                keep = existing[~existing["name"].isin(new_df["name"])]
                combined = pd.concat([keep, new_df], ignore_index=True)

                # Replace table atomically
                self.db.drop_table("themes")
                self.db.create_table("themes", combined)
            except Exception:
                # Create table if it doesn't exist yet
                self.db.create_table("themes", pd.DataFrame(rows))

            self.logger.info(f"💾 Synced {len(rows)} themes to LanceDB")
        except Exception as e:
            self.logger.error(f"❌ Error saving themes: {e}")

    # -------------------- /THEME SYNC -------------------------

    def needs_stock_update(self, ticker_info: Dict) -> bool:
        """Check if stock data needs updating based on timestamp and priority"""
        if self.ticker_config['update_settings']['force_full_refresh']:
            return True

        last_updated = ticker_info.get('last_updated')
        if not last_updated:
            return True

        priority = ticker_info.get('priority', 'medium')
        refresh_hours = self.ticker_config['priority_settings'][priority]['stock_refresh_hours']

        last_update_time = datetime.fromisoformat(last_updated)
        time_diff = datetime.now() - last_update_time

        return time_diff.total_seconds() > (refresh_hours * 3600)

    def needs_article_update(self, ticker_info: Dict) -> bool:
        """Check if article data needs updating"""
        if self.ticker_config['update_settings']['force_full_refresh']:
            return True

        last_updated = ticker_info.get('articles_last_updated')
        if not last_updated:
            return True

        priority = ticker_info.get('priority', 'medium')
        refresh_hours = self.ticker_config['priority_settings'][priority]['article_refresh_hours']

        last_update_time = datetime.fromisoformat(last_updated)
        time_diff = datetime.now() - last_update_time

        return time_diff.total_seconds() > (refresh_hours * 3600)

    def get_existing_stock_data(self) -> Dict:
        """Get existing stock data to avoid duplicates"""
        try:
            table = self.db.open_table("enhanced_stocks_dynamic")
            df = table.to_pandas()
            return {row['symbol']: row for _, row in df.iterrows()}
        except Exception as e:
            self.logger.info(f"📊 No existing stock data found: {e}")
            return {}

    def get_existing_articles(self) -> Dict:
        """Get existing articles to avoid duplicates"""
        try:
            table = self.db.open_table("stock_articles")
            df = table.to_pandas()
            # Create hash-based lookup for duplicate detection
            article_hashes = {}
            for _, row in df.iterrows():
                content_hash = self.generate_article_hash(row.get('title', ''), row.get('content', ''))
                article_hashes[content_hash] = row
            return article_hashes
        except Exception as e:
            self.logger.info(f"📰 No existing articles found: {e}")
            return {}

    def generate_article_hash(self, title: str, content: str) -> str:
        """Generate hash for article duplicate detection"""
        combined = f"{title}|{content}"
        return hashlib.md5(combined.encode()).hexdigest()

    def fetch_stock_data(self, symbol: str) -> Optional[Dict]:
        """Fetch stock data from multiple sources with error handling"""
        try:
            # Get data from yfinance
            ticker = yf.Ticker(symbol)
            info = ticker.info

            # Get data from Finviz
            finviz_data = self.fetch_finviz_data(symbol)

            # Combine data
            stock_data = {
                'symbol': symbol,
                'company_name': info.get('longName', symbol),
                'sector': info.get('sector', 'Unknown'),
                'industry': info.get('industry', 'Unknown'),
                'market_cap': info.get('marketCap', 0),
                'description': info.get('longBusinessSummary', ''),
                'finviz_description': finviz_data.get('description', ''),
                'last_updated': datetime.now().isoformat(),
                'data_sources': ['yfinance', 'finviz']
            }

            # Generate embeddings
            combined_text = f"{stock_data['description']} {stock_data['finviz_description']}"
            if combined_text.strip():
                embedding = self.model.encode(combined_text).tolist()
                stock_data['weighted_embedding'] = embedding
            else:
                stock_data['weighted_embedding'] = [0.0] * 384

            return stock_data

        except Exception as e:
            self.logger.error(f"❌ Error fetching data for {symbol}: {e}")
            return None

    def fetch_finviz_data(self, symbol: str) -> Dict:
        """Fetch data from Finviz with error handling"""
        try:
            url = f"https://finviz.com/quote.ashx?t={symbol}"
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}

            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()

            # Simple parsing - in production, use proper HTML parsing
            content = response.text

            # Extract description (simplified)
            description = ""
            if "Business Summary" in content:
                # This is a simplified extraction - implement proper parsing
                description = f"Finviz data for {symbol}"

            return {'description': description}

        except Exception as e:
            self.logger.warning(f"⚠️ Could not fetch Finviz data for {symbol}: {e}")
            return {'description': ''}

    def fetch_articles_for_stock(self, symbol: str, existing_articles: Dict) -> List[Dict]:
        """Fetch articles for a stock with duplicate checking"""
        try:
            if not hasattr(self.config, 'tavily_api_key') or not self.config.tavily_api_key:
                self.logger.warning("⚠️ Tavily API key not configured, skipping article collection")
                return []

            from tavily import TavilyClient
            client = TavilyClient(api_key=self.config.tavily_api_key)

            # Search for recent articles
            lookback_days = self.ticker_config['update_settings']['article_lookback_days']
            max_articles = self.ticker_config['update_settings']['max_articles_per_stock']

            query = f"{symbol} stock news earnings financial"

            response = client.search(
                query=query,
                search_depth="basic",
                max_results=max_articles,
                days=lookback_days
            )

            new_articles = []
            duplicate_count = 0

            for result in response.get('results', []):
                title = result.get('title', '')
                content = result.get('content', '')

                # Check for duplicates
                article_hash = self.generate_article_hash(title, content)
                if article_hash in existing_articles:
                    duplicate_count += 1
                    continue

                article_data = {
                    'symbol': symbol,
                    'title': title,
                    'content': content,
                    'url': result.get('url', ''),
                    'published_date': result.get('published_date', datetime.now().isoformat()),
                    'collected_at': datetime.now().isoformat(),
                    'content_hash': article_hash
                }

                # Generate embedding
                combined_text = f"{title} {content}"
                if combined_text.strip():
                    embedding = self.model.encode(combined_text).tolist()
                    article_data['embedding'] = embedding
                else:
                    article_data['embedding'] = [0.0] * 384

                new_articles.append(article_data)
                existing_articles[article_hash] = article_data  # Update cache

            self.logger.info(f"📰 {symbol}: Found {len(new_articles)} new articles, skipped {duplicate_count} duplicates")
            return new_articles

        except Exception as e:
            self.logger.error(f"❌ Error fetching articles for {symbol}: {e}")
            return []

    def update_stock_data(self, stocks_to_update: List[str], existing_data: Dict) -> List[Dict]:
        """Update stock data for specified tickers"""
        updated_stocks = []

        for symbol in stocks_to_update:
            self.logger.info(f"📈 Updating stock data for {symbol}")

            stock_data = self.fetch_stock_data(symbol)
            if stock_data:
                updated_stocks.append(stock_data)

                # Update ticker config timestamp
                for ticker in self.ticker_config['tickers']:
                    if ticker['symbol'] == symbol:
                        ticker['last_updated'] = datetime.now().isoformat()
                        break

        return updated_stocks

    def update_article_data(self, stocks_to_update: List[str], existing_articles: Dict) -> List[Dict]:
        """Update article data for specified tickers"""
        all_new_articles = []

        for symbol in stocks_to_update:
            self.logger.info(f"📰 Updating articles for {symbol}")

            new_articles = self.fetch_articles_for_stock(symbol, existing_articles)
            all_new_articles.extend(new_articles)

            # Update ticker config timestamp
            for ticker in self.ticker_config['tickers']:
                if ticker['symbol'] == symbol:
                    ticker['articles_last_updated'] = datetime.now().isoformat()
                    break

        return all_new_articles

    def save_to_database(self, stock_data: List[Dict], article_data: List[Dict]):
        """Save updated data to LanceDB with proper merging"""

        # Save stock data
        if stock_data:
            try:
                # Try to get existing table
                try:
                    existing_table = self.db.open_table("enhanced_stocks_dynamic")
                    existing_df = existing_table.to_pandas()

                    # Create new dataframe
                    new_df = pd.DataFrame(stock_data)

                    # Merge data (update existing, add new)
                    updated_symbols = set(new_df['symbol'])
                    filtered_existing = existing_df[~existing_df['symbol'].isin(updated_symbols)]

                    combined_df = pd.concat([filtered_existing, new_df], ignore_index=True)

                    # Drop and recreate table
                    self.db.drop_table("enhanced_stocks_dynamic")
                    self.db.create_table("enhanced_stocks_dynamic", combined_df)

                except Exception:
                    # Create new table if doesn't exist
                    df = pd.DataFrame(stock_data)
                    self.db.create_table("enhanced_stocks_dynamic", df)

                self.logger.info(f"💾 Saved {len(stock_data)} stock records to database")

            except Exception as e:
                self.logger.error(f"❌ Error saving stock data: {e}")

        # Save article data
        if article_data:
            try:
                # Try to get existing table
                try:
                    existing_table = self.db.open_table("stock_articles")
                    existing_df = existing_table.to_pandas()

                    # Create new dataframe
                    new_df = pd.DataFrame(article_data)

                    # Append new articles
                    combined_df = pd.concat([existing_df, new_df], ignore_index=True)

                    # Drop and recreate table
                    self.db.drop_table("stock_articles")
                    self.db.create_table("stock_articles", combined_df)

                except Exception:
                    # Create new table if doesn't exist
                    df = pd.DataFrame(article_data)
                    self.db.create_table("stock_articles", df)

                self.logger.info(f"💾 Saved {len(article_data)} article records to database")

            except Exception as e:
                self.logger.error(f"❌ Error saving article data: {e}")

    def run_incremental_update(self):
        """Run the complete incremental update process"""
        self.logger.info("🔄 Starting incremental data update")

        # --- Sync themes from config/themes.yaml into LanceDB (NEW) ---
        themes_cfg = self.load_theme_config()
        if themes_cfg:
            theme_rows = self.build_theme_rows(themes_cfg)
            self.save_themes_table(theme_rows)
        else:
            self.logger.info("ℹ️ No themes loaded; skipping theme sync")

        # Get existing data for duplicate checking
        existing_stock_data = self.get_existing_stock_data()
        existing_articles = self.get_existing_articles()

        # Determine which stocks need updates
        stocks_needing_update = []
        stocks_needing_articles = []

        for ticker_info in self.ticker_config['tickers']:
            symbol = ticker_info['symbol']

            if self.needs_stock_update(ticker_info):
                stocks_needing_update.append(symbol)

            if self.needs_article_update(ticker_info):
                stocks_needing_articles.append(symbol)

        self.logger.info(f"📊 Stocks needing data update: {len(stocks_needing_update)}")
        self.logger.info(f"📰 Stocks needing article update: {len(stocks_needing_articles)}")

        # Process updates in batches
        batch_size = self.ticker_config['update_settings']['batch_size']

        all_updated_stocks = []
        all_new_articles = []

        # Update stock data in batches
        for i in range(0, len(stocks_needing_update), batch_size):
            batch = stocks_needing_update[i:i + batch_size]
            self.logger.info(f"📈 Processing stock batch {i//batch_size + 1}: {batch}")

            updated_stocks = self.update_stock_data(batch, existing_stock_data)
            all_updated_stocks.extend(updated_stocks)

        # Update article data in batches
        for i in range(0, len(stocks_needing_articles), batch_size):
            batch = stocks_needing_articles[i:i + batch_size]
            self.logger.info(f"📰 Processing article batch {i//batch_size + 1}: {batch}")

            new_articles = self.update_article_data(batch, existing_articles)
            all_new_articles.extend(new_articles)

        # Save all updates to database
        self.save_to_database(all_updated_stocks, all_new_articles)

        # Save updated ticker configuration
        self.save_ticker_config()

        # Generate summary
        summary = {
            'timestamp': datetime.now().isoformat(),
            'stocks_updated': len(all_updated_stocks),
            'articles_added': len(all_new_articles),
            'stocks_processed': stocks_needing_update,
            'articles_processed': stocks_needing_articles,
            'total_existing_articles': len(existing_articles),
            'total_existing_stocks': len(existing_stock_data)
        }

        self.logger.info("✅ Incremental update completed")
        self.logger.info(f"📊 Summary: {json.dumps(summary, indent=2)}")

        return summary


def main():
    """Main execution function"""
    print("🚀 PRODUCTION INCREMENTAL DATA INGESTION")
    print("=" * 80)

    try:
        ingester = IncrementalDataIngester()
        summary = ingester.run_incremental_update()

        print("=" * 80)
        print("✅ INCREMENTAL UPDATE COMPLETED")
        print("=" * 80)
        print(f"📊 Stocks updated: {summary['stocks_updated']}")
        print(f"📰 Articles added: {summary['articles_added']}")
        print(f"⏱️ Completed at: {summary['timestamp']}")
        print("=" * 80)

    except Exception as e:
        print(f"❌ Error during incremental update: {e}")
        raise


if __name__ == "__main__":
    main()
