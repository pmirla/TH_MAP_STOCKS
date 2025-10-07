#!/usr/bin/env python3
"""
Display Latest Results from LanceDB
Shows the most recent stock-theme analysis results directly from the database.
"""

import os
import sys
import pandas as pd
import lancedb
from datetime import datetime
import warnings
from dotenv import load_dotenv

warnings.filterwarnings('ignore')

# Load environment variables
env_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'settings.env')
load_dotenv(env_path)

# Add the core directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from core.config import get_settings

def display_latest_results():
    """Display the latest stock-theme analysis results from LanceDB"""
    settings = get_settings()
    db = lancedb.connect(settings.lancedb_path)
    
    print("=" * 80)
    print("🎯 LATEST DYNAMIC THEME SYSTEM RESULTS")
    print("=" * 80)
    print(f"📊 Reading latest results from LanceDB: {settings.lancedb_path}")
    
    try:
        # Load stock-theme relationships from database
        table = db.open_table("stock_theme_relationships")
        df = table.to_pandas()
        
        print(f"\n📈 SUMMARY STATISTICS")
        print("-" * 40)
        print(f"Total stocks analyzed: {df['symbol'].nunique()}")
        print(f"Total themes: {df['theme_name'].nunique()}")
        print(f"Total mappings found: {len(df)}")
        print(f"Average score: {df['normalized_score'].mean():.3f}")
        
        # Sort by symbol and score
        df_sorted = df.sort_values(['symbol', 'normalized_score'], ascending=[True, False])
        
        print(f"\n🏢 STOCK-TO-THEME MAPPINGS (Top themes per stock)")
        print("-" * 60)
        
        for symbol in sorted(df['symbol'].unique()):
            stock_data = df_sorted[df_sorted['symbol'] == symbol].head(3)  # Top 3 themes per stock
            print(f"\n{symbol}:")
            for _, row in stock_data.iterrows():
                print(f"  • {row['theme_name']}: {row['normalized_score']:.3f} "
                      f"(confidence: {row['confidence']:.2f}, articles: {row['article_count']})")
        
        print(f"\n" + "=" * 60)
        print(f"🎯 THEME-TO-STOCK MAPPINGS (Top stocks per theme)")
        print("=" * 60)
        
        for theme in sorted(df['theme_name'].unique()):
            theme_data = df_sorted[df_sorted['theme_name'] == theme].head(5)  # Top 5 stocks per theme
            print(f"\n{theme.upper().replace('_', ' ')}:")
            for _, row in theme_data.iterrows():
                print(f"  • {row['symbol']}: {row['normalized_score']:.3f}")
        
        # Show new stocks that weren't in the original system
        original_stocks = {'AAPL', 'MSFT', 'GOOGL', 'NVDA', 'AMGN', 'BIIB', 'GILD', 'REGN', 'BNTX', 'VZ', 'T', 'CMCSA', 'TSLA'}
        new_stocks = set(df['symbol'].unique()) - original_stocks
        
        if new_stocks:
            print(f"\n🆕 NEWLY ADDED STOCKS ({len(new_stocks)} stocks)")
            print("-" * 40)
            for symbol in sorted(new_stocks):
                stock_data = df_sorted[df_sorted['symbol'] == symbol].head(2)  # Top 2 themes
                themes_str = ", ".join([f"{row['theme_name']} ({row['normalized_score']:.3f})" 
                                      for _, row in stock_data.iterrows()])
                print(f"  • {symbol}: {themes_str}")
        
        # Show biotech and communications performance
        print(f"\n🧬 BIOTECH THEME PERFORMANCE")
        print("-" * 30)
        biotech_data = df[df['theme_name'] == 'biotechnology'].sort_values('normalized_score', ascending=False)
        for _, row in biotech_data.head(10).iterrows():
            print(f"  • {row['symbol']}: {row['normalized_score']:.3f} ({row['article_count']} articles)")
        
        print(f"\n📡 COMMUNICATIONS NETWORKS THEME PERFORMANCE")
        print("-" * 45)
        comm_data = df[df['theme_name'] == 'communications_networks'].sort_values('normalized_score', ascending=False)
        for _, row in comm_data.head(10).iterrows():
            print(f"  • {row['symbol']}: {row['normalized_score']:.3f} ({row['article_count']} articles)")
        
        print(f"\n" + "=" * 80)
        print(f"✅ ANALYSIS COMPLETE - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"📊 Total relationships: {len(df)}")
        print(f"🎯 Stocks with themes: {df['symbol'].nunique()}")
        print(f"🔍 Themes discovered: {df['theme_name'].nunique()}")
        print("=" * 80)
        
    except Exception as e:
        print(f"❌ Error reading from database: {e}")
        print("💡 Make sure the theme analysis has been run successfully")

if __name__ == "__main__":
    display_latest_results()
