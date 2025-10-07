#!/usr/bin/env python3
"""
Contrastive Learning for DYNAMIC_THEME_SYSTEM (Unified, Dynamic Labeling)
=======================================================================

This script updates the original contrastive pipeline to use the robust, working
logic from `unified_contrastive_dynamic.py`:

- Auto-detects and loads from a unified LanceDB table (prefers `enhanced_stocks_tavily`,
  falls back to `enhanced_stocks_dynamic`).
- Dynamic, threshold-based semantic labeling (no dependency on pre-built
  relationship tables, but will log if present).
- Early stopping and best-checkpoint restore.
- Clean evaluation and JSON report dump (compatible with prior reports path).

Paths are kept compatible with the original project layout where possible.
"""

import os
import sys
import json
import ast
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

import lancedb

# --- Optional project settings (kept from original) ---
# Add the core directory to the Python path if available
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
try:
    from core.config import get_settings  # noqa: F401
except Exception:
    get_settings = None


# =============================
# Model
# =============================
class ContrastiveModel(nn.Module):
    """Production-quality contrastive model with twin MLP projectors and a
    similarity head. Mirrors the architecture that worked in the unified script."""

    def __init__(self, input_dim=384, hidden_dim=256, dropout=0.3):
        super().__init__()
        # Stock projector
        self.stock_projector = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
        )
        # Theme projector
        self.theme_projector = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
        )
        # Similarity head on concatenated projections
        self.similarity_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, stock_emb, theme_emb):
        s = self.stock_projector(stock_emb)
        t = self.theme_projector(theme_emb)
        combined = torch.cat([s, t], dim=-1)
        sim = torch.sigmoid(self.similarity_head(combined))
        return sim, s, t


# =============================
# Trainer
# =============================
class DynamicContrastiveTrainer:
    """Trainer updated to the unified, dynamic-labeling approach.

    - Prefers `data/lancedb` but accepts an explicit db_path.
    - Auto-detects unified table vs legacy table names.
    - Builds labels by cosine similarity thresholds (pos/neg), balancing with sampled
      negatives. No dependency on stock_articles/relationships.
    """

    def __init__(self, db_path: str = "./data/lancedb", embedding_type: str = "weighted", device: str = "auto"):
        self.db_path = db_path
        self.embedding_type = embedding_type  # for unified table variants (e.g., weighted, average)
        self.device = torch.device('cuda' if torch.cuda.is_available() and device == 'auto' else 'cpu')
        self.db = lancedb.connect(db_path)

        print("🔧 Dynamic Contrastive Trainer (Unified) Initialized")
        print(f"🔧 Database: {db_path}")
        print(f"🔧 Device: {self.device}")
        print(f"🔧 Embedding type preference: {embedding_type}")
        print("-" * 80)

        # Placeholders
        self.stocks_df: pd.DataFrame | None = None
        self.themes_df: pd.DataFrame | None = None
        self.stock_embeddings: np.ndarray | None = None
        self.theme_embeddings: np.ndarray | None = None

    # -------- Data Loading --------
    def _open_if_exists(self, name: str):
        try:
            return self.db.open_table(name)
        except Exception:
            return None

    def load_data(self):
        """Load unified data (preferred) or legacy dynamic data. Extract embeddings robustly."""
        print("📊 Loading data from LanceDB with unified-first strategy")

        # Try unified first
        stocks_table = self._open_if_exists("enhanced_stocks_tavily")
        themes_table = self._open_if_exists("themes")

        if stocks_table is not None and themes_table is not None:
            mode = "unified"
            stocks_df = stocks_table.to_pandas()
            themes_df = themes_table.to_pandas()
            print(f"✅ Loaded {len(stocks_df)} enhanced stocks (unified) and {len(themes_df)} themes")

            # Decide embedding column
            candidate_cols = [
                f"{self.embedding_type}_embedding",  # preferred in unified
                "weighted_embedding",               # fallback
                "average_embedding",
                "embedding",
            ]
            stock_emb_col = next((c for c in candidate_cols if c in stocks_df.columns), None)
            if stock_emb_col is None:
                available = [c for c in stocks_df.columns if "embedding" in c]
                raise ValueError(f"No stock embedding column found. Available: {available}")

        else:
            # Fallback to legacy dynamic tables
            print("ℹ️ Unified table not found. Falling back to legacy dynamic tables…")
            stocks_table = self._open_if_exists("enhanced_stocks_dynamic")
            themes_table = self._open_if_exists("themes")
            if stocks_table is None or themes_table is None:
                raise RuntimeError("Neither unified nor legacy tables are available.")
            mode = "legacy"
            stocks_df = stocks_table.to_pandas()
            themes_df = themes_table.to_pandas()
            print(f"✅ Loaded {len(stocks_df)} stocks (legacy) and {len(themes_df)} themes")
            stock_emb_col = next((c for c in ["weighted_embedding", "embedding"] if c in stocks_df.columns), None)
            if stock_emb_col is None:
                available = [c for c in stocks_df.columns if "embedding" in c]
                raise ValueError(f"No stock embedding column found in legacy mode. Available: {available}")

        # Parse embeddings safely (string → list[float] or already list/np)
        def _to_array(series: pd.Series) -> np.ndarray:
            data = []
            for emb in series:
                if isinstance(emb, (list, np.ndarray)):
                    data.append(np.array(emb))
                elif isinstance(emb, str):
                    try:
                        data.append(np.array(ast.literal_eval(emb)))
                    except Exception:
                        # last resort eval (not ideal, but preserved for parity with original)
                        data.append(np.array(eval(emb)))  # noqa: S307
                else:
                    data.append(np.array(emb))
            return np.vstack(data)

        stock_embeddings = _to_array(stocks_df[stock_emb_col])
        theme_embeddings = _to_array(themes_df["embedding"]) if "embedding" in themes_df.columns else _to_array(themes_df.iloc[:, -1])

        print(f"📈 Stock embeddings shape: {stock_embeddings.shape}")
        print(f"📈 Theme embeddings shape: {theme_embeddings.shape}")

        # Save
        self.mode = mode
        self.stocks_df = stocks_df
        self.themes_df = themes_df
        self.stock_embeddings = stock_embeddings
        self.theme_embeddings = theme_embeddings
        return True

    # -------- Labeling (Unified Dynamic) --------
    def create_dynamic_mappings(self, similarity_threshold: float = 0.35, negative_threshold: float = 0.15):
        """Create threshold-based positive/negative mappings using cosine similarity.
        Balances negatives per-stock to roughly match positives (min 3).
        """
        assert self.stocks_df is not None and self.themes_df is not None
        assert self.stock_embeddings is not None and self.theme_embeddings is not None

        print("🎯 Creating DYNAMIC mappings (threshold-based)")
        print(f"   📈 Positive ≥ {similarity_threshold}")
        print(f"   📉 Negative ≤ {negative_threshold}")

        stock_embs = F.normalize(torch.FloatTensor(self.stock_embeddings), p=2, dim=1)
        theme_embs = F.normalize(torch.FloatTensor(self.theme_embeddings), p=2, dim=1)
        sim_matrix = torch.mm(stock_embs, theme_embs.t())
        print(f"   🔍 Sim matrix: {sim_matrix.shape}, range: [{sim_matrix.min().item():.3f}, {sim_matrix.max().item():.3f}]")

        mappings: list[tuple[int, int, float, float]] = []
        for s_idx in range(len(self.stocks_df)):
            sims = sim_matrix[s_idx]
            pos_idx = torch.where(sims >= similarity_threshold)[0]
            neg_idx = torch.where(sims <= negative_threshold)[0]

            # positives
            for t_idx in pos_idx.tolist():
                mappings.append((s_idx, t_idx, 1.0, float(sims[t_idx].item())))

            # negatives (sample ~balance, ≥3)
            if len(neg_idx) > 0:
                n_pos = max(1, len(pos_idx))
                n_neg = min(len(neg_idx), max(n_pos, 3))
                choice = neg_idx[torch.randperm(len(neg_idx))[:n_neg]].tolist()
                for t_idx in choice:
                    mappings.append((s_idx, int(t_idx), 0.0, float(sims[t_idx].item())))

            # log a sample per stock (first positive only)
            if len(pos_idx) > 0:
                theme_name = self.themes_df.iloc[pos_idx[0].item()]["name"]
                print(f"   {self.stocks_df.iloc[s_idx]['symbol']} → {theme_name} (sim={sims[pos_idx[0]].item():.3f})")

        pos = sum(1 for m in mappings if m[2] > 0.5)
        neg = len(mappings) - pos
        print(f"✅ Created {len(mappings)} mappings  |  📈 {pos} pos  📉 {neg} neg  ⚖️ {pos/(neg or 1):.2f}")
        return mappings

    # -------- Tensors --------
    @staticmethod
    def prepare_training_data(stocks_df, themes_df, stock_embeddings, theme_embeddings, mappings):
        Xs, Xt, y, sims = [], [], [], []
        for s_idx, t_idx, label, sim in mappings:
            Xs.append(stock_embeddings[s_idx])
            Xt.append(theme_embeddings[t_idx])
            y.append(label)
            sims.append(sim)
        Xs = torch.FloatTensor(np.array(Xs))
        Xt = torch.FloatTensor(np.array(Xt))
        y = torch.FloatTensor(y)
        sims = torch.FloatTensor(sims)
        print(f"✅ Training tensors: {Xs.shape[0]} examples | Stock {Xs.shape} | Theme {Xt.shape}")
        return Xs, Xt, y, sims

    # -------- Train --------
    def train_model(self, X_stock, X_theme, y, epochs: int = 100, lr: float = 1e-3, validation_split: float = 0.2):
        print("🚀 Training Contrastive Model (unified)")
        n = X_stock.shape[0]
        idx = torch.randperm(n)
        n_train = int((1 - validation_split) * n)
        tr_idx, va_idx = idx[:n_train], idx[n_train:]

        Xs_tr, Xt_tr, y_tr = X_stock[tr_idx].to(self.device), X_theme[tr_idx].to(self.device), y[tr_idx].to(self.device)
        Xs_va, Xt_va, y_va = X_stock[va_idx].to(self.device), X_theme[va_idx].to(self.device), y[va_idx].to(self.device)

        model = ContrastiveModel(input_dim=X_stock.shape[1]).to(self.device)
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        crit = nn.BCELoss()

        best_val_acc, patience, wait = 0.0, 15, 0
        history = []

        for epoch in range(epochs):
            model.train()
            opt.zero_grad()
            pred, _, _ = model(Xs_tr, Xt_tr)
            loss = crit(pred.squeeze(), y_tr)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            model.eval()
            with torch.no_grad():
                vpred, _, _ = model(Xs_va, Xt_va)
                vloss = crit(vpred.squeeze(), y_va)
                tr_acc = ((pred.squeeze() > 0.5) == y_tr).float().mean().item()
                va_acc = ((vpred.squeeze() > 0.5) == y_va).float().mean().item()

            history.append({
                'epoch': int(epoch),
                'train_loss': float(loss.item()),
                'val_loss': float(vloss.item()),
                'train_acc': float(tr_acc),
                'val_acc': float(va_acc),
            })

            improved = va_acc > best_val_acc
            if improved:
                best_val_acc = va_acc
                wait = 0
                torch.save(model.state_dict(), 'dynamic_contrastive_best.pt')
            else:
                wait += 1
                if wait >= patience:
                    print(f"   Early stopping at epoch {epoch}")
                    break

            if epoch % 10 == 0:
                print(f"   Epoch {epoch:3d} | Train {loss.item():.4f} / Val {vloss.item():.4f} | Acc {tr_acc:.4f}/{va_acc:.4f}")

        model.load_state_dict(torch.load('dynamic_contrastive_best.pt'))
        print(f"✅ Training complete. Best Val Acc: {best_val_acc:.4f}")
        return model, history, best_val_acc

    # -------- Evaluate --------
    def evaluate_model(self, model):
        assert self.stocks_df is not None and self.themes_df is not None
        assert self.stock_embeddings is not None and self.theme_embeddings is not None

        print("📊 Evaluating model over all stock–theme pairs…")
        model.eval()
        results = []

        with torch.no_grad():
            for s_idx in range(len(self.stocks_df)):
                stock_symbol = self.stocks_df.iloc[s_idx].get('symbol', f'stock_{s_idx}')
                stock_sector = self.stocks_df.iloc[s_idx].get('sector', '')
                s_emb = torch.FloatTensor(self.stock_embeddings[s_idx]).unsqueeze(0).to(self.device)

                scores = []
                for t_idx in range(len(self.themes_df)):
                    theme_name = self.themes_df.iloc[t_idx].get('name', f'theme_{t_idx}')
                    t_emb = torch.FloatTensor(self.theme_embeddings[t_idx]).unsqueeze(0).to(self.device)
                    pred, _, _ = model(s_emb, t_emb)
                    scores.append({
                        'stock': stock_symbol,
                        'theme': theme_name,
                        'score': float(pred.item()),
                        'stock_sector': stock_sector,
                    })

                scores.sort(key=lambda x: x['score'], reverse=True)
                results.extend(scores)

                # Optional: log a few prominent tickers
                if stock_symbol in {'BNTX', 'BIIB', 'VZ', 'T', 'TSLA'}:
                    print(f"   {stock_symbol} ({stock_sector}) top themes:")
                    for i, r in enumerate(scores[:3]):
                        print(f"     {i+1}. {r['theme']}: {r['score']:.4f}")
        return results


# =============================
# Main
# =============================

def main():
    print("🚀 CONTRASTIVE LEARNING (Unified/Dynamic)")
    print("=" * 80)

    try:
        # Init
        trainer = DynamicContrastiveTrainer(db_path=os.environ.get('LANCEDB_PATH', './data/lancedb'),
                                            embedding_type=os.environ.get('EMBEDDING_TYPE', 'weighted'))

        # Load
        if not trainer.load_data():
            print("❌ Failed to load data")
            return

        # Build mappings (dynamic)
        mappings = trainer.create_dynamic_mappings(
            similarity_threshold=float(os.environ.get('POS_THRESHOLD', 0.35)),
            negative_threshold=float(os.environ.get('NEG_THRESHOLD', 0.15)),
        )
        if len(mappings) == 0:
            print("❌ No mappings produced")
            return

        # Tensors
        Xs, Xt, y, sims = trainer.prepare_training_data(
            trainer.stocks_df, trainer.themes_df, trainer.stock_embeddings, trainer.theme_embeddings, mappings
        )

        # Train
        model, history, best_val_acc = trainer.train_model(Xs, Xt, y, epochs=int(os.environ.get('EPOCHS', 100)))

        # Evaluate
        results = trainer.evaluate_model(model)

        # Save
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        Path('reports').mkdir(parents=True, exist_ok=True)
        out_path = f"reports/contrastive_results_{ts}.json"
        summary = {
            'timestamp': ts,
            'best_val_accuracy': float(best_val_acc),
            'num_mappings': int(len(mappings)),
            'num_stocks': int(len(trainer.stocks_df)),
            'num_themes': int(len(trainer.themes_df)),
            'sample_results': results[:20],
            'history_tail': history[-10:],
            'mode': getattr(trainer, 'mode', 'unknown'),
            'embedding_type': trainer.embedding_type,
        }
        with open(out_path, 'w') as f:
            json.dump(summary, f, indent=2)

        print("\n" + "=" * 80)
        print("🎯 CONTRASTIVE LEARNING COMPLETE")
        print("=" * 80)
        print(f"📊 Best val accuracy: {best_val_acc:.4f}")
        print(f"📊 Total mappings: {len(mappings)}")
        print(f"💾 Results saved to: {out_path}")

    except Exception as e:
        print(f"❌ Error during contrastive learning: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
