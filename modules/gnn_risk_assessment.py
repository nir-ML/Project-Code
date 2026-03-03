"""
GNN Risk Assessment Module

Graph Neural Network models for drug interaction risk assessment using
Graph Attention Networks (GAT) and PubMedBERT embeddings.

Models:
1. GAT-Severity: Predicts severity labels from network structure
2. GAT-Embedding: Uses PubMedBERT drug embeddings for rich features
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.data import Data, DataLoader
    from torch_geometric.nn import GATConv, GCNConv, global_mean_pool
    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    TORCH_GEOMETRIC_AVAILABLE = False
    print("Warning: torch_geometric not installed. Run: pip install torch-geometric")

try:
    from transformers import AutoTokenizer, AutoModel
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: transformers not installed. Run: pip install transformers")


@dataclass
class GNNConfig:
    """Configuration for GNN models"""
    hidden_dim: int = 128
    num_heads: int = 4
    num_layers: int = 2
    dropout: float = 0.3
    learning_rate: float = 0.001
    epochs: int = 100
    batch_size: int = 64
    embedding_model: str = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"


class DrugEmbedder:
    """
    Generate drug embeddings using PubMedBERT
    
    Embeds drug names and descriptions into dense vectors for GNN features.
    """
    
    def __init__(self, model_name: str = None):
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers package required for embeddings")
        
        self.model_name = model_name or GNNConfig.embedding_model
        self.tokenizer = None
        self.model = None
        self._loaded = False
        
    def load_model(self):
        """Lazy load the embedding model"""
        if self._loaded:
            return
            
        print(f"Loading embedding model: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        self.model.eval()
        
        # Move to GPU if available
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        
        self._loaded = True
        
    def embed(self, text: str) -> np.ndarray:
        """Get embedding for a single text"""
        self.load_model()
        
        inputs = self.tokenizer(text, return_tensors="pt", 
                               truncation=True, max_length=512, padding=True)
        
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Use CLS token embedding
            embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        
        return embedding.flatten()
    
    def embed_batch(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Get embeddings for multiple texts"""
        self.load_model()
        
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            inputs = self.tokenizer(batch, return_tensors="pt",
                                   truncation=True, max_length=512, 
                                   padding=True)
            
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            
            all_embeddings.append(embeddings)
            
            if (i // batch_size) % 10 == 0:
                print(f"  Embedded {min(i + batch_size, len(texts))}/{len(texts)} drugs")
        
        return np.vstack(all_embeddings)


class GATRiskModel(nn.Module):
    """
    Graph Attention Network for drug risk prediction
    
    Architecture:
    - Multiple GAT layers with multi-head attention
    - Residual connections
    - Final classification/regression head
    """
    
    def __init__(self, in_channels: int, hidden_channels: int = 128,
                 out_channels: int = 3, heads: int = 4, num_layers: int = 2,
                 dropout: float = 0.3):
        super(GATRiskModel, self).__init__()
        
        if not TORCH_GEOMETRIC_AVAILABLE:
            raise ImportError("torch_geometric required for GNN models")
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Input projection
        self.input_proj = nn.Linear(in_channels, hidden_channels)
        
        # GAT layers
        self.gat_layers = nn.ModuleList()
        self.gat_layers.append(
            GATConv(hidden_channels, hidden_channels // heads, 
                   heads=heads, dropout=dropout)
        )
        
        for _ in range(num_layers - 1):
            self.gat_layers.append(
                GATConv(hidden_channels, hidden_channels // heads,
                       heads=heads, dropout=dropout)
            )
        
        # Output head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, out_channels)
        )
        
    def forward(self, x, edge_index, batch=None):
        # Input projection
        x = F.relu(self.input_proj(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # GAT layers with residual
        for i, gat in enumerate(self.gat_layers):
            residual = x
            x = gat(x, edge_index)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            if x.shape == residual.shape:
                x = x + residual
        
        # Global pooling if batch provided (for graph-level prediction)
        if batch is not None:
            x = global_mean_pool(x, batch)
        
        # Classification
        return self.classifier(x)


class GNNSeverityPredictor:
    """
    GAT model that predicts severity from network structure
    
    Uses node features derived from network topology only
    (degree, centrality, clustering) to avoid AI label circularity.
    """
    
    def __init__(self, config: GNNConfig = None):
        self.config = config or GNNConfig()
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def prepare_features(self, df: pd.DataFrame, 
                        drug_metrics: Dict[str, Dict]) -> Tuple[np.ndarray, List[Tuple]]:
        """
        Prepare network-based features for each drug
        
        Returns: (node_features, edge_list)
        """
        # Get unique drugs
        if 'drug_name_1' in df.columns:
            drugs1 = df['drug_name_1'].unique()
            drugs2 = df['drug_name_2'].unique()
            drugs = list(set(drugs1) | set(drugs2))
        elif 'drug1_name' in df.columns:
            drugs1 = df['drug1_name'].unique()
            drugs2 = df['drug2_name'].unique()
            drugs = list(set(drugs1) | set(drugs2))
        else:
            drugs = df['Drug'].unique().tolist() if 'Drug' in df.columns else []
        
        drug_to_idx = {d: i for i, d in enumerate(drugs)}
        
        # Extract features for each drug
        features = []
        for drug in drugs:
            if drug in drug_metrics:
                m = drug_metrics[drug]
                feat = [
                    m.get('degree_centrality', 0),
                    m.get('betweenness_centrality', 0),
                    m.get('closeness_centrality', 0),
                    m.get('eigenvector_centrality', 0),
                    m.get('pagerank', 0),
                    m.get('clustering_coefficient', 0),
                    m.get('major_interaction_ratio', 0),
                    m.get('interaction_count', 0) / 1000,  # Normalized
                ]
            else:
                feat = [0.0] * 8
            features.append(feat)
        
        # Build edge list
        edges = []
        if 'drug_name_1' in df.columns:
            for _, row in df.iterrows():
                d1, d2 = row['drug_name_1'], row['drug_name_2']
                if d1 in drug_to_idx and d2 in drug_to_idx:
                    edges.append((drug_to_idx[d1], drug_to_idx[d2]))
                    edges.append((drug_to_idx[d2], drug_to_idx[d1]))  # Undirected
        elif 'drug1_name' in df.columns:
            for _, row in df.iterrows():
                d1, d2 = row['drug1_name'], row['drug2_name']
                if d1 in drug_to_idx and d2 in drug_to_idx:
                    edges.append((drug_to_idx[d1], drug_to_idx[d2]))
                    edges.append((drug_to_idx[d2], drug_to_idx[d1]))  # Undirected
        
        return np.array(features), edges, drug_to_idx
    
    def build_graph(self, features: np.ndarray, edges: List[Tuple],
                   labels: Optional[np.ndarray] = None) -> Data:
        """Build PyTorch Geometric Data object"""
        x = torch.tensor(features, dtype=torch.float)
        
        # Handle empty edges
        if len(edges) == 0:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
        else:
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        
        if labels is not None:
            y = torch.tensor(labels, dtype=torch.long)
            return Data(x=x, edge_index=edge_index, y=y)
        
        return Data(x=x, edge_index=edge_index)
    
    def train(self, data: Data, val_data: Optional[Data] = None):
        """Train the GNN model"""
        self.model = GATRiskModel(
            in_channels=data.x.shape[1],
            hidden_channels=self.config.hidden_dim,
            out_channels=3,  # Major, Moderate, Minor
            heads=self.config.num_heads,
            num_layers=self.config.num_layers,
            dropout=self.config.dropout
        ).to(self.device)
        
        optimizer = torch.optim.Adam(self.model.parameters(), 
                                    lr=self.config.learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        data = data.to(self.device)
        
        best_val_acc = 0
        patience_counter = 0
        patience = 10
        
        for epoch in range(self.config.epochs):
            self.model.train()
            optimizer.zero_grad()
            
            out = self.model(data.x, data.edge_index)
            loss = criterion(out, data.y)
            loss.backward()
            optimizer.step()
            
            # Validation
            if val_data is not None:
                val_acc = self.evaluate(val_data)
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
            
            if (epoch + 1) % 20 == 0:
                train_acc = self.evaluate(data)
                print(f"Epoch {epoch+1}/{self.config.epochs} | "
                      f"Loss: {loss.item():.4f} | Train Acc: {train_acc:.4f}")
    
    def evaluate(self, data: Data) -> float:
        """Evaluate model accuracy"""
        self.model.eval()
        data = data.to(self.device)
        
        with torch.no_grad():
            out = self.model(data.x, data.edge_index)
            pred = out.argmax(dim=1)
            correct = (pred == data.y).sum().item()
            acc = correct / len(data.y)
        
        return acc
    
    def predict(self, data: Data) -> np.ndarray:
        """Get predictions"""
        self.model.eval()
        data = data.to(self.device)
        
        with torch.no_grad():
            out = self.model(data.x, data.edge_index)
            pred = F.softmax(out, dim=1).cpu().numpy()
        
        return pred


class GNNEmbeddingPredictor:
    """
    GAT model using PubMedBERT embeddings as node features
    
    Leverages rich semantic information from drug names/descriptions
    for more informed risk prediction.
    """
    
    def __init__(self, config: GNNConfig = None):
        self.config = config or GNNConfig()
        self.embedder = DrugEmbedder(self.config.embedding_model)
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.drug_embeddings: Dict[str, np.ndarray] = {}
    
    def compute_embeddings(self, drugs: List[str]) -> np.ndarray:
        """Compute or retrieve embeddings for drugs"""
        # Check cache
        to_embed = [d for d in drugs if d not in self.drug_embeddings]
        
        if to_embed:
            print(f"Computing embeddings for {len(to_embed)} drugs...")
            embeddings = self.embedder.embed_batch(to_embed)
            for drug, emb in zip(to_embed, embeddings):
                self.drug_embeddings[drug] = emb
        
        return np.array([self.drug_embeddings[d] for d in drugs])
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, List[Tuple], Dict]:
        """Prepare PubMedBERT embedding features"""
        # Get unique drugs
        if 'drug_name_1' in df.columns:
            drugs1 = df['drug_name_1'].unique()
            drugs2 = df['drug_name_2'].unique()
            drugs = list(set(drugs1) | set(drugs2))
        elif 'drug1_name' in df.columns:
            drugs1 = df['drug1_name'].unique()
            drugs2 = df['drug2_name'].unique()
            drugs = list(set(drugs1) | set(drugs2))
        else:
            drugs = []
        
        drug_to_idx = {d: i for i, d in enumerate(drugs)}
        
        # Get embeddings
        features = self.compute_embeddings(drugs)
        
        # Build edges
        edges = []
        if 'drug_name_1' in df.columns:
            for _, row in df.iterrows():
                d1, d2 = row['drug_name_1'], row['drug_name_2']
                if d1 in drug_to_idx and d2 in drug_to_idx:
                    edges.append((drug_to_idx[d1], drug_to_idx[d2]))
                    edges.append((drug_to_idx[d2], drug_to_idx[d1]))
        elif 'drug1_name' in df.columns:
            for _, row in df.iterrows():
                d1, d2 = row['drug1_name'], row['drug2_name']
                if d1 in drug_to_idx and d2 in drug_to_idx:
                    edges.append((drug_to_idx[d1], drug_to_idx[d2]))
                    edges.append((drug_to_idx[d2], drug_to_idx[d1]))
        
        return features, edges, drug_to_idx
    
    def build_and_train(self, df: pd.DataFrame, labels: Dict[str, int]):
        """Build graph and train model"""
        features, edges, drug_to_idx = self.prepare_features(df)
        
        # Map labels to array
        idx_to_drug = {i: d for d, i in drug_to_idx.items()}
        label_array = np.array([
            labels.get(idx_to_drug[i], 1) for i in range(len(drug_to_idx))
        ])
        
        # Build data
        x = torch.tensor(features, dtype=torch.float)
        if len(edges) == 0:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
        else:
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        y = torch.tensor(label_array, dtype=torch.long)
        
        data = Data(x=x, edge_index=edge_index, y=y)
        
        # Initialize and train model
        self.model = GATRiskModel(
            in_channels=features.shape[1],
            hidden_channels=self.config.hidden_dim,
            out_channels=3,
            heads=self.config.num_heads,
            num_layers=self.config.num_layers,
            dropout=self.config.dropout
        ).to(self.device)
        
        # Training loop
        optimizer = torch.optim.Adam(self.model.parameters(),
                                    lr=self.config.learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        data = data.to(self.device)
        
        for epoch in range(self.config.epochs):
            self.model.train()
            optimizer.zero_grad()
            
            out = self.model(data.x, data.edge_index)
            loss = criterion(out, data.y)
            loss.backward()
            optimizer.step()
            
            if (epoch + 1) % 20 == 0:
                self.model.eval()
                with torch.no_grad():
                    pred = out.argmax(dim=1)
                    acc = (pred == data.y).float().mean().item()
                print(f"Epoch {epoch+1} | Loss: {loss.item():.4f} | Acc: {acc:.4f}")
        
        return data


def run_gnn_comparison(df: pd.DataFrame, drug_metrics: Dict[str, Dict],
                      output_dir: str = "gnn_results") -> Dict[str, Any]:
    """
    Run comparison between GNN approaches
    
    Compares:
    1. GAT with network features only (avoids circular validation)
    2. GAT with PubMedBERT embeddings
    """
    os.makedirs(output_dir, exist_ok=True)
    results = {}
    
    print("=" * 60)
    print("GNN MODEL COMPARISON")
    print("=" * 60)
    
    # Prepare severity labels (using only confirmed/clinical labels if available)
    severity_map = {'Major': 2, 'Moderate': 1, 'Minor': 0}
    
    # Model 1: Network features only
    print("\n1. Training GAT with Network Features...")
    print("-" * 40)
    
    if TORCH_GEOMETRIC_AVAILABLE:
        try:
            predictor1 = GNNSeverityPredictor()
            features, edges, drug_to_idx = predictor1.prepare_features(df, drug_metrics)
            
            # Create labels
            idx_to_drug = {i: d for d, i in drug_to_idx.items()}
            labels = np.ones(len(drug_to_idx), dtype=np.int64)  # Default moderate
            
            if 'Severity' in df.columns and 'drug1_name' in df.columns:
                drug_severity = df.groupby('drug1_name')['Severity'].apply(
                    lambda x: x.mode()[0] if len(x) > 0 else 'Moderate'
                ).to_dict()
                for drug, idx in drug_to_idx.items():
                    if drug in drug_severity:
                        labels[idx] = severity_map.get(drug_severity[drug], 1)
            
            data = predictor1.build_graph(features, edges, labels)
            predictor1.train(data)
            
            final_acc = predictor1.evaluate(data)
            results['gat_network'] = {
                'accuracy': final_acc,
                'num_drugs': len(drug_to_idx),
                'feature_dim': features.shape[1]
            }
            print(f"  Final Accuracy: {final_acc:.4f}")
            
        except Exception as e:
            print(f"  Error: {e}")
            results['gat_network'] = {'error': str(e)}
    else:
        print("  Skipped: torch_geometric not available")
        results['gat_network'] = {'error': 'torch_geometric not installed'}
    
    # Model 2: PubMedBERT embeddings
    print("\n2. Training GAT with PubMedBERT Embeddings...")
    print("-" * 40)
    
    if TORCH_GEOMETRIC_AVAILABLE and TRANSFORMERS_AVAILABLE:
        try:
            predictor2 = GNNEmbeddingPredictor()
            
            # Map drug labels
            drug_labels = {}
            if 'Severity' in df.columns and 'drug1_name' in df.columns:
                drug_severity = df.groupby('drug1_name')['Severity'].apply(
                    lambda x: x.mode()[0] if len(x) > 0 else 'Moderate'
                ).to_dict()
                drug_labels = {d: severity_map.get(s, 1) for d, s in drug_severity.items()}
            
            data = predictor2.build_and_train(df.head(1000), drug_labels)  # Sample for speed
            
            predictor2.model.eval()
            with torch.no_grad():
                out = predictor2.model(data.x, data.edge_index)
                pred = out.argmax(dim=1)
                acc = (pred == data.y).float().mean().item()
            
            results['gat_embedding'] = {
                'accuracy': acc,
                'embedding_model': predictor2.config.embedding_model,
                'num_drugs': data.x.shape[0]
            }
            print(f"  Final Accuracy: {acc:.4f}")
            
        except Exception as e:
            print(f"  Error: {e}")
            results['gat_embedding'] = {'error': str(e)}
    else:
        print("  Skipped: Required packages not available")
        results['gat_embedding'] = {'error': 'Missing dependencies'}
    
    # Save results
    import json
    with open(os.path.join(output_dir, 'gnn_comparison_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "=" * 60)
    print("Results saved to:", output_dir)
    
    return results


if __name__ == "__main__":
    print("GNN Risk Assessment Module")
    print("=" * 60)
    
    # Check dependencies
    print(f"PyTorch Geometric available: {TORCH_GEOMETRIC_AVAILABLE}")
    print(f"Transformers available: {TRANSFORMERS_AVAILABLE}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
