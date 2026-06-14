import pandas as pd
import numpy as np
import networkx as nx
from sklearn.manifold import SpectralEmbedding
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, accuracy_score, classification_report
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

def generate_embeddings(G, n_components=16):
    print("Generating node embeddings using Spectral Embedding...")
    nodes = list(G.nodes())
    adj_matrix = nx.to_scipy_sparse_array(G, nodelist=nodes, weight='weight')
    
    # We use SpectralEmbedding to learn structural representations of the hubs
    embedder = SpectralEmbedding(n_components=n_components, affinity='precomputed', random_state=42)
    
    # Symmetrize adj_matrix for spectral embedding
    sym_adj = adj_matrix.maximum(adj_matrix.transpose()).tocsr()
    
    # Fix for scikit-learn SpectralEmbedding expecting 32-bit integer indices
    sym_adj.indices = sym_adj.indices.astype(np.int32)
    sym_adj.indptr = sym_adj.indptr.astype(np.int32)
    
    embeddings = embedder.fit_transform(sym_adj)
    
    emb_df = pd.DataFrame(embeddings, columns=[f'emb_{i}' for i in range(n_components)])
    emb_df['node_id'] = nodes
    return emb_df

def prepare_eta_data(trips_df, node_metrics, embeddings):
    print("Preparing ETA prediction dataset...")
    # Baseline features
    base_features = ['osrm_time', 'osrm_distance', 'hour_of_day']
    
    # We will encode categorical features
    trips_df['route_type_encoded'] = (trips_df['route_type'] == 'FTL').astype(int)
    base_features.append('route_type_encoded')
    
    # Merge graph features for source
    graph_cols = ['betweenness_centrality', 'in_degree', 'out_degree', 'clustering_coefficient']
    emb_cols = [c for c in embeddings.columns if c.startswith('emb_')]
    
    df = pd.merge(trips_df, node_metrics[['node_id'] + graph_cols], left_on='source_center', right_on='node_id', how='left')
    df = pd.merge(df, embeddings, left_on='source_center', right_on='node_id', how='left')
    
    # Rename for source
    rename_dict = {c: f"src_{c}" for c in graph_cols + emb_cols}
    df.rename(columns=rename_dict, inplace=True)
    df.drop(columns=['node_id_x', 'node_id_y'], inplace=True, errors='ignore')
    
    graph_features = list(rename_dict.values())
    
    # Filter out NaNs if any node wasn't in graph
    df = df.dropna(subset=base_features + graph_features + ['actual_time'])
    
    return df, base_features, graph_features

def evaluate_model(y_true, y_pred, model_name):
    mae = mean_absolute_error(y_true, y_pred)
    # Calculate % of trips within 15% error
    pct_error = np.abs(y_true - y_pred) / y_true
    acc_15 = np.mean(pct_error <= 0.15) * 100
    print(f"[{model_name}] MAE: {mae:.2f} mins | accuracy@15%: {acc_15:.2f}%")
    return mae, acc_15

def train_eta_models(df, base_features, graph_features):
    print("\n--- Training ETA Prediction Models ---")
    X_base = df[base_features]
    X_graph = df[base_features + graph_features]
    y = df['actual_time']
    
    # Split
    # Same indices for both models to make comparison fair
    idx_train, idx_test = train_test_split(np.arange(len(df)), test_size=0.2, random_state=42)
    
    # Baseline Model
    m_base = lgb.LGBMRegressor(n_estimators=100, random_state=42)
    m_base.fit(X_base.iloc[idx_train], y.iloc[idx_train])
    y_pred_base = m_base.predict(X_base.iloc[idx_test])
    
    # Graph-Enhanced Model
    m_graph = lgb.LGBMRegressor(n_estimators=100, random_state=42)
    m_graph.fit(X_graph.iloc[idx_train], y.iloc[idx_train])
    y_pred_graph = m_graph.predict(X_graph.iloc[idx_test])
    
    evaluate_model(y.iloc[idx_test], y_pred_base, "Baseline Model")
    evaluate_model(y.iloc[idx_test], y_pred_graph, "Graph-Enhanced Model")
    
    return m_graph

def ftl_vs_carting_framework(edges_df, node_metrics):
    print("\n--- FTL vs Carting Decision Framework ---")
    # For decision framework, we predict which route type is optimal
    # Or analyze the factor difference
    
    # We will build a profile for corridors:
    df = pd.merge(edges_df, node_metrics[['node_id', 'betweenness_centrality']], left_on='source_center', right_on='node_id')
    df['dist_bucket'] = pd.qcut(df['median_segment_osrm_distance'], q=3, labels=['Short', 'Medium', 'Long'])
    
    profile_analysis = df.groupby(['dist_bucket', 'time_of_day', 'route_type']).agg(
        avg_delay_factor=('median_segment_factor', 'mean'),
        avg_actual_time=('median_segment_actual_time', 'mean'),
        corridor_count=('source_center', 'count')
    ).reset_index()
    
    print("\nSample Cost-Time Tradeoff Matrix (FTL vs Carting):")
    # Pivot to compare directly
    comparison = profile_analysis.pivot(index=['dist_bucket', 'time_of_day'], columns='route_type', values=['avg_delay_factor', 'avg_actual_time'])
    print(comparison.head(10).to_string())
    
    comparison.to_csv("data/processed/ftl_carting_framework.csv")
    print("FTL vs Carting decision table saved.")

if __name__ == "__main__":
    trips_df = pd.read_csv("data/processed/trips_clean.csv")
    edges_df = pd.read_csv("data/processed/graph_edges.csv")
    node_metrics = pd.read_csv("data/processed/node_metrics.csv")
    
    # Build DiGraph for embeddings
    G = nx.DiGraph()
    for _, row in edges_df.iterrows():
        G.add_edge(row['source_center'], row['destination_center'], weight=row['median_segment_factor'])
        
    embeddings = generate_embeddings(G, n_components=16)
    
    eta_df, base_feats, graph_feats = prepare_eta_data(trips_df, node_metrics, embeddings)
    
    m_graph = train_eta_models(eta_df, base_feats, graph_feats)
    
    import pickle
    with open("data/processed/eta_model.pkl", "wb") as f:
        pickle.dump(m_graph, f)
    print("Saved trained model to data/processed/eta_model.pkl")
    
    # Save feature metadata
    with open("data/processed/feature_metadata.pkl", "wb") as f:
        pickle.dump({"base_features": base_feats, "graph_features": graph_feats}, f)
        
    # Save combined node features for dashboard inference
    node_features = pd.merge(node_metrics, embeddings, on='node_id', how='left')
    node_features.to_csv("data/processed/node_features.csv", index=False)
    print("Saved node features to data/processed/node_features.csv")
    
    ftl_vs_carting_framework(edges_df, node_metrics)
    
    print("\nLogiMesh Modeling Phase completed successfully!")
