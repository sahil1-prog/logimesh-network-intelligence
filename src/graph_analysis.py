import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import os

def build_graph(edges_df, nodes_df):
    print("Building NetworkX DiGraph...")
    G = nx.DiGraph()
    
    # Add nodes with attributes
    for _, row in nodes_df.iterrows():
        G.add_node(row['node_id'], name=row['node_name'])
        
    # Add edges with attributes
    for _, row in edges_df.iterrows():
        G.add_edge(
            row['source_center'], 
            row['destination_center'], 
            weight=row['median_segment_factor'], # weight > 1 means delayed
            median_actual_time=row['median_segment_actual_time'],
            median_osrm_time=row['median_segment_osrm_time'],
            trip_count=row['trip_count']
        )
    return G

def compute_centralities(G):
    print("Computing centrality metrics...")
    
    # We invert the weight for betweenness so that higher delay factor = lower "cost" path 
    # to find paths that are most delayed? Actually, we want to find structural bottlenecks.
    # Standard betweenness just uses uniform weights or distance.
    # Let's compute unweighted betweenness first to capture structural importance.
    betweenness = nx.betweenness_centrality(G, weight=None)
    in_degree = dict(G.in_degree())
    out_degree = dict(G.out_degree())
    clustering = nx.clustering(G)
    
    metrics_df = pd.DataFrame({
        'node_id': list(G.nodes()),
        'betweenness_centrality': [betweenness.get(n, 0) for n in G.nodes()],
        'in_degree': [in_degree.get(n, 0) for n in G.nodes()],
        'out_degree': [out_degree.get(n, 0) for n in G.nodes()],
        'clustering_coefficient': [clustering.get(n, 0) for n in G.nodes()]
    })
    return metrics_df

def bottleneck_audit(edges_df, nodes_df, metrics_df):
    print("Running bottleneck and corridor audit...")
    
    # 1. Chronically delayed corridors (Factor > 1.20)
    delayed_corridors = edges_df[edges_df['median_segment_factor'] > 1.20].copy()
    
    # 2. Estimate SLA breach contribution per node
    # A simple metric: Number of outgoing delayed trips * hub's betweenness centrality
    # This identifies structural chokepoints that actually experience delays
    node_delays = delayed_corridors.groupby('source_center').agg(
        delayed_trips=('trip_count', 'sum'),
        avg_delay_factor=('median_segment_factor', 'mean')
    ).reset_index().rename(columns={'source_center': 'node_id'})
    
    hub_analysis = pd.merge(metrics_df, node_delays, on='node_id', how='left').fillna(0)
    hub_analysis = pd.merge(hub_analysis, nodes_df, on='node_id', how='left')
    
    # Rank by SLA Breach Score: betweenness_centrality * delayed_trips
    hub_analysis['sla_breach_score'] = hub_analysis['betweenness_centrality'] * hub_analysis['delayed_trips']
    top_bottlenecks = hub_analysis.sort_values('sla_breach_score', ascending=False).head(10)
    
    return delayed_corridors, hub_analysis, top_bottlenecks

def plot_network_metrics(hub_analysis, top_bottlenecks):
    print("Generating visualizations...")
    os.makedirs('reports/figures', exist_ok=True)
    
    # 1. Bottleneck Bar Chart
    plt.figure(figsize=(10, 6))
    sns.barplot(data=top_bottlenecks.head(5), x='sla_breach_score', y='node_name', palette='Reds_r')
    plt.title('LogiMesh: Top 5 Bottleneck Hubs by SLA Breach Contribution')
    plt.xlabel('SLA Breach Score (Structural Risk * Delayed Trips)')
    plt.ylabel('Hub Name')
    plt.tight_layout()
    plt.savefig('reports/figures/top_bottlenecks.png')
    plt.close()

if __name__ == "__main__":
    edges_df = pd.read_csv("data/processed/graph_edges.csv")
    nodes_df = pd.read_csv("data/processed/graph_nodes.csv")
    
    G = build_graph(edges_df, nodes_df)
    metrics_df = compute_centralities(G)
    
    delayed_corridors, hub_analysis, top_bottlenecks = bottleneck_audit(edges_df, nodes_df, metrics_df)
    
    print("\n--- TOP 5 LOGIMESH BOTTLENECK HUBS ---")
    for _, row in top_bottlenecks.head(5).iterrows():
        print(f"{row['node_name']} (ID: {row['node_id']}) - SLA Score: {row['sla_breach_score']:.4f}")
        
    plot_network_metrics(hub_analysis, top_bottlenecks)
    
    # Save node metrics for ML models
    hub_analysis.to_csv("data/processed/node_metrics.csv", index=False)
    delayed_corridors.to_csv("data/processed/delayed_corridors.csv", index=False)
    print("LogiMesh Graph Analysis completed successfully!")
