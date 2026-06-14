import pandas as pd
import numpy as np
import os

def load_and_clean_data(file_path):
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)
    
    # 1. Handle missing names
    df['source_name'] = df['source_name'].fillna('Unknown_Source')
    df['destination_name'] = df['destination_name'].fillna('Unknown_Destination')
    
    # 2. Handle negative actual times
    initial_len = len(df)
    df = df[df['segment_actual_time'] >= 0]
    print(f"Dropped {initial_len - len(df)} rows with negative segment_actual_time.")
    
    # 3. Handle zero OSRM times
    df.loc[df['segment_osrm_time'] == 0, 'segment_osrm_time'] = 1.0
    
    # 4. Recalculate segment_factor just to be safe
    df['segment_factor'] = df['segment_actual_time'] / df['segment_osrm_time']
    
    # 5. Extract hour from trip_creation_time
    df['trip_creation_time'] = pd.to_datetime(df['trip_creation_time'])
    df['hour_of_day'] = df['trip_creation_time'].dt.hour
    
    # Bin hour into time of day
    def get_time_of_day(h):
        if 5 <= h < 12: return 'Morning'
        elif 12 <= h < 17: return 'Afternoon'
        elif 17 <= h < 22: return 'Evening'
        else: return 'Night'
    
    df['time_of_day'] = df['hour_of_day'].apply(get_time_of_day)
    return df

def build_graph_data(df):
    print("Building graph edges...")
    # Group by corridor, route_type, and time_of_day
    groupby_cols = ['source_center', 'destination_center', 'route_type', 'time_of_day']
    
    edges = df.groupby(groupby_cols).agg(
        median_segment_factor=('segment_factor', 'median'),
        median_segment_actual_time=('segment_actual_time', 'median'),
        median_segment_osrm_time=('segment_osrm_time', 'median'),
        median_segment_osrm_distance=('segment_osrm_distance', 'median'),
        trip_count=('trip_uuid', 'count')
    ).reset_index()
    
    # Add source and destination names for context
    source_names = df[['source_center', 'source_name']].drop_duplicates(subset=['source_center']).set_index('source_center')['source_name']
    dest_names = df[['destination_center', 'destination_name']].drop_duplicates(subset=['destination_center']).set_index('destination_center')['destination_name']
    
    edges['source_name'] = edges['source_center'].map(source_names)
    edges['destination_name'] = edges['destination_center'].map(dest_names)
    
    # Create node metadata
    nodes_src = df[['source_center', 'source_name']].rename(columns={'source_center': 'node_id', 'source_name': 'node_name'})
    nodes_dst = df[['destination_center', 'destination_name']].rename(columns={'destination_center': 'node_id', 'destination_name': 'node_name'})
    nodes = pd.concat([nodes_src, nodes_dst]).drop_duplicates(subset=['node_id'])
    
    return edges, nodes

if __name__ == "__main__":
    os.makedirs("data/processed", exist_ok=True)
    raw_file = "data/raw/delivery_data.csv"
    
    df_clean = load_and_clean_data(raw_file)
    edges, nodes = build_graph_data(df_clean)
    
    print(f"Saving cleaned trips ({len(df_clean)} rows)...")
    df_clean.to_csv("data/processed/trips_clean.csv", index=False)
    
    print(f"Saving graph edges ({len(edges)} rows)...")
    edges.to_csv("data/processed/graph_edges.csv", index=False)
    
    print(f"Saving graph nodes ({len(nodes)} rows)...")
    nodes.to_csv("data/processed/graph_nodes.csv", index=False)
    
    print("Data Pipeline completed successfully!")
