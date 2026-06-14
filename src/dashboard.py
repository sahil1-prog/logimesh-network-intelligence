import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
import pickle
import numpy as np
import os

# Set page configuration with a premium icon and title
st.set_page_config(
    layout="wide", 
    page_title="LogiMesh - Logistics Network Intelligence Platform", 
    page_icon="🕸️"
)

# Custom CSS for high-end styling (dark mode, glassmorphism cards, modern fonts)
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&display=swap');

/* Apply Outfit font family */
html, body, [class*="css"], .stApp {
    font-family: 'Outfit', sans-serif !important;
    background-color: #0f172a;
    color: #e2e8f0;
}

/* Glassmorphism Styling for Streamlit Metric Cards */
div[data-testid="stMetric"] {
    background: rgba(30, 41, 59, 0.45) !important;
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255, 255, 255, 0.05) !important;
    border-radius: 16px !important;
    padding: 20px 24px !important;
    box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3) !important;
    transition: transform 0.3s ease, border-color 0.3s ease !important;
}

div[data-testid="stMetric"]:hover {
    transform: translateY(-4px);
    border-color: rgba(99, 102, 241, 0.3) !important;
    box-shadow: 0 15px 30px -5px rgba(99, 102, 241, 0.1) !important;
}

/* Target Metric Label & Value colors */
div[data-testid="stMetricLabel"] {
    color: #94a3b8 !important;
    font-size: 0.95rem !important;
    font-weight: 500 !important;
}

div[data-testid="stMetricValue"] {
    color: #f8fafc !important;
    font-size: 2.2rem !important;
    font-weight: 700 !important;
}

/* Sidebar styling overrides */
section[data-testid="stSidebar"] {
    background-color: #090d16 !important;
    border-right: 1px solid rgba(255, 255, 255, 0.05);
}

/* Rebrand header styling */
.main-title {
    background: linear-gradient(135deg, #818cf8 0%, #c084fc 50%, #fb7185 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 2.3rem;
    font-weight: 800;
    margin-bottom: 5px;
    letter-spacing: -0.5px;
}
.subtitle-text {
    color: #94a3b8;
    font-size: 1.05rem;
    margin-bottom: 25px;
}

/* Sidebar Branding */
.sidebar-brand {
    font-weight: 800;
    font-size: 1.4rem;
    background: linear-gradient(135deg, #818cf8 0%, #c084fc 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 20px;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# Cache data loading
@st.cache_data
def load_data():
    edges_df = pd.read_csv("data/processed/graph_edges.csv")
    nodes_df = pd.read_csv("data/processed/graph_nodes.csv")
    metrics_df = pd.read_csv("data/processed/node_metrics.csv")
    ftl_df = pd.read_csv("data/processed/ftl_carting_framework.csv")
    node_features = pd.read_csv("data/processed/node_features.csv")
    return edges_df, nodes_df, metrics_df, ftl_df, node_features

# Cache ML model loading
@st.cache_resource
def load_model():
    model_path = "data/processed/eta_model.pkl"
    meta_path = "data/processed/feature_metadata.pkl"
    if os.path.exists(model_path) and os.path.exists(meta_path):
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        return model, meta
    return None, None

try:
    edges_df, nodes_df, metrics_df, ftl_df, node_features = load_data()
    model, meta = load_model()

    # ─── Sidebar Navigation ───────────────────────────────────────────────────
    st.sidebar.markdown('<div class="sidebar-brand">LogiMesh Platform</div>', unsafe_allow_html=True)
    st.sidebar.markdown("---")
    
    view = st.sidebar.radio(
        "Select Platform View",
        [
            "📊 Executive Dashboard",
            "🔍 Network Hub & Corridor Auditor",
            "🔮 Interactive ETA Predictor (ML)",
            "🚛 Route Mode Optimizer",
            "📝 Strategic Memo"
        ]
    )

    # Global Filters for applicable views
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔍 Global Filters")
    route_type_filter = st.sidebar.selectbox("Filter Route Type", ["All", "FTL", "Carting"])
    time_of_day_filter = st.sidebar.selectbox("Filter Time of Day", ["All", "Morning", "Afternoon", "Evening", "Night"])

    # Filtered edges
    filtered_edges = edges_df.copy()
    if route_type_filter != "All":
        filtered_edges = filtered_edges[filtered_edges['route_type'] == route_type_filter]
    if time_of_day_filter != "All":
        filtered_edges = filtered_edges[filtered_edges['time_of_day'] == time_of_day_filter]

    # ─── View 1: Executive Dashboard ──────────────────────────────────────────
    if view == "📊 Executive Dashboard":
        st.markdown('<div class="main-title">📊 Executive Portfolio Dashboard</div>', unsafe_allow_html=True)
        st.markdown('<div class="subtitle-text">Real-time network efficiency indicators, bottleneck centrality, and SLA compliance metrics.</div>', unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Corridors", f"{len(filtered_edges):,}")
        with col2:
            avg_delay = filtered_edges['median_segment_factor'].mean()
            st.metric("Avg Delay Factor", f"{avg_delay:.2f}x")
        with col3:
            breach_pct = (filtered_edges['median_segment_factor'] > 1.2).mean() * 100
            st.metric("Corridors Breaching SLA", f"{breach_pct:.1f}%")

        st.markdown("---")
        st.subheader("🛑 Top Bottlenecks by SLA Breach Score")
        
        col_left, col_right = st.columns([1.1, 0.9])
        top_hubs = metrics_df.sort_values('sla_breach_score', ascending=False).head(8)
        
        with col_left:
            st.dataframe(
                top_hubs[['node_name', 'betweenness_centrality', 'delayed_trips', 'sla_breach_score']],
                column_config={
                    "node_name": st.column_config.TextColumn("Facility Hub", width="large"),
                    "betweenness_centrality": st.column_config.NumberColumn("Betweenness Centrality", format="%.4f"),
                    "delayed_trips": st.column_config.NumberColumn("Delayed Trips Count"),
                    "sla_breach_score": st.column_config.ProgressColumn(
                        "SLA Breach Score", format="%.0f", min_value=0, max_value=float(top_hubs['sla_breach_score'].max())
                    ),
                },
                hide_index=True,
                use_container_width=True
            )
            
        with col_right:
            fig_bar = px.bar(
                top_hubs.head(5).sort_values('sla_breach_score'),
                x='sla_breach_score', y='node_name', orientation='h',
                labels={'sla_breach_score': 'SLA Breach Score', 'node_name': 'Facility Hub'},
                color='sla_breach_score', color_continuous_scale='purples'
            )
            fig_bar.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#94a3b8', family='Outfit'),
                showlegend=False, coloraxis_showscale=False,
                margin=dict(t=10, b=10, l=10, r=10),
                xaxis=dict(gridcolor='rgba(255,255,255,0.05)', tickfont=dict(color='#94a3b8')),
                yaxis=dict(tickfont=dict(color='#94a3b8'))
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        st.markdown("---")
        st.subheader("🕸️ Logistics Network Topology Map")
        st.markdown("Visualizing the top 150 most critical corridors by SLA breach risk.")
        
        risk_edges = filtered_edges[filtered_edges['median_segment_factor'] > 1.2].sort_values('trip_count', ascending=False).head(150)
        
        if len(risk_edges) > 0:
            G_vis = nx.from_pandas_edgelist(risk_edges, 'source_center', 'destination_center', ['median_segment_factor', 'trip_count'])
            pos = nx.spring_layout(G_vis, seed=42)
            
            edge_x, edge_y = [], []
            for edge in G_vis.edges():
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])
                
            edge_trace = go.Scatter(
                x=edge_x, y=edge_y,
                line=dict(width=0.6, color='rgba(148, 163, 184, 0.25)'),
                hoverinfo='none', mode='lines')
                
            node_x, node_y, node_text, node_size, node_color = [], [], [], [], []
            node_meta = metrics_df.set_index('node_id')
            
            for node in G_vis.nodes():
                x, y = pos[node]
                node_x.append(x)
                node_y.append(y)
                
                try:
                    score = node_meta.loc[node, 'sla_breach_score']
                    name = node_meta.loc[node, 'node_name']
                except KeyError:
                    score = 0
                    name = str(node)
                    
                node_text.append(f"{name}<br>SLA Score: {score:.1f}")
                node_size.append(10 + min(score / 150, 45) if score > 0 else 10)
                node_color.append(score)
                
            node_trace = go.Scatter(
                x=node_x, y=node_y, mode='markers', hoverinfo='text', text=node_text,
                marker=dict(
                    showscale=True, colorscale='Viridis', reversescale=False, color=node_color, size=node_size,
                    colorbar=dict(title=dict(text='SLA Risk Scale', font=dict(color='#94a3b8')), tickfont=dict(color='#94a3b8')),
                    line=dict(width=1, color='rgba(255,255,255,0.2)'))
            )
            fig_net = go.Figure(data=[edge_trace, node_trace],
                         layout=go.Layout(
                            showlegend=False, hovermode='closest',
                            paper_bgcolor='rgba(15, 23, 42, 0.4)', plot_bgcolor='rgba(0,0,0,0)',
                            margin=dict(b=0,l=0,r=0,t=0),
                            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
            )
            st.plotly_chart(fig_net, use_container_width=True)
        else:
            st.info("No breached corridors found for the selected filters.")

    # ─── View 2: Network Hub & Corridor Auditor ──────────────────────────────
    elif view == "🔍 Network Hub & Corridor Auditor":
        st.markdown('<div class="main-title">🔍 Network Hub & Corridor Auditor</div>', unsafe_allow_html=True)
        st.markdown('<div class="subtitle-text">Drill down into specific facility metrics and audit outgoing corridor latency.</div>', unsafe_allow_html=True)

        st.subheader("🏢 Hub Facility Inspector")
        # Let user select a hub based on sorted betweenness centrality
        hub_list = metrics_df.sort_values('betweenness_centrality', ascending=False)
        selected_hub_name = st.selectbox("Select Hub Facility to Inspect", hub_list['node_name'].tolist())
        selected_hub_row = hub_list[hub_list['node_name'] == selected_hub_name].iloc[0]
        selected_hub_id = selected_hub_row['node_id']

        col_h1, col_h2, col_h3, col_h4 = st.columns(4)
        col_h1.metric("Betweenness Centrality", f"{selected_hub_row['betweenness_centrality']:.4f}")
        col_h2.metric("In-Degree (Connections)", int(selected_hub_row['in_degree']))
        col_h3.metric("Out-Degree (Connections)", int(selected_hub_row['out_degree']))
        col_h4.metric("SLA Breach Score", f"{selected_hub_row['sla_breach_score']:.1f}")

        st.markdown("---")
        st.subheader(f"🚛 Outgoing Corridors from {selected_hub_name}")
        
        # Find edges originating from this hub
        hub_edges = edges_df[edges_df['source_center'] == selected_hub_id].copy()
        
        if len(hub_edges) > 0:
            hub_edges['SLA Status'] = hub_edges['median_segment_factor'].apply(lambda x: '⚠️ Breached' if x > 1.2 else '✅ Compliant')
            st.dataframe(
                hub_edges[['destination_name', 'route_type', 'time_of_day', 'median_segment_factor', 'median_segment_actual_time', 'median_segment_osrm_time', 'SLA Status']]
                .sort_values('median_segment_factor', ascending=False),
                column_config={
                    "destination_name": st.column_config.TextColumn("Destination Facility Hub"),
                    "route_type": st.column_config.TextColumn("Route Mode"),
                    "time_of_day": st.column_config.TextColumn("Dispatch Window"),
                    "median_segment_factor": st.column_config.NumberColumn("Median Delay Factor", format="%.2fx"),
                    "median_segment_actual_time": st.column_config.NumberColumn("Actual Time (m)"),
                    "median_segment_osrm_time": st.column_config.NumberColumn("OSRM Est. Time (m)"),
                    "SLA Status": st.column_config.TextColumn("SLA Status")
                },
                hide_index=True,
                use_container_width=True
            )
        else:
            st.info("No outgoing corridors found for the selected facility.")

        st.markdown("---")
        st.subheader("📊 Network-Wide Delay factor Distribution")
        fig_hist = px.histogram(
            filtered_edges,
            x='median_segment_factor', nbins=50,
            labels={'median_segment_factor': 'Median Segment Delay Factor (Actual / OSRM)'},
            color_discrete_sequence=['#818cf8']
        )
        fig_hist.add_vline(x=1.2, line_dash="dash", line_color="#fb7185",
                           annotation_text="SLA Breach Threshold (>1.2x)", annotation_font=dict(color="#fb7185"))
        fig_hist.update_layout(
            bargap=0.1, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#94a3b8', family='Outfit'),
            xaxis=dict(gridcolor='rgba(255,255,255,0.05)', tickfont=dict(color='#94a3b8')),
            yaxis=dict(gridcolor='rgba(255,255,255,0.05)', tickfont=dict(color='#94a3b8'))
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    # ─── View 3: Interactive ETA Predictor (ML) ──────────────────────────────
    elif view == "🔮 Interactive ETA Predictor (ML)":
        st.markdown('<div class="main-title">🔮 Interactive ETA Predictor</div>', unsafe_allow_html=True)
        st.markdown('<div class="subtitle-text">Estimate shipping actual duration using real-time node embeddings and graph features.</div>', unsafe_allow_html=True)

        if model is None:
            st.error("LogiMesh ML Model files (`eta_model.pkl`) not found. Please verify the modeling phase completed.")
        else:
            st.markdown("### 📝 Enter Shipment Specifications")
            
            col_in1, col_in2 = st.columns(2)
            
            with col_in1:
                # Select source facility
                hubs_sorted = metrics_df.sort_values('node_name')
                source_facility_name = st.selectbox("Origin Facility Hub", hubs_sorted['node_name'].tolist())
                source_facility_row = hubs_sorted[hubs_sorted['node_name'] == source_facility_name].iloc[0]
                source_id = source_facility_row['node_id']
                
                route_type = st.selectbox("Route Transport Mode", ["FTL", "Carting"])
                hour_of_day = st.slider("Dispatch Hour of Day", 0, 23, 12)
                
            with col_in2:
                osrm_distance = st.number_input("OSRM Estimated Distance (km)", min_value=1.0, max_value=2500.0, value=120.0)
                osrm_time = st.number_input("OSRM Standard Transit Time (minutes)", min_value=1.0, max_value=3000.0, value=100.0)
                
            st.markdown("---")
            if st.button("🔮 Calculate Predicted Transit Time & Analyze Risk", type="primary"):
                # Lookup node features
                node_feats_row = node_features[node_features['node_id'] == source_id]
                
                if len(node_feats_row) > 0:
                    node_row = node_feats_row.iloc[0]
                    
                    # 1. Base tabular features
                    route_type_encoded = 1 if route_type == "FTL" else 0
                    x_base = [osrm_time, osrm_distance, hour_of_day, route_type_encoded]
                    
                    # 2. Graph topology features (centralities)
                    graph_cols = ['betweenness_centrality', 'in_degree', 'out_degree', 'clustering_coefficient']
                    x_centrality = [node_row[col] for col in graph_cols]
                    
                    # 3. Spectral embeddings
                    x_emb = [node_row[f'emb_{i}'] for i in range(16)]
                    
                    # Merge features
                    X = np.array([x_base + x_centrality + x_emb])
                    
                    # Run LightGBM Model Predict
                    predicted_time = model.predict(X)[0]
                    predicted_delay_factor = predicted_time / osrm_time
                    
                    st.markdown("### 📊 Predictive Intelligence Report")
                    
                    col_res1, col_res2, col_res3 = st.columns(3)
                    
                    with col_res1:
                        st.metric("Predicted Actual Time", f"{predicted_time:.1f} mins", 
                                  delta=f"{predicted_time - osrm_time:+.1f} mins vs OSRM", delta_color="inverse")
                    with col_res2:
                        st.metric("Predicted Delay Factor", f"{predicted_delay_factor:.2fx}")
                    with col_res3:
                        sla_status = "⚠️ SLA BREACH RISK (High)" if predicted_delay_factor > 1.2 else "✅ COMPLIANT"
                        st.metric("Predicted SLA Status", sla_status, 
                                  delta="Breach if delay > 1.2x" if predicted_delay_factor > 1.2 else "Safe", 
                                  delta_color="normal" if predicted_delay_factor <= 1.2 else "inverse")
                        
                    # Detailed diagnostic text
                    st.info(f"**Diagnostic Output**: The origin facility **{source_facility_name}** holds a Betweenness Centrality of **{node_row['betweenness_centrality']:.4f}** (ranking in the top {100 * (metrics_df['betweenness_centrality'] > node_row['betweenness_centrality']).mean():.1f}% of network chokepoints). Our Graph-Enhanced LightGBM model integrated these topological features to adjust the raw OSRM time of **{osrm_time:.1f} mins** to a realistic actual projection of **{predicted_time:.1f} mins**.")
                else:
                    st.error("Selected origin facility features not found in precomputed embeddings.")

            # Model Performance Section
            st.markdown("---")
            st.subheader("🤖 Predictor Baseline Comparison")
            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            col_m1.metric("Baseline MAE", "54.58 mins")
            col_m2.metric("LogiMesh MAE", "43.25 mins", delta="-11.33 mins", delta_color="normal")
            col_m3.metric("Baseline Accuracy@15%", "43.56%")
            col_m4.metric("LogiMesh Accuracy@15%", "50.98%", delta="+7.42%", delta_color="normal")

    # ─── View 4: Route Mode Optimizer ─────────────────────────────────────────
    elif view == "🚛 Route Mode Optimizer":
        st.markdown('<div class="main-title">🚛 Route Mode Optimizer</div>', unsafe_allow_html=True)
        st.markdown('<div class="subtitle-text">Quantify cost-time tradeoffs of FTL vs. Carting modes using historical network delay factors.</div>', unsafe_allow_html=True)

        st.subheader("🚛 Mode Comparison Chart")
        
        # Render facet comparison chart
        strategy_df = edges_df.copy()
        strategy_df['dist_bucket'] = pd.qcut(strategy_df['median_segment_osrm_distance'], q=3, labels=['Short', 'Medium', 'Long'])
        strategy_grouped = strategy_df.groupby(['dist_bucket', 'time_of_day', 'route_type']).agg(
            avg_delay_factor=('median_segment_factor', 'mean')
        ).reset_index()

        fig_strat = px.bar(
            strategy_grouped,
            x='time_of_day', y='avg_delay_factor', color='route_type', barmode='group',
            facet_col='dist_bucket',
            labels={'avg_delay_factor': 'Avg Delay Factor', 'time_of_day': 'Time of Day', 'route_type': 'Mode'},
            color_discrete_map={'FTL': '#818cf8', 'Carting': '#f43f5e'}
        )
        fig_strat.add_hline(y=1.2, line_dash="dot", line_color="#fb7185",
                             annotation_text="SLA Threshold", annotation_font=dict(color="#fb7185"))
        fig_strat.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#94a3b8', family='Outfit'),
            legend=dict(font=dict(color='#e2e8f0')),
            xaxis=dict(tickfont=dict(color='#94a3b8')),
            yaxis=dict(gridcolor='rgba(255,255,255,0.05)', tickfont=dict(color='#94a3b8'))
        )
        st.plotly_chart(fig_strat, use_container_width=True)

        st.markdown("---")
        st.subheader("💡 Mode Selection Playbook")
        
        # Interactive mode helper
        col_play1, col_play2 = st.columns(2)
        with col_play1:
            play_dist = st.selectbox("Corridor Distance Classification", ["Short (< 50km)", "Medium (50km - 150km)", "Long (> 150km)"])
            play_time = st.selectbox("Dispatch Time of Day", ["Morning", "Afternoon", "Evening", "Night"])
            
        with col_play2:
            st.markdown("#### LogiMesh Route Recommendation")
            # Decision engine logic
            rec_mode = "Carting"
            rec_reason = "Efficient and cost-effective for short distances."
            
            if "Long" in play_dist:
                rec_mode = "FTL"
                rec_reason = "Carting exhibits high instability (average delay factor > 4.7) on long routes during PM slots. Mandating FTL secures SLA compliance."
            elif "Medium" in play_dist:
                if play_time in ["Evening", "Morning"]:
                    rec_mode = "FTL"
                    rec_reason = "During peak transit slots, FTL maintains a delay factor of 2.08x compared to Carting's 2.54x on medium corridors."
                else:
                    rec_mode = "Carting"
                    rec_reason = "During off-peak daylight hours, Carting delay factors are within 4% of FTL, saving unit mode cost."
            else: # Short distance
                if play_time == "Evening":
                    rec_mode = "FTL"
                    rec_reason = "Evening congestion at local sorting hubs causes Carting bottlenecks. Switch to FTL to bypass hub delays."
                else:
                    rec_mode = "Carting"
                    rec_reason = "Carting operates well within SLA thresholds during off-peak windows on short corridors."
                    
            st.success(f"**Recommended Mode**: {rec_mode}")
            st.info(f"**Reasoning**: {rec_reason}")

    # ─── View 5: Strategic Memo ───────────────────────────────────────────────
    elif view == "📝 Strategic Memo":
        st.markdown('<div class="main-title">📝 Strategic Memo</div>', unsafe_allow_html=True)
        st.markdown('<div class="subtitle-text">Rebranded executive summary report prepared for the Chief of Network Operations.</div>', unsafe_allow_html=True)
        
        memo_path = "reports/strategy_memo.md"
        if os.path.exists(memo_path):
            with open(memo_path, 'r', encoding='utf-8') as f:
                content = f.read()
            st.markdown(content)
        else:
            st.error("Strategy Memo file (`reports/strategy_memo.md`) not found. Re-run document rebranding script.")

except FileNotFoundError as e:
    st.error(f"Processed data files not found ({e}). Please run the data pipeline, graph analysis, and modeling scripts first.")
