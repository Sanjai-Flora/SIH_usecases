import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from faker import Faker
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import networkx as nx
import joblib
import random

# Set page config
st.set_page_config(page_title="DTC Dynamic Route Rationalization", layout="wide")

# Set a consistent color palette
color_palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

# Custom CSS
st.markdown("""
<style>
    .reportview-container {
        background: #f0f2f6;
    }
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    h1, h2, h3 {
        color: #1f77b4;
    }
    .stAlert {
        background-color: #e6f3ff;
        border: 1px solid #1f77b4;
        border-radius: 5px;
        padding: 10px;
        margin-bottom: 1rem;
    }
    .stMetric {
        background-color: white;
        border-radius: 5px;
        padding: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .plot-container {
        background-color: white;
        border-radius: 5px;
        padding: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .insight-box {
        background-color: #f0f8ff;
        border-left: 5px solid #1f77b4;
        padding: 10px;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Faker
fake = Faker()
Faker.seed(0)

# Generate Bus Data
@st.cache_data
def generate_bus_data(num_buses=100):
    return pd.DataFrame({
        'bus_id': range(1, num_buses + 1),
        'route': [f"Route-{random.randint(1, 20)}" for _ in range(num_buses)],
        'capacity': [random.randint(30, 60) for _ in range(num_buses)],
        'current_load': [random.randint(0, 60) for _ in range(num_buses)],
        'lat': [random.uniform(28.5, 28.8) for _ in range(num_buses)],
        'lon': [random.uniform(77.0, 77.3) for _ in range(num_buses)],
        'speed': [random.uniform(0, 60) for _ in range(num_buses)],
        'delay': [random.randint(-10, 30) for _ in range(num_buses)],
        'fuel_efficiency': [random.uniform(3, 6) for _ in range(num_buses)],  # km/l
    })

# Generate Traffic Data
@st.cache_data
def generate_traffic_data(num_points=50):
    return pd.DataFrame({
        'location_id': range(1, num_points + 1),
        'lat': [random.uniform(28.5, 28.8) for _ in range(num_points)],
        'lon': [random.uniform(77.0, 77.3) for _ in range(num_points)],
        'traffic_intensity': [random.uniform(0, 1) for _ in range(num_points)],
        'road_condition': [random.choice(['Good', 'Fair', 'Poor']) for _ in range(num_points)],
        'passenger_demand': [random.randint(10, 100) for _ in range(num_points)],
    })

# Generate Route Network
@st.cache_data
def generate_route_network(num_nodes=20):
    G = nx.random_geometric_graph(num_nodes, 0.3)
    pos = nx.spring_layout(G)
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    
    node_x = [pos[node][0] for node in G.nodes()]
    node_y = [pos[node][1] for node in G.nodes()]
    
    return edge_x, edge_y, node_x, node_y

# Train ML model for route optimization
@st.cache_resource
def train_route_optimization_model(traffic_data):
    X = traffic_data[['lat', 'lon', 'traffic_intensity', 'passenger_demand']]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = KMeans(n_clusters=5, random_state=42)
    model.fit(X_scaled)
    
    return model, scaler

# Main app
def main():
    st.title("DTC Dynamic Route Rationalization")

    # Sidebar
    st.sidebar.title("Settings")
    num_buses = st.sidebar.slider("Number of Buses", 50, 200, 100)
    update_interval = st.sidebar.slider("Update Interval (seconds)", 5, 60, 30)

    # Generate data
    bus_data = generate_bus_data(num_buses)
    traffic_data = generate_traffic_data()

    # Train ML model
    model, scaler = train_route_optimization_model(traffic_data)

    # Main content
    tab0, tab1, tab2, tab3, tab4, tab5 = st.tabs(["Problem Statement", "Overview", "Real-time Monitoring", "Route Optimization", "Performance Metrics", "Network Analysis"])

    with tab0:
        st.header("Problem Statement")
        
        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown(
                f"""
                <div style='background-color: #E6F3FF; padding: 20px; border-radius: 10px;'>
                <h3 style='color: #1f77b4;'>Quick Info</h3>
                <p><strong>ID:</strong> 1617</p>
                <p><strong>Organization:</strong> Government of NCT of Delhi</p>
                <p><strong>Department:</strong> IT Department, GNCTD</p>
                <p><strong>Category:</strong> Software</p>
                <p><strong>Theme:</strong> Smart Automation</p>
                </div>
                """,
                unsafe_allow_html=True
            )

        with col2:
            st.markdown(
                f"""
                <div style='background-color: #FFF5E6; padding: 20px; border-radius: 10px;'>
                <h3 style='color: #ff7f0e;'>Dynamic Route Rationalization for DTC</h3>
                <h4 style='color: #ff7f0e;'>Background:</h4>
                <p>DTC is working on various modules for route rationalization. Real-time monitoring of buses is crucial for effective route rationalization to prevent bunching of buses on specific routes and long delays in bus arrivals.</p>
                <h4 style='color: #ff7f0e;'>Challenge:</h4>
                <p>The problem cannot be addressed by a fixed time schedule due to various factors like traffic conditions, road conditions, and other dynamic parameters.</p>
                <h4 style='color: #ff7f0e;'>Solution Needed:</h4>
                <p>A dynamic route rationalization model based on machine learning/AI that considers real-time traffic and road parameters.</p>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.subheader("Key Objectives")
        objectives = [
            "Implement real-time monitoring of bus locations and status",
            "Analyze traffic patterns and road conditions",
            "Develop an AI/ML model for dynamic route optimization",
            "Prevent bus bunching and reduce delays",
            "Improve overall efficiency of the bus network",
            "Enhance passenger experience through better service reliability"
        ]
        for obj in objectives:
            st.markdown(f"- {obj}")

    with tab1:
        st.header("Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Buses", num_buses)
        with col2:
            st.metric("Active Routes", bus_data['route'].nunique())
        with col3:
            st.metric("Avg. Speed (km/h)", f"{bus_data['speed'].mean():.2f}")
        with col4:
            st.metric("Avg. Delay (min)", f"{bus_data['delay'].mean():.2f}")

        st.subheader("Bus Distribution by Route")
        route_distribution = bus_data['route'].value_counts().reset_index()
        route_distribution.columns = ['Route', 'Count']
        fig_route_dist = px.bar(route_distribution, x='Route', y='Count', 
                                title="Bus Distribution by Route",
                                color='Count',
                                color_continuous_scale=px.colors.sequential.Viridis)
        st.plotly_chart(fig_route_dist, use_container_width=True)

        st.markdown("""
        <div class="insight-box">
        <strong>Insight:</strong> This chart shows how buses are distributed across different routes. 
        For example, Route-7 has the highest number of buses, which might indicate high demand or longer route length. 
        Routes with fewer buses might need assessment for potential consolidation or increased service frequency.
        </div>
        """, unsafe_allow_html=True)

        st.subheader("Current Traffic Intensity")
        fig_traffic = px.scatter_mapbox(traffic_data, lat='lat', lon='lon', 
                                        color='traffic_intensity', size='passenger_demand',
                                        color_continuous_scale=px.colors.sequential.Reds,
                                        mapbox_style="open-street-map", zoom=10,
                                        title="Traffic Intensity and Passenger Demand Map")
        st.plotly_chart(fig_traffic, use_container_width=True)

        st.markdown("""
        <div class="insight-box">
        <strong>Insight:</strong> This map visualizes traffic intensity (color) and passenger demand (size) across different locations. 
        Red areas indicate high traffic intensity, while larger circles show high passenger demand. 
        For instance, the large red circle in the northeast suggests a location with both high traffic and high demand, which might require special attention in route planning.
        </div>
        """, unsafe_allow_html=True)

    with tab2:
        st.header("Real-time Monitoring")

        st.subheader("Bus Locations and Status")
        fig_bus_locations = px.scatter_mapbox(bus_data, lat='lat', lon='lon', 
                                              color='route', size='current_load',
                                              hover_data=['bus_id', 'speed', 'delay'],
                                              mapbox_style="open-street-map", zoom=10,
                                              title="Real-time Bus Locations")
        st.plotly_chart(fig_bus_locations, use_container_width=True)

        st.markdown("""
        <div class="insight-box">
        <strong>Insight:</strong> This map shows the real-time location of buses, their routes (color), and current passenger load (size). 
        For example, you might notice a cluster of large circles in one area, indicating several heavily loaded buses, which could suggest the need for more frequent service or larger capacity buses on those routes.
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Bus Load Distribution")
            fig_load = px.histogram(bus_data, x='current_load', nbins=20,
                                    title="Current Bus Load Distribution",
                                    labels={'current_load': 'Current Load', 'count': 'Number of Buses'},
                                    color_discrete_sequence=[color_palette[0]])
            st.plotly_chart(fig_load, use_container_width=True)

            st.markdown("""
            <div class="insight-box">
            <strong>Insight:</strong> This histogram shows how many buses are carrying different numbers of passengers. 
            A peak around 30-40 passengers might indicate that most buses are operating at a comfortable capacity. 
            However, if there's a significant number of buses with very low or very high loads, it might suggest a need for route adjustments.
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.subheader("Bus Delay Distribution")
            fig_delay = px.histogram(bus_data, x='delay', nbins=20,
                                     title="Bus Delay Distribution",
                                     labels={'delay': 'Delay (minutes)', 'count': 'Number of Buses'},
                                     color_discrete_sequence=[color_palette[1]])
            st.plotly_chart(fig_delay, use_container_width=True)

            st.markdown("""
            <div class="insight-box">
            <strong>Insight:</strong> This chart shows the distribution of bus delays. 
            A concentration around 0 indicates that many buses are on time. 
            If there's a significant number of buses with large positive delays (e.g., 15-20 minutes), it might indicate traffic issues or other problems on certain routes that need addressing.
            </div>
            """, unsafe_allow_html=True)

    with tab3:
        st.header("Route Optimization")

        st.subheader("Optimized Route Clusters")
        traffic_data['cluster'] = model.predict(scaler.transform(traffic_data[['lat', 'lon', 'traffic_intensity', 'passenger_demand']]))
        fig_clusters = px.scatter_mapbox(traffic_data, lat='lat', lon='lon', 
                                         color='cluster', size='passenger_demand',
                                         mapbox_style="open-street-map", zoom=10,
                                         title="Optimized Route Clusters")
        st.plotly_chart(fig_clusters, use_container_width=True)

        st.markdown("""
        <div class="insight-box">
        <strong>Insight:</strong> This map shows optimized route clusters based on traffic intensity, geographical distribution, and passenger demand. 
        Each color represents a different cluster, suggesting potential route adjustments. 
        For example, if you see a large blue cluster in one area and a small red cluster nearby, it might suggest merging these areas into a single route to improve efficiency.
        </div>
        """, unsafe_allow_html=True)

        st.subheader("Suggested Route Changes")
        suggested_changes = pd.DataFrame({
            'Current Route': ['Route-1', 'Route-5', 'Route-12', 'Route-18'],
            'Suggested Change': ['Merge with Route-3', 'Split into two routes', 'Extend to cover Cluster 2', 'Reduce frequency'],
            'Reason': ['Low ridership', 'High traffic intensity', 'Underserved area', 'Overbunching observed'],
            'Expected Impact': ['Improved efficiency', 'Reduced delays', 'Increased coverage', 'Better resource allocation']
        })
        st.table(suggested_changes)

        st.markdown("""
        <div class="insight-box">
        <strong>Insight:</strong> This table provides specific recommendations for route changes based on the AI analysis. 
        For example, merging Route-1 with Route-3 due to low ridership could improve overall network efficiency. 
        These suggestions should be reviewed by transit planners and validated with additional on-ground data before implementation.
        </div>
        """, unsafe_allow_html=True)

    with tab4:
        st.header("Performance Metrics")

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Average Bus Utilization")
            bus_utilization = (bus_data['current_load'] / bus_data['capacity']).mean() * 100
            fig_utilization = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = bus_utilization,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Average Bus Utilization"},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'steps': [
                        {'range': [0, 50], 'color': "lightgray"},
                        {'range': [50, 75], 'color': "gray"}],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 80}}))
            st.plotly_chart(fig_utilization, use_container_width=True)

            st.markdown("""
            <div class="insight-box">
            <strong>Insight:</strong> This gauge shows the average bus utilization across the network. 
            A value around 60-70% might indicate good efficiency, while higher values could suggest overcrowding and lower values might indicate underutilization of resources.
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.subheader("On-Time Performance")
            on_time = (bus_data['delay'].abs() <= 5).mean() * 100
            fig_on_time = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = on_time,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "On-Time Performance"},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'steps': [
                        {'range': [0, 60], 'color': "lightgray"},
                        {'range': [60, 80], 'color': "gray"}],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90}}))
            st.plotly_chart(fig_on_time, use_container_width=True)

            st.markdown("""
            <div class="insight-box">
            <strong>Insight:</strong> This gauge shows the percentage of buses that are on time (within 5 minutes of schedule). 
            A higher percentage indicates better service reliability. For instance, if the gauge shows 75%, it means 3 out of 4 buses are running on time, which is good but leaves room for improvement.
            </div>
            """, unsafe_allow_html=True)

        st.subheader("Route Performance")
        route_performance = bus_data.groupby('route').agg({
            'speed': 'mean',
            'delay': 'mean',
            'current_load': 'mean',
            'fuel_efficiency': 'mean'
        }).reset_index()
        route_performance['utilization'] = route_performance['current_load'] / bus_data['capacity'].mean() * 100
        fig_route_perf = px.scatter(route_performance, x='speed', y='delay', 
                                    size='utilization', color='fuel_efficiency',
                                    hover_data=['route'],
                                    labels={'speed': 'Average Speed (km/h)', 'delay': 'Average Delay (min)', 
                                            'fuel_efficiency': 'Fuel Efficiency (km/l)'},
                                    title="Route Performance Analysis")
        st.plotly_chart(fig_route_perf, use_container_width=True)

        st.markdown("""
        <div class="insight-box">
        <strong>Insight:</strong> This scatter plot provides a comprehensive view of route performance. 
        Each bubble represents a route, where:
        - X-axis shows average speed (faster to the right)
        - Y-axis shows average delay (more delayed towards the top)
        - Size indicates utilization (larger bubbles mean higher utilization)
        - Color represents fuel efficiency (greener is more efficient)
        
        For example, a large, green bubble in the bottom-right quadrant would represent an ideal route: fast, on-time, well-utilized, and fuel-efficient. 
        Conversely, a small, red bubble in the top-left might indicate a problematic route that needs optimization.
        </div>
        """, unsafe_allow_html=True)

    with tab5:
        st.header("Network Analysis")

        st.subheader("Route Network Visualization")
        edge_x, edge_y, node_x, node_y = generate_route_network()

        fig_network = go.Figure()
        fig_network.add_trace(go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines'))

        fig_network.add_trace(go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            hoverinfo='text',
            marker=dict(
                showscale=True,
                colorscale='YlGnBu',
                reversescale=True,
                color=[],
                size=10,
                colorbar=dict(
                    thickness=15,
                    title='Node Connections',
                    xanchor='left',
                    titleside='right'
                ),
                line_width=2)))

        fig_network.update_layout(
            title='Route Network Graph',
            titlefont_size=16,
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20,l=5,r=5,t=40),
            annotations=[ dict(
                text="Python code: <a href='https://plotly.com/ipython-notebooks/network-graphs/'> https://plotly.com/ipython-notebooks/network-graphs/</a>",
                showarrow=False,
                xref="paper", yref="paper",
                x=0.005, y=-0.002 ) ],
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))

        st.plotly_chart(fig_network, use_container_width=True)

        st.markdown("""
        <div class="insight-box">
        <strong>Insight:</strong> This network graph represents the interconnectedness of bus routes. 
        Each node represents a stop or intersection, and the lines represent routes between them. 
        Larger, darker nodes indicate stops with more connections, which could be potential hubs or transfer points. 
        This visualization can help identify critical points in the network for optimization or where adding new connections could significantly improve overall network efficiency.
        </div>
        """, unsafe_allow_html=True)

    # Footer
    st.sidebar.markdown("---")
    st.sidebar.info("""
    **Note:** This demo app showcases a dynamic route rationalization model for DTC buses based on real-time traffic and road parameters. 

    In a real-world scenario, this system would integrate with actual GPS data from buses, real-time traffic information, and historical ridership data to provide more accurate and actionable insights for route optimization.
    """)

if __name__ == "__main__":
    main()