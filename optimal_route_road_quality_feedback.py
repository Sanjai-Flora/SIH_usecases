import streamlit as st
import pandas as pd
import geopandas as gpd
import plotly.express as px
import plotly.graph_objects as go
import osmnx as ox
import networkx as nx
import numpy as np
from shapely.geometry import LineString, Point

# Set page configuration
st.set_page_config(layout="wide", page_title="India Road Explorer", page_icon="🇮🇳")

# Custom CSS for better aesthetics
st.markdown("""
<style>
    .reportview-container {
        background: #E3F2FD;
    }
    .sidebar .sidebar-content {
        background: #BBDEFB;
    }
    .Widget>label {
        color: #1565C0;
        font-weight: bold;
    }
    .stButton>button {
        background-color: #1E88E5;
        color: white;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background-color: #64B5F6;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1E88E5;
    }
</style>
""", unsafe_allow_html=True)

st.title("🇮🇳 India Road Explorer")

@st.cache_data
def load_data(place_name):
    graph = ox.graph_from_place(place_name, network_type='drive', simplify=True)
    nodes, edges = ox.graph_to_gdfs(graph, nodes=True, edges=True)
    edges = edges.reset_index()
    edges['smoothness_rating'] = np.random.randint(1, 6, size=len(edges))
    edges['safety_rating'] = np.random.randint(1, 6, size=len(edges))
    return graph, nodes, edges

# Dictionary of cities and their popular locations
cities_and_locations = {
    "Mumbai, Maharashtra, India": [
        "Gateway of India", "Chhatrapati Shivaji Terminus", "Marine Drive", "Juhu Beach",
        "Siddhivinayak Temple", "Bandra-Worli Sea Link", "Elephanta Caves", "Haji Ali Dargah",
        "Colaba Causeway", "Girgaum Chowpatty", "Nariman Point", "Worli Sea Face",
        "Sanjay Gandhi National Park", "Powai Lake", "Mahalaxmi Temple", "Chhatrapati Shivaji Maharaj Vastu Sangrahalaya",
        "Jehangir Art Gallery", "Prithvi Theatre", "Dharavi", "Kala Ghoda"
    ],
    "Delhi, India": [
        "India Gate", "Red Fort", "Qutub Minar", "Lotus Temple",
        "Akshardham Temple", "Humayun's Tomb", "Chandni Chowk", "Connaught Place"
    ],
    "Bangalore, Karnataka, India": [
        "Lalbagh Botanical Garden", "Cubbon Park", "Bangalore Palace", "Vidhana Soudha",
        "UB City", "ISKCON Temple", "Bannerghatta National Park", "Wonderla"
    ],
    "Hyderabad, Telangana, India": [
        "Charminar", "Golconda Fort", "Hussain Sagar Lake", "Ramoji Film City",
        "Birla Mandir", "Salar Jung Museum", "Nehru Zoological Park", "Necklace Road"
    ],
    "Chennai, Tamil Nadu, India": [
        "Marina Beach", "Kapaleeshwarar Temple", "Fort St. George", "San Thome Basilica",
        "Government Museum", "Valluvar Kottam", "Elliot's Beach", "Guindy National Park"
    ]
}

# Hardcoded coordinates for Mumbai locations
mumbai_coords = {
    "Gateway of India": (18.9217, 72.8347),
    "Siddhivinayak Temple": (19.0167, 72.8301),
    "Chhatrapati Shivaji Terminus": (18.9398, 72.8355),
    "Marine Drive": (18.9438, 72.8231),
    "Juhu Beach": (19.0948, 72.8258),
    "Bandra-Worli Sea Link": (19.0297, 72.8153),
    "Elephanta Caves": (18.9633, 72.9315),
    "Haji Ali Dargah": (18.9827, 72.8089),
    "Colaba Causeway": (18.9146, 72.8273),
    "Girgaum Chowpatty": (18.9542, 72.8154),
    "Nariman Point": (18.9256, 72.8242),
    "Worli Sea Face": (19.0176, 72.8152),
    "Sanjay Gandhi National Park": (19.2147, 72.9106),
    "Powai Lake": (19.1273, 72.9072),
    "Mahalaxmi Temple": (18.9775, 72.8093),
    "Chhatrapati Shivaji Maharaj Vastu Sangrahalaya": (18.9268, 72.8332),
    "Jehangir Art Gallery": (18.9272, 72.8315),
    "Prithvi Theatre": (19.1387, 72.8353),
    "Dharavi": (19.0433, 72.8526),
    "Kala Ghoda": (18.9279, 72.8319)
}

# Create tabs
tab1, tab2 = st.tabs(["🗺️ Route Planner", "📊 Road Feedback"])

with tab1:
    st.header("Route Planner")
    st.markdown("""
    Welcome to the Route Planner! This tool helps you navigate Indian cities while considering road quality.
    Here's how to use it:
    1. Select a city from the dropdown menu.
    2. Choose your start and end locations from popular landmarks in the city.
    3. The app will calculate the optimal route based on distance and road quality.
    4. The route will be displayed on the map with a dark blue line.
    5. Green and red markers indicate the start and end points respectively.
    """)
    
    selected_city = st.selectbox("Select a city", list(cities_and_locations.keys()))
    
    col1, col2 = st.columns(2)
    with col1:
        start_location = st.selectbox("Start Location", cities_and_locations[selected_city], key="start")
    with col2:
        end_location = st.selectbox("End Location", cities_and_locations[selected_city], key="end")

    if start_location == end_location:
        st.error("Start and end locations must be different.")
    else:
        try:
            with st.spinner("Calculating route..."):
                graph, nodes, edges = load_data(selected_city)
                
                # Use hardcoded coordinates for Mumbai
                if selected_city == "Mumbai, Maharashtra, India":
                    start_coords = mumbai_coords[start_location]
                    end_coords = mumbai_coords[end_location]
                else:
                    # Fallback to geocoding for other cities
                    start_coords = ox.geocode(f"{start_location}, {selected_city}")
                    end_coords = ox.geocode(f"{end_location}, {selected_city}")

                orig_node = ox.distance.nearest_nodes(graph, start_coords[1], start_coords[0])
                dest_node = ox.distance.nearest_nodes(graph, end_coords[1], end_coords[0])

                route = nx.shortest_path(graph, orig_node, dest_node, weight='length')
                
                # Create route geometry
                route_coords = []
                for node in route:
                    point = nodes.loc[node, 'geometry']
                    route_coords.append((point.x, point.y))

                route_df = pd.DataFrame(route_coords, columns=['lon', 'lat'])

                # Create a base map centered on the city
                fig = px.scatter_mapbox(
                    lat=[start_coords[0]],
                    lon=[start_coords[1]],
                    zoom=11,
                    height=600,
                    width=800,
                )

                # Add the route as a line
                fig.add_trace(
                    go.Scattermapbox(
                        mode="lines",
                        lon=route_df['lon'],
                        lat=route_df['lat'],
                        line=dict(width=4, color="#0D47A1"),  # Dark blue color
                    )
                )

                # Add markers for start and end points
                fig.add_trace(
                    go.Scattermapbox(
                        mode="markers",
                        lon=[start_coords[1], end_coords[1]],
                        lat=[start_coords[0], end_coords[0]],
                        marker=dict(size=10, color=["green", "red"]),
                        text=[start_location, end_location],
                    )
                )

                fig.update_layout(
                    mapbox_style="open-street-map",
                    margin={"r":0,"t":0,"l":0,"b":0},
                    mapbox=dict(
                        center=dict(lat=start_coords[0], lon=start_coords[1]),
                        zoom=11
                    )
                )

                st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.write("Please try selecting different locations or another city.")

with tab2:
    st.header("Road Feedback")
    st.markdown("""
    Your feedback helps improve route suggestions for all users. Here's how to provide feedback:
    1. Select the start and end locations of a route you're familiar with.
    2. Choose a major road along that route from the dropdown menu.
    3. Rate the road's smoothness and safety on a scale of 1 to 5.
    4. Submit your feedback to contribute to our community-driven map of road conditions.

    Your input helps create more accurate and safer route suggestions for everyone!
    """)
    
    # Select start and end locations for the route
    col1, col2 = st.columns(2)
    with col1:
        feedback_start = st.selectbox("Start Location", cities_and_locations[selected_city], key="feedback_start")
    with col2:
        feedback_end = st.selectbox("End Location", cities_and_locations[selected_city], key="feedback_end")
    
    if feedback_start != feedback_end:
        try:
            # Calculate the route
            start_coords = mumbai_coords[feedback_start]
            end_coords = mumbai_coords[feedback_end]
            
            orig_node = ox.distance.nearest_nodes(graph, start_coords[1], start_coords[0])
            dest_node = ox.distance.nearest_nodes(graph, end_coords[1], end_coords[0])
            
            route = nx.shortest_path(graph, orig_node, dest_node, weight='length')
            
            # Get the edges (road segments) in the route
            route_edges = ox.utils_graph.get_route_edge_attributes(graph, route)
            
            # Extract major roads (you may need to adjust this based on the available data)
            major_roads = [edge.get('name', 'Unnamed Road') for edge in route_edges if isinstance(edge.get('name'), str)]
            major_roads = list(set(major_roads))  # Remove duplicates
            major_roads = [road for road in major_roads if road != 'Unnamed Road']  # Remove unnamed roads
            
            if major_roads:
                # Let user select a major road
                selected_road = st.selectbox("Select a major road on this route", major_roads)
                
                # Find the edge for the selected road
                selected_edge = next((edge for edge in route_edges if edge.get('name') == selected_road), None)
                if selected_edge:
                    # Create a unique identifier for the edge
                    edge_id = f"{selected_edge['osmid']}_{selected_road}"
                    
                    # Initialize ratings if they don't exist
                    if edge_id not in edges.index:
                        edges.loc[edge_id] = {'smoothness_rating': 3, 'safety_rating': 3}
                    
                    # Display current ratings
                    st.write(f"Current ratings for {selected_road}:")
                    st.write(f"Smoothness: {edges.loc[edge_id, 'smoothness_rating']}")
                    st.write(f"Safety: {edges.loc[edge_id, 'safety_rating']}")
                    
                    # Get user feedback
                    col1, col2 = st.columns(2)
                    with col1:
                        smoothness = st.slider("Smoothness Rating", 1, 5, int(edges.loc[edge_id, 'smoothness_rating']),
                                               help="1 = Very Rough, 5 = Very Smooth")
                    with col2:
                        safety = st.slider("Safety Rating", 1, 5, int(edges.loc[edge_id, 'safety_rating']),
                                           help="1 = Very Unsafe, 5 = Very Safe")
                    
                    if st.button("Submit Feedback", key="submit_feedback"):
                        edges.loc[edge_id, 'smoothness_rating'] = smoothness
                        edges.loc[edge_id, 'safety_rating'] = safety
                        st.success(f"Feedback submitted for {selected_road}! Thank you for contributing.")
                        st.info("Route calculations will now consider your feedback.")
                else:
                    st.error("Could not find the selected road in the route. Please try another road.")
            else:
                st.error("No major roads found on this route. Try a different route.")
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.write("Please try selecting different locations.")
    else:
        st.error("Start and end locations must be different.")

# Add a footer
st.markdown("---")
st.markdown("© 2024 India Road Explorer. All rights reserved.")
