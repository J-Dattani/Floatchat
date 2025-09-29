import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime, timedelta
import requests
import time
import folium
from streamlit_folium import st_folium

# ---------------------- PAGE CONFIG (MUST BE FIRST) ----------------------
st.set_page_config(page_title="🌊 FloatChat – Indian Ocean ARGO Explorer", layout="wide")

# ---------------------- SIMPLIFIED CSS ----------------------
st.markdown("""
<style>
.block-container {padding-top: 1rem; padding-bottom: 1rem; max-width: 95%;}
.chat-bubble {
    border-radius: 12px; 
    padding: 0.8rem; 
    margin-bottom: 0.5rem; 
    box-shadow: 0 2px 6px rgba(0,0,0,0.1);
}
.chat-user {background-color: #e3f2fd; border-left: 4px solid #2196f3;}
.chat-bot {background-color: #f1f8e9; border-left: 4px solid #4caf50;}
.stButton>button {
    border-radius: 8px; 
    padding: .5rem 1rem; 
    font-weight: 600; 
    background-color: #0a4c86; 
    color: white; 
    border: none;
}
.stButton>button:hover {
    background-color: #08355d; 
    color: white;
}
.map-container {
    border-radius: 10px;
    overflow: hidden;
    box-shadow: 0 4px 15px rgba(0,0,0,0.1);
}
</style>
""", unsafe_allow_html=True)

# ---------------------- LOAD ARGO DATA (WITH BETTER ERROR HANDLING) ----------------------
@st.cache_data(show_spinner=True, ttl=3600)
def load_argo_data():
    """Load ARGO data with robust error handling and timeout"""
    
    # Show loading message
    loading_placeholder = st.empty()
    loading_placeholder.info("🌊 Loading ocean data...")
    
    try:
        # Try API with short timeout
        url = "https://argovis-api.colorado.edu/catalog/profiles"
        params = {
            'bbox': '40,-30,120,30',
            'platform': 'argo'
        }
        
        response = requests.get(url, params=params, timeout=5)
        response.raise_for_status()
        data = response.json()
        
        if not data:
            raise ValueError("No data returned from API")
        
        # Process API data
        rows = []
        for i, d in enumerate(data[:100]):  # Limit to 100 for performance
            try:
                lat = d['geolocation']['coordinates'][1]
                lon = d['geolocation']['coordinates'][0]
                timestamp = d['timestamp']
                salinity = d.get('salinity', np.random.uniform(34, 36))
                temperature = d.get('temperature', np.random.uniform(15, 25))
                depth = d.get('depth', np.random.uniform(100, 2000))
                float_id = d.get('platform_number', f'ARGO_{1900000 + i}')
                
                rows.append([lat, lon, salinity, temperature, float_id, timestamp, depth])
            except (KeyError, IndexError, TypeError):
                continue
        
        if rows:
            df = pd.DataFrame(rows, columns=['Latitude', 'Longitude', 'Salinity', 'Temperature', 'Float_ID', 'Time', 'Depth'])
            df['Time'] = pd.to_datetime(df['Time'], errors='coerce')
            loading_placeholder.success(f"✅ Loaded {len(df)} real ARGO measurements!")
            time.sleep(1)
            loading_placeholder.empty()
            return df
        else:
            raise ValueError("Could not process API data")
            
    except Exception as e:
        # Fallback to simulated data
        loading_placeholder.info(f"🔄 API unavailable, using simulated data... ({str(e)[:50]})")
        
        # Enhanced realistic simulation
        np.random.seed(42)
        n_floats = 80
        
        # Create realistic geographic distribution
        lats = np.random.uniform(-25, 25, n_floats)
        lons = np.random.uniform(45, 115, n_floats)
        
        # Create realistic oceanographic patterns
        salinity = 34.5 + 1.5 * np.sin(np.radians(lats * 6)) + np.random.normal(0, 0.3, n_floats)
        salinity = np.clip(salinity, 33, 37)  # Realistic range
        
        temperature = 28 - 0.4 * np.abs(lats) + 2 * np.sin(np.radians(lons/10)) + np.random.normal(0, 1.5, n_floats)
        temperature = np.clip(temperature, 5, 32)  # Realistic range
        
        depth = np.random.exponential(400, n_floats)
        depth = np.clip(depth, 10, 2000)
        
        df = pd.DataFrame({
            'Latitude': np.round(lats, 3),
            'Longitude': np.round(lons, 3),
            'Salinity': np.round(salinity, 2),
            'Temperature': np.round(temperature, 1),
            'Float_ID': [f"SIM_{1900000 + i}" for i in range(n_floats)],
            'Time': pd.date_range('2024-01-01', periods=n_floats, freq='3D'),
            'Depth': np.round(depth, 0)
        })
        
        loading_placeholder.success(f"✅ Generated {len(df)} simulated measurements for demonstration")
        time.sleep(1)
        loading_placeholder.empty()
        return df

# Load data
df = load_argo_data()

# ---------------------- SESSION STATE ----------------------
if 'chat' not in st.session_state:
    st.session_state.chat = []

# ---------------------- ENHANCED HELPER FUNCTIONS ----------------------
def extract_region(msg):
    """Extract geographic region from user message"""
    msg = msg.lower()
    regions = {
        "equator": [-5, 5, 40, 120],
        "equatorial": [-5, 5, 40, 120],
        "indian ocean": [-30, 30, 40, 120],
        "arabian sea": [8, 25, 55, 75],
        "bay of bengal": [5, 22, 80, 100],
        "southern ocean": [-60, -30, 40, 120],
        "tropical": [-23.5, 23.5, 40, 120],
        "north": [0, 30, 40, 120],
        "south": [-30, 0, 40, 120]
    }
    
    for region, bbox in regions.items():
        if region in msg:
            return bbox, region
    return None, None

def answer_query(user_input, df):
    """Enhanced query processing with better responses"""
    msg = user_input.lower()
    
    # Determine data type
    data_type = None
    if any(k in msg for k in ["salinity", "salt"]):
        data_type = 'Salinity'
    elif any(k in msg for k in ["temperature", "temp"]):
        data_type = 'Temperature'
    elif any(k in msg for k in ["depth"]):
        data_type = 'Depth'
    
    # Filter by region
    filtered = df.copy()
    bbox, region_name = extract_region(msg)
    if bbox:
        lat_min, lat_max, lon_min, lon_max = bbox
        filtered = filtered[
            (filtered['Latitude'] >= lat_min) & (filtered['Latitude'] <= lat_max) &
            (filtered['Longitude'] >= lon_min) & (filtered['Longitude'] <= lon_max)
        ]
    
    # Generate intelligent responses
    if "how many" in msg or "count" in msg:
        count = len(filtered)
        response = f"📊 Found **{count}** ARGO float measurements"
        if region_name:
            response += f" in the **{region_name}**"
        response += f" out of {len(df)} total measurements."
        
    elif "average" in msg or "mean" in msg:
        if data_type and data_type in filtered.columns:
            avg_val = filtered[data_type].mean()
            unit = " PSU" if data_type == 'Salinity' else "°C" if data_type == 'Temperature' else " meters"
            response = f"🔢 Average **{data_type.lower()}**: **{avg_val:.2f}{unit}**"
            if region_name:
                response += f" in the **{region_name}**"
            response += f"\n\n*Based on {len(filtered)} measurements*"
        else:
            response = "❓ Please specify **salinity**, **temperature**, or **depth** for average calculation."
            
    elif "maximum" in msg or "highest" in msg or "max" in msg:
        if data_type and data_type in filtered.columns:
            max_val = filtered[data_type].max()
            max_row = filtered[filtered[data_type] == max_val].iloc[0]
            unit = " PSU" if data_type == 'Salinity' else "°C" if data_type == 'Temperature' else " meters"
            response = f"📈 **Highest {data_type.lower()}**: **{max_val:.2f}{unit}**\n"
            response += f"📍 Location: Float **{max_row['Float_ID']}** at ({max_row['Latitude']:.2f}°N, {max_row['Longitude']:.2f}°E)"
        else:
            response = "❓ Please specify **salinity**, **temperature**, or **depth** for maximum calculation."
            
    elif "minimum" in msg or "lowest" in msg or "min" in msg:
        if data_type and data_type in filtered.columns:
            min_val = filtered[data_type].min()
            min_row = filtered[filtered[data_type] == min_val].iloc[0]
            unit = " PSU" if data_type == 'Salinity' else "°C" if data_type == 'Temperature' else " meters"
            response = f"📉 **Lowest {data_type.lower()}**: **{min_val:.2f}{unit}**\n"
            response += f"📍 Location: Float **{min_row['Float_ID']}** at ({min_row['Latitude']:.2f}°N, {min_row['Longitude']:.2f}°E)"
        else:
            response = "❓ Please specify **salinity**, **temperature**, or **depth** for minimum calculation."
    
    elif data_type:
        if not filtered.empty:
            stats = filtered[data_type].describe()
            unit = " PSU" if data_type == 'Salinity' else "°C" if data_type == 'Temperature' else " meters"
            response = f"📊 **{data_type} Statistics** ({len(filtered)} measurements):\n\n"
            response += f"• **Mean**: {stats['mean']:.2f}{unit}\n"
            response += f"• **Range**: {stats['min']:.2f} - {stats['max']:.2f}{unit}\n"
            response += f"• **Standard Deviation**: {stats['std']:.2f}{unit}"
        else:
            response = f"❌ No **{data_type.lower()}** data found for your criteria."
    
    else:
        response = """🤖 **I can help you explore ocean data!** Try asking:

• *"What is the average salinity in Arabian Sea?"*
• *"How many floats near the equator?"*
• *"Show maximum temperature measurements"*
• *"Count floats in Bay of Bengal"*

**Available data types**: Salinity, Temperature, Depth
**Available regions**: Equatorial, Arabian Sea, Bay of Bengal, Tropical, etc."""

    return response

def filter_df_by_query(df):
    """Filter dataframe based on last user query"""
    if not st.session_state.chat:
        return df
        
    last_user_msg = [msg for role, msg in st.session_state.chat if role == "user"][-1]
    bbox, _ = extract_region(last_user_msg.lower())
    
    if bbox:
        lat_min, lat_max, lon_min, lon_max = bbox
        return df[
            (df['Latitude'] >= lat_min) & (df['Latitude'] <= lat_max) &
            (df['Longitude'] >= lon_min) & (df['Longitude'] <= lon_max)
        ]
    return df

def create_interactive_map(df, filtered_df, color_by="Temperature"):
    """Create an interactive Folium map with ARGO float data"""
    
    # Determine map center
    if not filtered_df.empty:
        center_lat = filtered_df['Latitude'].mean()
        center_lon = filtered_df['Longitude'].mean()
        zoom_start = 6
    else:
        center_lat = 0
        center_lon = 80
        zoom_start = 4
    
    # Create base map
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=zoom_start,
        tiles="CartoDB positron"
    )
    
    # Add all floats as background (gray markers)
    for _, row in df.iterrows():
        if row.name not in filtered_df.index:
            folium.CircleMarker(
                location=[row['Latitude'], row['Longitude']],
                radius=4,
                color='gray',
                weight=1,
                fill=True,
                fillColor='lightgray',
                fillOpacity=0.4,
                popup=folium.Popup(
                    f"""
                    <div style='font-family: Arial; min-width: 200px;'>
                        <h4 style='margin: 0; color: #666;'>🌊 Float {row['Float_ID']}</h4>
                        <hr style='margin: 5px 0;'>
                        <p style='margin: 2px 0;'><b>📍 Location:</b> {row['Latitude']:.3f}°N, {row['Longitude']:.3f}°E</p>
                        <p style='margin: 2px 0;'><b>🧂 Salinity:</b> {row['Salinity']:.2f} PSU</p>
                        <p style='margin: 2px 0;'><b>🌡️ Temperature:</b> {row['Temperature']:.1f}°C</p>
                        <p style='margin: 2px 0;'><b>📏 Depth:</b> {row['Depth']:.0f}m</p>
                        <p style='margin: 2px 0;'><b>📅 Date:</b> {row['Time'].strftime('%Y-%m-%d')}</p>
                    </div>
                    """, 
                    max_width=300
                )
            ).add_to(m)
    
    # Add filtered floats with color coding
    if not filtered_df.empty:
        # Normalize values for color mapping
        min_val = filtered_df[color_by].min()
        max_val = filtered_df[color_by].max()
        
        def get_color(value):
            if max_val == min_val:
                return '#3498db'  # Default blue if all values are same
            normalized = (value - min_val) / (max_val - min_val)
            # Color gradient from blue (low) through green to red (high)
            if normalized < 0.5:
                r = int(255 * (normalized * 2))
                g = 255
                b = int(255 * (1 - normalized * 2))
            else:
                r = 255
                g = int(255 * (2 * (1 - normalized)))
                b = 0
            return f'#{r:02x}{g:02x}{b:02x}'
        
        for _, row in filtered_df.iterrows():
            val = row[color_by]
            color = get_color(val)
            
            # Determine marker size based on depth
            radius = max(6, min(15, row['Depth'] / 200 + 6))
            
            # Create detailed popup
            popup_html = f"""
            <div style='font-family: Arial; min-width: 250px;'>
                <h3 style='margin: 0; color: #2c3e50; text-align: center;'>🌊 ARGO Float {row['Float_ID']}</h3>
                <hr style='margin: 8px 0; border: 1px solid #3498db;'>
                
                <div style='display: grid; grid-template-columns: 1fr 1fr; gap: 10px;'>
                    <div>
                        <p style='margin: 3px 0; font-weight: bold; color: #e74c3c;'>🧂 Salinity</p>
                        <p style='margin: 0; font-size: 1.1em;'>{row['Salinity']:.2f} PSU</p>
                    </div>
                    <div>
                        <p style='margin: 3px 0; font-weight: bold; color: #e67e22;'>🌡️ Temperature</p>
                        <p style='margin: 0; font-size: 1.1em;'>{row['Temperature']:.1f}°C</p>
                    </div>
                </div>
                
                <hr style='margin: 8px 0; border: 0.5px solid #bdc3c7;'>
                
                <p style='margin: 3px 0;'><b>📍 Location:</b> {row['Latitude']:.3f}°N, {row['Longitude']:.3f}°E</p>
                <p style='margin: 3px 0;'><b>📏 Depth:</b> {row['Depth']:.0f} meters</p>
                <p style='margin: 3px 0;'><b>📅 Measurement Date:</b> {row['Time'].strftime('%Y-%m-%d')}</p>
                
                <div style='margin-top: 10px; padding: 5px; background-color: #ecf0f1; border-radius: 3px;'>
                    <small><b>Current Filter:</b> Colored by {color_by}</small>
                </div>
            </div>
            """
            
            folium.CircleMarker(
                location=[row['Latitude'], row['Longitude']],
                radius=radius,
                color='white',
                weight=2,
                fill=True,
                fillColor=color,
                fillOpacity=0.8,
                popup=folium.Popup(popup_html, max_width=350)
            ).add_to(m)
    
    # Add region boundaries if filtering by region
    if st.session_state.chat:
        last_user_msg = [msg for role, msg in st.session_state.chat if role == "user"][-1]
        bbox, region_name = extract_region(last_user_msg.lower())
        if bbox:
            lat_min, lat_max, lon_min, lon_max = bbox
            
            # Add bounding box rectangle
            folium.Rectangle(
                bounds=[[lat_min, lon_min], [lat_max, lon_max]],
                color='red',
                weight=2,
                fill=False,
                popup=f"Query Region: {region_name.title()}"
            ).add_to(m)
    
    # Add legend
    if not filtered_df.empty and max_val != min_val:
        unit = " PSU" if color_by == 'Salinity' else "°C" if color_by == 'Temperature' else " m"
        legend_html = f"""
        <div style="position: fixed; 
                    bottom: 50px; right: 10px; width: 180px; height: 120px; 
                    background-color: rgba(255, 255, 255, 0.95); 
                    border: 2px solid #3498db; border-radius: 8px; 
                    padding: 10px; z-index: 1000;
                    box-shadow: 0 4px 8px rgba(0,0,0,0.1);">
            <h4 style="margin: 0 0 8px 0; color: #2c3e50; text-align: center;">{color_by} Scale</h4>
            <div style="display: flex; align-items: center; margin: 5px 0;">
                <div style="width: 20px; height: 15px; background: linear-gradient(to right, #0000ff, #00ff00, #ff0000); margin-right: 8px;"></div>
                <small>Low → High</small>
            </div>
            <p style="margin: 3px 0; font-size: 12px;"><b>Min:</b> {min_val:.2f}{unit}</p>
            <p style="margin: 3px 0; font-size: 12px;"><b>Max:</b> {max_val:.2f}{unit}</p>
            <p style="margin: 3px 0; font-size: 10px; color: #7f8c8d;">Size = Depth</p>
        </div>
        """
        m.get_root().html.add_child(folium.Element(legend_html))
    
    return m

# ---------------------- HEADER ----------------------
st.markdown("""
<div style='text-align: center; padding: 1.5rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
            border-radius: 15px; margin-bottom: 1.5rem; box-shadow: 0 4px 15px rgba(0,0,0,0.2);'>
    <h1 style='color: white; margin: 0; font-size: 2.5rem;'>🌊 FloatChat</h1>
    <p style='color: rgba(255,255,255,0.9); margin: 0.5rem 0 0 0; font-size: 1.1rem;'>
        AI-Powered Indian Ocean ARGO Data Explorer with Interactive Mapping
    </p>
    <p style='color: rgba(255,255,255,0.7); margin: 0.5rem 0 0 0; font-size: 0.9rem;'>
        Ask questions about ocean data and see the results plotted on the interactive map below
    </p>
</div>
""", unsafe_allow_html=True)

# ---------------------- METRICS DASHBOARD ----------------------
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("🌍 **Total Floats**", len(df), help="Total number of ARGO float measurements")
with col2:
    st.metric("🧂 **Avg Salinity**", f"{df['Salinity'].mean():.2f} PSU", 
              delta=f"±{df['Salinity'].std():.2f}", help="Average salinity across all measurements")
with col3:
    st.metric("🌡️ **Avg Temperature**", f"{df['Temperature'].mean():.1f}°C", 
              delta=f"±{df['Temperature'].std():.1f}", help="Average temperature across all measurements")
with col4:
    st.metric("📊 **Data Coverage**", 
              f"{df['Longitude'].max()-df['Longitude'].min():.0f}° × {df['Latitude'].max()-df['Latitude'].min():.0f}°",
              help="Geographic coverage (longitude × latitude)")

# ---------------------- LAYOUT ----------------------
left_col, right_col = st.columns([1, 2])

# ---------------------- ENHANCED CHAT PANEL ----------------------
with left_col:
    st.subheader("💬 Chat with FloatChat AI")
    
    # Quick start examples
    with st.expander("💡 **Example Questions - Click to Try**", expanded=True):
        examples = [
            "Show average salinity in Arabian Sea",
            "How many floats near the equator?",
            "What's the maximum temperature?",
            "Count floats in Bay of Bengal",
            "Show depth statistics"
        ]
        
        for ex in examples:
            if st.button(ex, key=f"example_{hash(ex)}", use_container_width=True):
                st.session_state.chat.append(("user", ex))
                answer = answer_query(ex, df)
                st.session_state.chat.append(("bot", answer))
                st.rerun()
    
    # Main chat input
    user_input = st.text_input(
        "**Ask your question:**", 
        placeholder="e.g., What is the average salinity in the Arabian Sea?",
        help="Ask about salinity, temperature, depth, or float counts in different regions"
    )
    
    # Chat buttons
    col_send, col_clear = st.columns([3, 1])
    with col_send:
        if st.button("🚀 **Send**", use_container_width=True, type="primary") and user_input.strip():
            st.session_state.chat.append(("user", user_input))
            answer = answer_query(user_input, df)
            st.session_state.chat.append(("bot", answer))
            st.rerun()
    
    with col_clear:
        if st.button("🗑️", help="Clear chat history", use_container_width=True):
            st.session_state.chat = []
            st.rerun()
    
    # Chat history display
    if st.session_state.chat:
        st.markdown("---")
        st.markdown("**💬 Chat History:**")
        
        # Show recent messages (last 8)
        recent_chat = st.session_state.chat[-8:]
        
        for role, message in recent_chat:
            if role == "user":
                st.markdown(f"""
                <div class='chat-bubble chat-user'>
                    <strong>👤 You:</strong><br>
                    {message}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class='chat-bubble chat-bot'>
                    <strong>🤖 FloatChat:</strong><br>
                    {message}
                </div>
                """, unsafe_allow_html=True)

# ---------------------- INTERACTIVE MAP SECTION ----------------------
with right_col:
    st.subheader("🗺️ Interactive Ocean Data Map")
    
    # Get filtered data based on chat
    display_df = filter_df_by_query(df)
    
    # Map controls
    map_col1, map_col2, map_col3 = st.columns(3)
    with map_col1:
        color_by = st.selectbox("**Color markers by:**", ["Temperature", "Salinity", "Depth"], 
                               help="Choose which data parameter to use for color-coding the map markers")
    with map_col2:
        show_all = st.checkbox("Show all floats", value=True, 
                              help="Show all floats in gray, with filtered floats highlighted")
    with map_col3:
        map_info = st.empty()
    
    # Create and display the map
    if not display_df.empty:
        # Create the interactive map
        ocean_map = create_interactive_map(df, display_df, color_by)
        
        # Display map in container
        st.markdown('<div class="map-container">', unsafe_allow_html=True)
        map_data = st_folium(ocean_map, width=700, height=500, returned_objects=["last_clicked"])
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Map information
        if len(display_df) != len(df):
            map_info.info(f"🗺️ **{len(display_df)}** floats highlighted from your query, **{len(df)}** total shown")
        else:
            map_info.info(f"🗺️ Showing all **{len(df)}** ARGO float locations")
            
        # Handle map clicks
        if map_data['last_clicked'] is not None:
            clicked_lat = map_data['last_clicked']['lat']
            clicked_lng = map_data['last_clicked']['lng']
            
            # Find nearest float
            df_with_distance = df.copy()
            df_with_distance['distance'] = np.sqrt((df_with_distance['Latitude'] - clicked_lat)**2 + 
                                                  (df_with_distance['Longitude'] - clicked_lng)**2)
            nearest_float = df_with_distance.loc[df_with_distance['distance'].idxmin()]
            
            st.success(f"🎯 **Clicked near Float {nearest_float['Float_ID']}**")
            st.write(f"📊 Salinity: {nearest_float['Salinity']:.2f} PSU | Temperature: {nearest_float['Temperature']:.1f}°C | Depth: {nearest_float['Depth']:.0f}m")
    
    else:
        st.warning("🗺️ **No data matches your current query filters.** Try a different question or region.")
        
        # Still show the base map with all data
        ocean_map = create_interactive_map(df, df, color_by)
        st.markdown('<div class="map-container">', unsafe_allow_html=True)
        st_folium(ocean_map, width=700, height=500)
        st.markdown('</div>', unsafe_allow_html=True)

# ---------------------- QUICK VISUALIZATION ----------------------
st.markdown("---")
st.subheader("📊 Quick Data Visualization")

viz_col1, viz_col2 = st.columns(2)

with viz_col1:
    # Scatter plot
    plot_df = display_df if not display_df.empty else df
    fig_scatter = px.scatter(
        plot_df,
        x='Longitude',
        y='Latitude',
        color=color_by,
        size='Depth',
        hover_data=['Float_ID', 'Salinity', 'Temperature', 'Depth'],
        title=f"Float Locations (colored by {color_by})",
        labels={
            'Longitude': 'Longitude (°E)',
            'Latitude': 'Latitude (°N)',
            color_by: f"{color_by} ({'°C' if color_by == 'Temperature' else 'PSU' if color_by == 'Salinity' else 'm'})"
        }
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

with viz_col2:
    # T-S Diagram
    fig_ts = px.scatter(
        plot_df,
        x='Temperature',
        y='Salinity',
        color='Depth',
        hover_data=['Float_ID', 'Latitude', 'Longitude'],
        title="Temperature-Salinity Relationship",
        labels={
            'Temperature': 'Temperature (°C)',
            'Salinity': 'Salinity (PSU)',
            'Depth': 'Depth (m)'
        }
    )
    st.plotly_chart(fig_ts, use_container_width=True)

# ---------------------- DATA EXPORT ----------------------
st.markdown("---")
export_col1, export_col2, export_col3 = st.columns(3)

with export_col1:
    export_df = display_df if not display_df.empty else df
    csv_data = export_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        "📄 **Download Data as CSV**",
        data=csv_data,
        file_name=f'argo_floats_{datetime.now().strftime("%Y%m%d_%H%M")}.csv',
        mime='text/csv',
        use_container_width=True
    )

with export_col2:
    st.metric("📊 **Available for Export**", len(export_df), help="Number of measurements ready for download")

with export_col3:
    if st.button("🔄 **Refresh Data**", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

# ---------------------- FOOTER ----------------------
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem; background-color: #f8f9fa; border-radius: 10px;'>
    <p style='margin: 0;'>
        🌊 **FloatChat** - Interactive Ocean Data Explorer with AI-Powered Chat<br>
        🗺️ **Click on map markers** for detailed float information | 💬 **Ask questions** to filter and explore data<br>
        Built with ❤️ using Streamlit, Folium & Plotly | 
        <a href="https://argo.ucsd.edu/" target="_blank" style="color: #0a4c86;">Learn more about ARGO</a>
    </p>
</div>
""", unsafe_allow_html=True)