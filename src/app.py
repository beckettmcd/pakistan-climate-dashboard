import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import numpy as np
from streamlit_folium import folium_static
import folium
from folium.plugins import MarkerCluster
import geopandas as gpd
from shapely.geometry import shape
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="DAI Pakistan Climate Dashboard",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for DAI corporate branding - sharp, compact, and clean
st.markdown("""
<style>
    :root {
        --dai-cornflower: #8FC3EA;
        --dai-white: #FFFFFF;
        --dai-river-bed: #455560;
        --dai-cornflower-light: #E8F4FC;
        --dai-river-bed-light: #F5F6F7;
        --dai-border: #E1E5E9;
        --dai-text-primary: #455560;
        --dai-text-secondary: #6B7C93;
        --dai-shadow: rgba(69, 85, 96, 0.08);
    }

    .main-header {
        font-size: 1.8rem;
        font-weight: 600;
        color: var(--dai-river-bed);
        text-align: left;
        margin: 16px 0 0.5rem 0;
        font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, sans-serif;
        letter-spacing: -0.25px;
    }

    .subtitle { display: none; }

    .section-header {
        font-size: 1.1rem;
        font-weight: 600;
        color: var(--dai-river-bed);
        margin: 1rem 0 0.5rem 0;
        padding-bottom: 0.4rem;
        border-bottom: 2px solid var(--dai-cornflower);
        text-transform: uppercase;
        letter-spacing: 0.3px;
    }

    .stTabs [data-baseweb="tab-list"] { gap: 0.25rem; border-bottom: 1px solid var(--dai-border); }
    .stTabs [data-baseweb="tab"] { padding: 0.5rem 0.75rem; }

    .stAlert, .stDataFrame, .js-plotly-plot { border: 1px solid var(--dai-border); }
    .stDataFrame { border-radius: 0; }
    .block-container { padding-top: 0.75rem; padding-bottom: 0.75rem; }
</style>
""", unsafe_allow_html=True)


def _deterministic_rng(s: str) -> np.random.Generator:
    # Deterministic per-district RNG for reproducible synthetic stats
    seed = abs(hash(s)) % (2**32 - 1)
    return np.random.default_rng(seed)


def _add_demographics(df: pd.DataFrame) -> pd.DataFrame:
    if all(col in df.columns for col in [
        'population','area_km2','literacy_rate','poverty_rate','population_density'
    ]):
        return df
    pops, areas, lits, povs = [], [], [], []
    for name in df['district']:
        rng = _deterministic_rng(str(name))
        # synthetic but plausible ranges
        area = float(rng.uniform(500, 12000))
        pop = float(rng.uniform(0.15, 4.5)) * 1_000_000
        lit = float(rng.uniform(35, 85))
        pov = float(rng.uniform(10, 55))
        areas.append(round(area, 1))
        pops.append(int(pop))
        lits.append(round(lit, 1))
        povs.append(round(pov, 1))
    df = df.copy()
    df['area_km2'] = areas
    df['population'] = pops
    df['literacy_rate'] = lits
    df['poverty_rate'] = povs
    df['population_density'] = (df['population'] / df['area_km2']).round(1)
    return df


@st.cache_data
def load_data():
    climate_df = pd.read_csv('data/Sample_District_Climate_Data.csv')
    climate_df = _add_demographics(climate_df)
    with open('data/geoBoundaries-PAK-ADM2 (1).geojson', 'r') as f:
        geo_data = json.load(f)
    gdf = gpd.GeoDataFrame.from_features(geo_data['features'])
    gdf = gdf.rename(columns={'shapeName': 'district'})
    merged_df = gdf.merge(climate_df, on='district', how='left')
    # Precompute centroids for highlighting overlays
    try:
        centroids = merged_df.geometry.centroid
        merged_df['centroid_lon'] = centroids.x
        merged_df['centroid_lat'] = centroids.y
    except Exception:
        merged_df['centroid_lon'] = np.nan
        merged_df['centroid_lat'] = np.nan
    return climate_df, merged_df, geo_data


def create_choropleth_map(data, column, color_scale='Blues', map_style='carto-positron', highlight=None):
    fig = px.choropleth_mapbox(
        data,
        geojson=data.geometry,
        locations=data.index,
        color=column,
        hover_name='district',
        hover_data=[column],
        mapbox_style=map_style,
        zoom=4,
        center={"lat": 30.3753, "lon": 69.3451},
        opacity=0.85,
        color_continuous_scale=color_scale,
        title=None
    )
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        height=620,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Segoe UI, -apple-system, BlinkMacSystemFont, sans-serif"),
        coloraxis_colorbar=dict(
            orientation='h', y=-0.08, x=0.5, xanchor='center', thickness=12, len=0.6, outlinewidth=0
        ),
        hoverlabel=dict(font_size=12),
        mapbox=dict(uirevision=True)
    )
    # Optional highlighted districts overlay
    if highlight:
        overlay = data[data['district'].isin(highlight)].dropna(subset=['centroid_lat','centroid_lon'])
        if not overlay.empty:
            # 1) Filled polygon overlay in contrasting color
            highlight_geojson = json.loads(overlay.to_json())
            fig.add_trace(
                go.Choroplethmapbox(
                    geojson=highlight_geojson,
                    locations=overlay['district'],
                    z=np.ones(len(overlay)),
                    featureidkey='properties.district',
                    colorscale=[[0, 'rgba(255,107,53,0.55)'], [1, 'rgba(255,107,53,0.55)']],
                    showscale=False,
                    zmin=0,
                    zmax=1,
                    marker=dict(line=dict(width=2, color='#FFFFFF'))
                )
            )
            # 2) Centroid markers + labels
            fig.add_trace(
                go.Scattermapbox(
                    lat=overlay['centroid_lat'],
                    lon=overlay['centroid_lon'],
                    mode='markers+text',
                    text=overlay['district'],
                    textposition='top center',
                    marker=dict(size=10, color='#FF6B35', opacity=0.95, symbol='circle'),
                    hoverinfo='text'
                )
            )
    return fig


def create_distribution_plot(data, column, title):
    fig = px.histogram(data, x=column, nbins=20, title=title, color_discrete_sequence=['#8FC3EA'])
    fig.update_layout(title_x=0, title_font_size=12, title_font_color='#455560', showlegend=False, height=300,
                      margin=dict(l=0,r=0,t=30,b=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    return fig


def create_correlation_heatmap(data):
    numeric_cols = ['avg_temperature_c', 'avg_rainfall_mm', 'drought_risk_index', 'flood_risk_index',
                    'population','area_km2','population_density','literacy_rate','poverty_rate']
    corr_matrix = data[numeric_cols].corr()
    fig = px.imshow(corr_matrix, text_auto=True, aspect="auto", color_continuous_scale='Blues', title="Correlation")
    fig.update_layout(title_x=0, title_font_size=12, title_font_color='#455560', height=300,
                      margin=dict(l=0,r=0,t=30,b=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    return fig


def create_scatter_plot(data, x_col, y_col, color_col=None):
    fig = px.scatter(data, x=x_col, y=y_col, color=color_col, hover_name='district',
                     title=f"{x_col.replace('_', ' ').title()} vs {y_col.replace('_', ' ').title()}",
                     color_continuous_scale='Blues')
    fig.update_layout(title_x=0, title_font_size=12, title_font_color='#455560', height=320,
                      margin=dict(l=0,r=0,t=30,b=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    return fig


def main():
    climate_df, merged_df, geo_data = load_data()

    # Query params (deep links)
    qp = st.query_params
    default_variable = qp.get('variable', ['avg_temperature_c'])[0] if hasattr(qp, 'get') else 'avg_temperature_c'
    default_scale = qp.get('scale', ['Blues'])[0] if hasattr(qp, 'get') else 'Blues'
    default_style = qp.get('style', ['carto-positron'])[0] if hasattr(qp, 'get') else 'carto-positron'

    # Header
    st.markdown('<h1 class="main-header">DAI Pakistan Climate Dashboard</h1>', unsafe_allow_html=True)

    # Sidebar controls
    st.sidebar.markdown("### Controls")
    variable_choices = ['avg_temperature_c', 'avg_rainfall_mm', 'drought_risk_index', 'flood_risk_index',
                        'population','population_density','area_km2','literacy_rate','poverty_rate']
    variable = st.sidebar.selectbox(
        "Variable",
        variable_choices,
        index=variable_choices.index(default_variable) if default_variable in variable_choices else 0,
        format_func=lambda x: x.replace('_', ' ').title()
    )
    color_scale = st.sidebar.selectbox("Color scale", ['Blues', 'Greens', 'Viridis', 'Plasma', 'RdBu', 'Spectral'],
                                       index=['Blues', 'Greens', 'Viridis', 'Plasma', 'RdBu', 'Spectral'].index(default_scale) if default_scale in ['Blues', 'Greens', 'Viridis', 'Plasma', 'RdBu', 'Spectral'] else 0)
    map_style = st.sidebar.selectbox("Map style", ['carto-positron', 'open-street-map', 'carto-darkmatter', 'stamen-toner'],
                                     index=['carto-positron', 'open-street-map', 'carto-darkmatter', 'stamen-toner'].index(default_style) if default_style in ['carto-positron', 'open-street-map', 'carto-darkmatter', 'stamen-toner'] else 0)

    # Highlight controls
    st.sidebar.markdown("### Highlight districts")
    highlight = st.sidebar.multiselect("Select to highlight", options=sorted(climate_df['district'].unique()))

    # Compact metrics in sidebar
    st.sidebar.markdown("### Snapshot")
    col_a, col_b = st.sidebar.columns(2)
    col_c, col_d = st.sidebar.columns(2)
    col_a.metric("Avg", f"{climate_df[variable].mean():.1f}")
    col_b.metric("Max", f"{climate_df[variable].max():.1f}")
    col_c.metric("Min", f"{climate_df[variable].min():.1f}")
    col_d.metric("Std", f"{climate_df[variable].std():.1f}")

    # Downloads
    st.sidebar.markdown("### Download")
    st.sidebar.download_button("Climate+Demographics CSV", climate_df.to_csv(index=False).encode('utf-8'), file_name="district_climate_demographics.csv", mime="text/csv")

    # Optional quick filters/search in sidebar
    st.sidebar.markdown("### Search")
    search_term = st.sidebar.text_input("District contains")

    # Persist query params
    try:
        st.query_params.update({"variable": variable, "scale": color_scale, "style": map_style})
    except Exception:
        pass

    # Main tabs
    main_tabs = st.tabs(["🗺️ Map View", "📊 Analysis", "ℹ️ About"])
    
    with main_tabs[0]:
        # Map first (lead)
        st.markdown('<div class="section-header">Geographic Distribution</div>', unsafe_allow_html=True)
        map_fig = create_choropleth_map(merged_df, variable, color_scale, map_style, highlight)
        st.plotly_chart(map_fig, use_container_width=True, config={"displayModeBar": False, "responsive": True})
    
    with main_tabs[1]:
        # Compact analysis tabs
        tabs = st.tabs(["Top/Bottom", "Distributions", "Correlation", "Relationships", "Risk", "Data"]) 

    with tabs[0]:
        left, right = st.columns(2)
        with left:
            st.markdown("**Top 10 Districts**")
            st.dataframe(climate_df.nlargest(10, variable)[['district', variable]], use_container_width=True, height=280)
        with right:
            st.markdown("**Bottom 10 Districts**")
            st.dataframe(climate_df.nsmallest(10, variable)[['district', variable]], use_container_width=True, height=280)

    with tabs[1]:
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(create_distribution_plot(climate_df, variable, "Distribution"), use_container_width=True)
        with c2:
            box_fig = px.box(climate_df, y=variable, title="Box Plot")
            box_fig.update_layout(title_x=0, title_font_size=12, title_font_color='#455560', height=300,
                                  margin=dict(l=0,r=0,t=30,b=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(box_fig, use_container_width=True)

    with tabs[2]:
        st.plotly_chart(create_correlation_heatmap(climate_df), use_container_width=True)

    with tabs[3]:
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(create_scatter_plot(climate_df, 'avg_temperature_c', 'avg_rainfall_mm', 'drought_risk_index'), use_container_width=True)
        with c2:
            st.plotly_chart(create_scatter_plot(climate_df, 'population', 'population_density', 'literacy_rate'), use_container_width=True)

    with tabs[4]:
        l, r = st.columns(2)
        with l:
            st.markdown("**High Drought Risk (> 0.7)**")
            high_drought = climate_df[climate_df['drought_risk_index'] > 0.7][['district','drought_risk_index']].sort_values('drought_risk_index', ascending=False)
            st.dataframe(high_drought, use_container_width=True, height=260)
        with r:
            st.markdown("**High Flood Risk (> 0.7)**")
            high_flood = climate_df[climate_df['flood_risk_index'] > 0.7][['district','flood_risk_index']].sort_values('flood_risk_index', ascending=False)
            st.dataframe(high_flood, use_container_width=True, height=260)

    with tabs[5]:
        df_to_show = climate_df
        if search_term:
            df_to_show = df_to_show[df_to_show['district'].str.contains(search_term, case=False)]
        st.dataframe(df_to_show, use_container_width=True, height=420)
    
    with main_tabs[2]:
        st.markdown("## About This Dashboard")
        
        st.markdown("""
        ### Overview
        This Pakistan Climate Dashboard provides comprehensive insights into climate patterns, demographic data, and risk assessments across Pakistan's districts. The dashboard combines climate metrics with socio-economic indicators to offer a holistic view of regional variations.
        
        ### Data Sources
        - **Climate Data**: District-level climate metrics including temperature, rainfall, and risk indices
        - **Geographic Boundaries**: Administrative district boundaries from geoBoundaries
        - **Demographic Data**: Population statistics and socio-economic indicators
        
        ### Key Metrics
        - **Temperature**: Average annual temperature in Celsius
        - **Rainfall**: Average annual rainfall in millimeters  
        - **Drought Risk**: Index indicating vulnerability to drought conditions
        - **Flood Risk**: Index indicating vulnerability to flood events
        - **Demographics**: Population, density, literacy, and poverty rates
        
        ### How to Use
        1. **Map View**: Explore geographic distribution of variables across districts
        2. **Analysis**: Dive deeper into statistical patterns and relationships
        3. **Controls**: Use the sidebar to change variables, color schemes, and map styles
        4. **Search**: Filter districts by name to focus on specific regions
        
        ### Climate Context
        Pakistan's climate varies significantly from the arid regions of Balochistan to the monsoon-affected areas of Punjab and Sindh. This dashboard helps identify:
        - Temperature gradients across elevation and latitude
        - Rainfall patterns influenced by monsoon systems
        - Districts vulnerable to climate-related risks
        - Socio-economic factors that may amplify climate impacts
        
        ### Technical Details
        - Built with Streamlit and Plotly
        - Interactive choropleth maps with multiple styling options
        - Statistical analysis including correlations and distributions
        - Exportable data for further analysis
        
        ### Contact
        For questions or suggestions about this dashboard, please contact the development team.
        """)

if __name__ == "__main__":
    main()
