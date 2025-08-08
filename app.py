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
import os
warnings.filterwarnings('ignore')

# Optional Sentry import (will not crash if package is missing)
SENTRY_DSN = os.getenv("SENTRY_DSN", "")
try:
    if SENTRY_DSN:
        import sentry_sdk
        from sentry_sdk.integrations.logging import LoggingIntegration
        from sentry_sdk.integrations.stdlib import StdlibIntegration
        from sentry_sdk.integrations.excepthook import ExcepthookIntegration
        from sentry_sdk.integrations.modules import ModulesIntegration
        from sentry_sdk.integrations.aiohttp import AioHttpIntegration

        sentry_sdk.init(
            dsn=SENTRY_DSN,
            traces_sample_rate=float(os.getenv("SENTRY_TRACES_SAMPLE_RATE", "0.2")),
            profiles_sample_rate=float(os.getenv("SENTRY_PROFILES_SAMPLE_RATE", "0.0")),
            integrations=[
                LoggingIntegration(level=None, event_level=None),
                StdlibIntegration(),
                ExcepthookIntegration(always_run=True),
                ModulesIntegration(),
                AioHttpIntegration(),
            ],
            environment=os.getenv("SENTRY_ENVIRONMENT", "production"),
            release=os.getenv("SENTRY_RELEASE"),
        )
    else:
        sentry_sdk = None  # type: ignore
except Exception:
    sentry_sdk = None  # type: ignore

# Page configuration
st.set_page_config(
    page_title="DAI Pakistan Climate Dashboard",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS inspired by DAI (dai.com) — clean, understated, accessible
st.markdown("""
<style>
    :root {
        /* Modern color palette */
        --primary: #2563EB;                /* vibrant blue */
        --primary-light: #60A5FA;          /* light blue */
        --accent: #10B981;                 /* emerald */
        --text-primary: #1F2937;           /* near black */
        --text-secondary: #4B5563;         /* medium gray */
        --text-tertiary: #9CA3AF;          /* light gray */
        --border: #E5E7EB;                 /* subtle border */
        --bg-primary: #FFFFFF;             /* white */
        --bg-secondary: #F9FAFB;           /* off white */
        --shadow-sm: 0 1px 2px rgba(0, 0, 0, 0.05);
        --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
    }

    /* Modern typography */
    body {
        font-family: "Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        -webkit-font-smoothing: antialiased;
        -moz-osx-font-smoothing: grayscale;
    }

    .block-container { 
        padding-top: 1rem; 
        padding-bottom: 1rem;
        max-width: 1280px;
        margin: 0 auto;
    }

    /* Enhanced page header */
    .page-header { 
        background: var(--bg-primary);
        border: 1px solid var(--border);
        border-left: 4px solid var(--primary);
        border-radius: 8px;
        padding: 1.5rem;
        box-shadow: var(--shadow-md);
        margin-bottom: 1.5rem;
    }
    .page-header .brand { 
        color: var(--primary);
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-size: 0.875rem;
        font-weight: 600;
    }
    .page-header h1 { 
        color: var(--text-primary);
        margin: 0.5rem 0;
        font-size: 2rem;
        font-weight: 800;
        letter-spacing: -0.025em;
        line-height: 1.2;
    }
    .page-header .lede { 
        margin: 0;
        color: var(--text-secondary);
        font-size: 1.125rem;
        line-height: 1.5;
    }

    /* Section header */
    .section-header {
        font-size: 1.25rem;
        font-weight: 700;
        color: var(--text-primary);
        margin: 1.5rem 0 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid var(--primary);
        letter-spacing: -0.025em;
    }

    /* Enhanced tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
        border-bottom: 1px solid var(--border);
        padding-bottom: 0.25rem;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 0.75rem 1rem;
        color: var(--text-secondary);
        font-weight: 500;
        transition: all 0.2s ease;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: var(--primary);
        background: var(--bg-secondary);
        border-radius: 6px;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        color: var(--primary);
        border-bottom: 2px solid var(--primary);
        font-weight: 600;
    }

    /* Modern cards and data displays */
    .stAlert, .stDataFrame, .js-plotly-plot {
        border: 1px solid var(--border);
        border-radius: 8px;
        box-shadow: var(--shadow-sm);
        transition: box-shadow 0.2s ease;
    }
    .stAlert:hover, .stDataFrame:hover, .js-plotly-plot:hover {
        box-shadow: var(--shadow-md);
    }
    
    /* Sidebar refinements */
    .css-1d391kg {
        background: var(--bg-secondary);
        border-right: 1px solid var(--border);
    }
    
    /* Input fields */
    .stTextInput > div > div {
        border-radius: 6px;
        border-color: var(--border);
    }
    .stTextInput > div > div:focus-within {
        border-color: var(--primary);
        box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.1);
    }

    /* Enhanced buttons */
    .stButton > button {
        background: var(--primary);
        color: white;
        border: 0;
        padding: 0.5rem 1rem;
        border-radius: 6px;
        font-weight: 500;
        transition: all 0.2s ease;
    }
    .stButton > button:hover {
        background: var(--primary-light);
        transform: translateY(-1px);
        box-shadow: var(--shadow-md);
    }
    
    /* Metrics and KPIs */
    .css-1xarl3l {
        background: var(--bg-primary);
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 1rem;
        box-shadow: var(--shadow-sm);
    }
    
    /* Select boxes */
    .stSelectbox > div > div {
        border-radius: 6px;
        border-color: var(--border);
    }
    .stSelectbox > div > div:hover {
        border-color: var(--primary-light);
    }
    
    /* Slider */
    .stSlider > div > div > div > div {
        background: var(--primary);
    }
    
    /* Data tables */
    .stDataFrame {
        font-family: "Inter", sans-serif;
    }
    .stDataFrame th {
        background: var(--bg-secondary);
        font-weight: 600;
        color: var(--text-primary);
    }
    .stDataFrame td {
        color: var(--text-secondary);
    }

    /* Title override (we render our own header) */
    .main-header { display: none; }
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


def _scale_0_1(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors='coerce').astype(float)
    min_val = np.nanmin(s.values) if len(s) else np.nan
    max_val = np.nanmax(s.values) if len(s) else np.nan
    if not np.isfinite(min_val) or not np.isfinite(max_val) or (max_val - min_val) == 0:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - min_val) / (max_val - min_val)


def _standardize_district_name(name: str) -> str:
    if not isinstance(name, str):
        return ""
    return name.strip().lower().replace(" district", "").replace("-", " ")


@st.cache_data
def load_profiles() -> pd.DataFrame | None:
    """Load rich district profiles CSV if present and extract useful columns.

    The source has very long, sometimes duplicated column names. We use
    case-insensitive substring matching to pick the most useful columns.
    """
    path_options = [
        'district_profiles_parsed_from_dps1.csv',
        'data/district_profiles_parsed_from_dps1.csv',
    ]
    csv_path = next((p for p in path_options if os.path.exists(p)), None)
    if not csv_path:
        return None
    try:
        raw = pd.read_csv(csv_path, low_memory=False)
    except Exception:
        return None

    # Build a helper for fuzzy column picking
    cols_lower = {c.lower(): c for c in raw.columns}
    def pick(*patterns: str) -> str | None:
        for pat in patterns:
            pat_l = pat.lower()
            for c_l, orig in cols_lower.items():
                if pat_l in c_l:
                    return orig
        return None

    mapping: dict[str, str | None] = {
        'province': pick('province'),
        'district': pick('district'),
        'area_km2_profile': pick('area (sq.', 'area (sq. km)'),
        'population_profile': pick('population population', 'population,'),
        'density_profile': pick('population density per sq.'),
        'sex_ratio_profile': pick('sex r', 'sex ratio (males'),
        'urban_population_profile': pick('urban population'),
        'rural_population_profile': pick('rural population'),
        'mdpi': pick('multi dimensional poverty index'),
        'toilet_access_pct': pick('percentage of households with toilet'),
        'learning_score': pick('learning score'),
        'electricity_availability': pick('availability of electricity'),
        'water_availability': pick('availability of water'),
        'toilet_availability': pick('availability of toilet'),
        'temp_june_max_c_profile': pick('mean temperature in june (max)'),
        'temp_jan_min_c_profile': pick('mean temperature in january (min)'),
        'rainfall_aug_mm_profile': pick('rainfall in august (mm)'),
        'rainfall_nov_mm_profile': pick('rainfall in november (mm)'),
        # Health capacity signals (optional)
        'govt_health_institutions': pick('govt. health institutions','govt health institutions'),
        'bed_strength': pick('bed strength'),
        'private_health_institutions': pick('priv. health institutions','private health'),
        # Education signals
        'num_primary_schools': pick('number of primary schools'),
        'num_middle_schools': pick('number of middle schools'),
        'num_high_schools': pick('number of high schools'),
        'num_higher_secondary_schools': pick('number of higher secondary schools'),
    }

    selected_cols = {k: v for k, v in mapping.items() if v is not None}
    if 'district' not in selected_cols:
        return None
    df = raw[list(selected_cols.values())].rename(columns={v: k for k, v in selected_cols.items()})
    # Standardize district name
    df['district'] = df['district'].astype(str).map(_standardize_district_name)

    # Coerce numerics
    for col in df.columns:
        if col == 'district' or col == 'province':
            continue
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


def compute_vulnerability_index(
    df: pd.DataFrame,
    *,
    weight_hazard: float,
    weight_exposure: float,
    weight_sensitivity: float,
    weight_capacity: float,
    drought_shock: float = 1.0,
    flood_shock: float = 1.0,
) -> pd.DataFrame:
    """Compute a composite Climate Vulnerability Index (0-100).

    Components:
    - Hazard: drought and flood risk (after shock multipliers)
    - Exposure: population density
    - Sensitivity: poverty rate
    - Adaptive capacity (reduces vulnerability): literacy rate
    """
    out = df.copy()
    for col in ['drought_risk_index','flood_risk_index','population_density','poverty_rate','literacy_rate']:
        if col not in out.columns:
            out[col] = np.nan

    hazard_raw = (
        pd.to_numeric(out['drought_risk_index'], errors='coerce').astype(float) * float(drought_shock) * 0.5 +
        pd.to_numeric(out['flood_risk_index'], errors='coerce').astype(float) * float(flood_shock) * 0.5
    )
    hazard = _scale_0_1(hazard_raw)
    exposure = _scale_0_1(out['population_density'])
    sensitivity = _scale_0_1(out['poverty_rate'])
    capacity = _scale_0_1(out['literacy_rate'])

    # Normalize weights
    w = np.array([weight_hazard, weight_exposure, weight_sensitivity, weight_capacity], dtype=float)
    w = np.where(np.isfinite(w), w, 0)
    w = w / (w.sum() if w.sum() > 0 else 1)

    composite = (w[0] * hazard + w[1] * exposure + w[2] * sensitivity + w[3] * (1 - capacity))
    out['vulnerability_index'] = _scale_0_1(composite) * 100.0
    return out


@st.cache_data
def load_data():
    climate_df = pd.read_csv('data/Sample_District_Climate_Data.csv')
    climate_df = _add_demographics(climate_df)
    climate_df['district'] = climate_df['district'].astype(str).map(_standardize_district_name)
    with open('data/geoBoundaries-PAK-ADM2 (1).geojson', 'r') as f:
        geo_data = json.load(f)
    gdf = gpd.GeoDataFrame.from_features(geo_data['features'])
    gdf = gdf.rename(columns={'shapeName': 'district'})
    gdf['district'] = gdf['district'].astype(str).map(_standardize_district_name)
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
    # Enrich with district profiles if available
    profiles_df = load_profiles()
    if profiles_df is not None:
        try:
            climate_df = climate_df.merge(profiles_df, on='district', how='left')
            merged_df = merged_df.merge(
                profiles_df[[
                    'district',
                    *[c for c in ['mdpi','toilet_access_pct','learning_score','electricity_availability','water_availability','toilet_availability'] if c in profiles_df.columns]
                ]],
                on='district', how='left'
            )
        except Exception:
            pass

    # Query params (deep links)
    qp = st.query_params
    default_variable = qp.get('variable', ['avg_temperature_c'])[0] if hasattr(qp, 'get') else 'avg_temperature_c'
    default_scale = qp.get('scale', ['Blues'])[0] if hasattr(qp, 'get') else 'Blues'
    default_style = qp.get('style', ['carto-positron'])[0] if hasattr(qp, 'get') else 'carto-positron'

    # Minimal header
    st.markdown(
        """
        <div class="page-header">
          <div class="brand">DAI • Climate</div>
          <h1>Pakistan Climate Dashboard</h1>
          <p class="lede">Evidence to action across Pakistan’s districts.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Sidebar controls
    st.sidebar.markdown("### Controls")

    # Optional: user feedback sent to Sentry as "user_feedback" message
    if SENTRY_DSN and sentry_sdk:
        with st.sidebar.expander("Send quick feedback"):
            fb = st.text_area("Tell us what broke or what to improve", placeholder="Type here…")
            if st.button("Send feedback") and fb.strip():
                try:
                    sentry_sdk.capture_message(f"user_feedback: {fb.strip()}")
                    st.success("Thanks! Your feedback was sent.")
                except Exception:
                    st.warning("Feedback service unavailable right now.")
    variable_choices = ['avg_temperature_c', 'avg_rainfall_mm', 'drought_risk_index', 'flood_risk_index',
                        'population','population_density','area_km2','literacy_rate','poverty_rate']
    if 'mdpi' in climate_df.columns:
        variable_choices.append('mdpi')
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

    # Composite vulnerability index controls
    st.sidebar.markdown("### Vulnerability Model")
    w_hazard = st.sidebar.slider("Weight: Hazard", 0.0, 1.0, 0.35, 0.05)
    w_exposure = st.sidebar.slider("Weight: Exposure", 0.0, 1.0, 0.25, 0.05)
    w_sensitivity = st.sidebar.slider("Weight: Sensitivity", 0.0, 1.0, 0.25, 0.05)
    w_capacity = st.sidebar.slider("Weight: Capacity", 0.0, 1.0, 0.15, 0.05)
    shock_drought = st.sidebar.slider("Scenario shock: Drought", 0.5, 1.5, 1.0, 0.05)
    shock_flood = st.sidebar.slider("Scenario shock: Flood", 0.5, 1.5, 1.0, 0.05)
    climate_df = compute_vulnerability_index(
        climate_df,
        weight_hazard=w_hazard,
        weight_exposure=w_exposure,
        weight_sensitivity=w_sensitivity,
        weight_capacity=w_capacity,
        drought_shock=shock_drought,
        flood_shock=shock_flood,
    )
    merged_df = merged_df.merge(climate_df[['district','vulnerability_index']], on='district', how='left')

    # Main tabs
    main_tabs = st.tabs(["Map View", "Analysis", "Scenarios", "Portfolio", "District Profiles", "About"])
    
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
            st.markdown("**High Vulnerability (top 15)**")
            top_vuln = climate_df.nlargest(15, 'vulnerability_index')[['district','vulnerability_index']]
            st.dataframe(top_vuln, use_container_width=True, height=260)
            st.markdown("**High Drought Risk (> 0.7)**")
            high_drought = climate_df[climate_df['drought_risk_index'] > 0.7][['district','drought_risk_index']].sort_values('drought_risk_index', ascending=False)
            st.dataframe(high_drought, use_container_width=True, height=260)
        with r:
            st.markdown("**High Flood Risk (> 0.7)**")
            high_flood = climate_df[climate_df['flood_risk_index'] > 0.7][['district','flood_risk_index']].sort_values('flood_risk_index', ascending=False)
            st.dataframe(high_flood, use_container_width=True, height=260)
        if 'mdpi' in climate_df.columns:
            st.markdown("**High Poverty (MDPI top 15)**")
            mdpi_top = climate_df.dropna(subset=['mdpi']).nlargest(15, 'mdpi')[['district','mdpi']]
            st.dataframe(mdpi_top, use_container_width=True, height=260)

    with main_tabs[2]:
        st.markdown('<div class="section-header">Scenario Explorer</div>', unsafe_allow_html=True)
        c1, c2 = st.columns([2, 1])
        with c1:
            map_fig = create_choropleth_map(merged_df, 'vulnerability_index', 'RdBu', map_style, highlight)
            map_fig.update_coloraxes(cmin=0, cmax=100)
            st.plotly_chart(map_fig, use_container_width=True, config={"displayModeBar": False})
        with c2:
            st.caption("Weights drive a composite Climate Vulnerability Index (0–100). Adjust shocks and weights to test policy scenarios.")
            sel = st.selectbox("Focus district", options=sorted(climate_df['district'].unique()))
            row = climate_df[climate_df['district'] == sel].iloc[0]
            st.metric("Vulnerability Index", f"{row['vulnerability_index']:.1f}")
            # Simple priority recommendations
            recs = []
            if row['drought_risk_index'] > 0.7:
                recs.append("Drought resilience: water harvesting, drought-tolerant seed, drip irrigation")
            if row['flood_risk_index'] > 0.7:
                recs.append("Flood resilience: early warning, embankments, elevating assets")
            if row['poverty_rate'] > 40:
                recs.append("Social protection: cash-for-work on resilience assets, livelihood grants")
            if row['literacy_rate'] < 55:
                recs.append("Capacity: community climate information services, risk education")
            if not recs:
                recs.append("Maintain risk monitoring and update local adaptation plans")
            st.markdown("**Priority actions**")
            for r in recs:
                st.write("- ", r)

    # New: Policy Portfolio Optimizer
    with main_tabs[3]:
        st.markdown('<div class="section-header">Policy Portfolio Optimizer</div>', unsafe_allow_html=True)
        st.caption("Allocate a budget across interventions to minimize the Climate Vulnerability Index (CVI). Uses simple elasticities for transparent decisions.")

        # Inputs
        col1, col2, col3 = st.columns(3)
        with col1:
            target_scope = st.selectbox("Scope", ["National (all districts)", "Focus on selected districts"], index=0)
            selected_districts = []
            if target_scope == "Focus on selected districts":
                selected_districts = st.multiselect("Districts", options=sorted(climate_df['district'].unique()))
        with col2:
            total_budget = st.number_input("Total budget (USD m)", min_value=1.0, value=50.0, step=1.0)
        with col3:
            planning_horizon = st.slider("Horizon (years)", 1, 5, 3)

        # Simple cost-effectiveness parameters (USD m per unit reduction of component)
        st.markdown("**Interventions**")
        i1, i2, i3, i4 = st.columns(4)
        with i1:
            cost_drought = st.number_input("Drought resilience cost per unit", min_value=0.1, value=2.0, step=0.1)
            eff_drought = st.slider("Effect on drought risk (per $m)", 0.0, 0.05, 0.02, 0.005)
        with i2:
            cost_flood = st.number_input("Flood resilience cost per unit", min_value=0.1, value=2.0, step=0.1)
            eff_flood = st.slider("Effect on flood risk (per $m)", 0.0, 0.05, 0.02, 0.005)
        with i3:
            cost_capacity = st.number_input("Capacity building cost per unit", min_value=0.1, value=1.5, step=0.1)
            eff_capacity = st.slider("Effect on literacy (pp per $m)", 0.0, 0.5, 0.2, 0.05)
        with i4:
            cost_social = st.number_input("Social protection cost per unit", min_value=0.1, value=1.5, step=0.1)
            eff_social = st.slider("Effect on poverty (pp per $m)", 0.0, 0.5, 0.2, 0.05)

        # Determine working set
        if target_scope == "Focus on selected districts" and selected_districts:
            work_df = climate_df[climate_df['district'].isin(selected_districts)].copy()
        else:
            work_df = climate_df.copy()

        # Greedy allocation by marginal CVI reduction per $m
        weights = dict(hazard= w_hazard, exposure= w_exposure, sensitivity= w_sensitivity, capacity= w_capacity)
        remaining_budget = total_budget
        allocation = {"Drought": 0.0, "Flood": 0.0, "Capacity": 0.0, "Social": 0.0}

        def compute_cvi(df_: pd.DataFrame) -> pd.Series:
            hazard = _scale_0_1((df_['drought_risk_index'] + df_['flood_risk_index'])/2)
            exposure = _scale_0_1(df_['population_density'])
            sensitivity = _scale_0_1(df_['poverty_rate'])
            capacity = _scale_0_1(df_['literacy_rate'])
            w = np.array([weights['hazard'], weights['exposure'], weights['sensitivity'], weights['capacity']], dtype=float)
            w = w / (w.sum() if w.sum() > 0 else 1)
            cvi = (w[0]*hazard + w[1]*exposure + w[2]*sensitivity + w[3]*(1-capacity))
            return _scale_0_1(cvi) * 100

        baseline_cvi = compute_cvi(work_df).mean()
        base_cvi = baseline_cvi

        # Precompute per-$m effects on components average
        def marginal_gain(intervention: str) -> float:
            tmp = work_df.copy()
            if intervention == "Drought":
                tmp['drought_risk_index'] = (tmp['drought_risk_index'] - eff_drought).clip(lower=0)
            elif intervention == "Flood":
                tmp['flood_risk_index'] = (tmp['flood_risk_index'] - eff_flood).clip(lower=0)
            elif intervention == "Capacity":
                tmp['literacy_rate'] = (tmp['literacy_rate'] + eff_capacity).clip(upper=100)
            elif intervention == "Social":
                tmp['poverty_rate'] = (tmp['poverty_rate'] - eff_social).clip(lower=0)
            return base_cvi - compute_cvi(tmp).mean()

        cost_map = {"Drought": cost_drought, "Flood": cost_flood, "Capacity": cost_capacity, "Social": cost_social}

        # Greedy loop
        max_iterations = int(total_budget * 10)
        for _ in range(max_iterations):
            if remaining_budget <= 0:
                break
            options = []
            for k in allocation.keys():
                gain = marginal_gain(k)
                score = gain / cost_map[k] if cost_map[k] > 0 else 0
                options.append((score, k, gain))
            options.sort(reverse=True)
            best = options[0]
            if best[0] <= 0:
                break
            k = best[1]
            step = min(0.5, remaining_budget)  # allocate in 0.5m increments
            allocation[k] += step
            remaining_budget -= step
            # Apply effect to working dataframe to update base for next iteration
            if k == "Drought":
                work_df['drought_risk_index'] = (work_df['drought_risk_index'] - eff_drought*step).clip(lower=0)
            elif k == "Flood":
                work_df['flood_risk_index'] = (work_df['flood_risk_index'] - eff_flood*step).clip(lower=0)
            elif k == "Capacity":
                work_df['literacy_rate'] = (work_df['literacy_rate'] + eff_capacity*step).clip(upper=100)
            elif k == "Social":
                work_df['poverty_rate'] = (work_df['poverty_rate'] - eff_social*step).clip(lower=0)
            base_cvi = compute_cvi(work_df).mean()

        # Results
        res1, res2 = st.columns([1,1])
        with res1:
            st.markdown("**Recommended Allocation (USD m)**")
            st.dataframe(pd.DataFrame({"Intervention": allocation.keys(), "USD_m": [round(v,2) for v in allocation.values()]}), use_container_width=True, height=180)
        with res2:
            st.markdown("**Projected Impact**")
            projected_cvi = compute_cvi(work_df).mean()
            st.metric("Average CVI", f"{projected_cvi:.1f}", delta=f"{(baseline_cvi - projected_cvi):.1f}")
            st.caption("Delta = reduction in CVI vs. baseline (positive is good).")
        # Optional social co-benefits note
        try:
            if 'mdpi' in climate_df.columns:
                st.info("Tip: Combine allocation with MDPI by focusing investments in districts with highest MDPI for equity.")
        except Exception:
            pass

    # New: District Profiles Page
    with main_tabs[4]:
        st.markdown('<div class="section-header">District Profiles</div>', unsafe_allow_html=True)
        st.caption("Comprehensive district-level information for evidence-based decision making.")

        # District selector
        selected_district = st.selectbox(
            "Select District",
            options=sorted(climate_df['district'].unique()),
            index=0
        )

        # Get district data
        district_data = climate_df[climate_df['district'] == selected_district].iloc[0]

        # Header with district name
        st.markdown(f"## {selected_district.title()}")

        # Create tabs for different information categories
        profile_tabs = st.tabs(["Overview", "Climate", "Demographics", "Infrastructure", "Economic"])

        with profile_tabs[0]:
            st.markdown("### Key Metrics")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Population", f"{district_data['population']:,.0f}")
            with col2:
                st.metric("Area (km²)", f"{district_data['area_km2']:,.1f}")
            with col3:
                st.metric("Density (per km²)", f"{district_data['population_density']:,.1f}")
            with col4:
                if 'mdpi' in district_data and pd.notna(district_data['mdpi']):
                    st.metric("MDPI", f"{district_data['mdpi']:.3f}")
                else:
                    st.metric("Poverty Rate", f"{district_data['poverty_rate']:.1f}%")

            # Vulnerability assessment
            st.markdown("### Climate Vulnerability Assessment")
            vuln_col1, vuln_col2, vuln_col3 = st.columns(3)
            
            with vuln_col1:
                st.metric("Vulnerability Index", f"{district_data['vulnerability_index']:.1f}")
            with vuln_col2:
                st.metric("Drought Risk", f"{district_data['drought_risk_index']:.3f}")
            with vuln_col3:
                st.metric("Flood Risk", f"{district_data['flood_risk_index']:.3f}")

        with profile_tabs[1]:
            st.markdown("### Climate Profile")
            climate_col1, climate_col2 = st.columns(2)
            
            with climate_col1:
                st.markdown("**Temperature**")
                st.metric("Average Temperature", f"{district_data['avg_temperature_c']:.1f}°C")
                
                st.markdown("**Rainfall**")
                st.metric("Average Rainfall", f"{district_data['avg_rainfall_mm']:.0f} mm")

            with climate_col2:
                # Show profile climate data if available
                if 'temp_june_max_c_profile' in district_data and pd.notna(district_data['temp_june_max_c_profile']):
                    st.markdown("**Seasonal Patterns**")
                    st.metric("June Max Temp", f"{district_data['temp_june_max_c_profile']:.1f}°C")
                    st.metric("January Min Temp", f"{district_data['temp_jan_min_c_profile']:.1f}°C")
                    st.metric("August Rainfall", f"{district_data['rainfall_aug_mm_profile']:.0f} mm")
                    st.metric("November Rainfall", f"{district_data['rainfall_nov_mm_profile']:.0f} mm")

        with profile_tabs[2]:
            st.markdown("### Demographics & Social Indicators")
            demo_col1, demo_col2 = st.columns(2)
            
            with demo_col1:
                st.markdown("**Population**")
                st.metric("Total Population", f"{district_data['population']:,.0f}")
                st.metric("Population Density", f"{district_data['population_density']:,.1f} per km²")
                
                st.markdown("**Education**")
                st.metric("Literacy Rate", f"{district_data['literacy_rate']:.1f}%")
                if 'learning_score' in district_data and pd.notna(district_data['learning_score']):
                    st.metric("Learning Score", f"{district_data['learning_score']:.1f}/100")

            with demo_col2:
                st.markdown("**Poverty & Development**")
                if 'mdpi' in district_data and pd.notna(district_data['mdpi']):
                    st.metric("MDPI", f"{district_data['mdpi']:.3f}")
                else:
                    st.metric("Poverty Rate", f"{district_data['poverty_rate']:.1f}%")
                
                st.markdown("**Basic Services**")
                if 'toilet_access_pct' in district_data and pd.notna(district_data['toilet_access_pct']):
                    st.metric("Toilet Access", f"{district_data['toilet_access_pct']:.1f}%")
                if 'electricity_availability' in district_data and pd.notna(district_data['electricity_availability']):
                    st.metric("Electricity Access", f"{district_data['electricity_availability']:.1f}%")
                if 'water_availability' in district_data and pd.notna(district_data['water_availability']):
                    st.metric("Water Access", f"{district_data['water_availability']:.1f}%")

        with profile_tabs[3]:
            st.markdown("### Infrastructure & Services")
            infra_col1, infra_col2 = st.columns(2)
            
            with infra_col1:
                st.markdown("**Education Infrastructure**")
                if 'num_primary_schools' in district_data and pd.notna(district_data['num_primary_schools']):
                    st.metric("Primary Schools", f"{district_data['num_primary_schools']:.0f}")
                if 'num_middle_schools' in district_data and pd.notna(district_data['num_middle_schools']):
                    st.metric("Middle Schools", f"{district_data['num_middle_schools']:.0f}")
                if 'num_high_schools' in district_data and pd.notna(district_data['num_high_schools']):
                    st.metric("High Schools", f"{district_data['num_high_schools']:.0f}")
                if 'num_higher_secondary_schools' in district_data and pd.notna(district_data['num_higher_secondary_schools']):
                    st.metric("Higher Secondary", f"{district_data['num_higher_secondary_schools']:.0f}")

            with infra_col2:
                st.markdown("**Health Infrastructure**")
                if 'govt_health_institutions' in district_data and pd.notna(district_data['govt_health_institutions']):
                    st.metric("Govt Health Institutions", f"{district_data['govt_health_institutions']:.0f}")
                if 'bed_strength' in district_data and pd.notna(district_data['bed_strength']):
                    st.metric("Hospital Beds", f"{district_data['bed_strength']:.0f}")
                if 'private_health_institutions' in district_data and pd.notna(district_data['private_health_institutions']):
                    st.metric("Private Health Institutions", f"{district_data['private_health_institutions']:.0f}")

        with profile_tabs[4]:
            st.markdown("### Economic Profile")
            econ_col1, econ_col2 = st.columns(2)
            
            with econ_col1:
                st.markdown("**Employment & Industry**")
                if 'employment_cost' in district_data and pd.notna(district_data['employment_cost']):
                    st.metric("Employment Cost", f"{district_data['employment_cost']:,.0f}")
                if 'salaries' in district_data and pd.notna(district_data['salaries']):
                    st.metric("Total Salaries", f"{district_data['salaries']:,.0f}")
                if 'value_produced_manufacturing' in district_data and pd.notna(district_data['value_produced_manufacturing']):
                    st.metric("Manufacturing Value", f"{district_data['value_produced_manufacturing']:,.0f}")

            with econ_col2:
                st.markdown("**Agriculture**")
                if 'number_of_cattle' in district_data and pd.notna(district_data['number_of_cattle']):
                    st.metric("Cattle Population", f"{district_data['number_of_cattle']:,.0f}")
                if 'cultivated_area' in district_data and pd.notna(district_data['cultivated_area']):
                    st.metric("Cultivated Area (ha)", f"{district_data['cultivated_area']:,.0f}")
                if 'cropped_area' in district_data and pd.notna(district_data['cropped_area']):
                    st.metric("Cropped Area (ha)", f"{district_data['cropped_area']:,.0f}")

        # Priority recommendations
        st.markdown("### Priority Recommendations")
        recommendations = []
        
        if district_data['drought_risk_index'] > 0.7:
            recommendations.append("**Drought Resilience**: Implement water harvesting systems, promote drought-tolerant crops, and establish drip irrigation infrastructure.")
        if district_data['flood_risk_index'] > 0.7:
            recommendations.append("**Flood Protection**: Develop early warning systems, construct embankments, and implement asset elevation programs.")
        if district_data['poverty_rate'] > 40 or ('mdpi' in district_data and district_data['mdpi'] > 0.3):
            recommendations.append("**Social Protection**: Establish cash-for-work programs on resilience assets and provide livelihood grants.")
        if district_data['literacy_rate'] < 55:
            recommendations.append("**Capacity Building**: Implement community climate information services and risk education programs.")
        if 'toilet_access_pct' in district_data and district_data['toilet_access_pct'] < 50:
            recommendations.append("**Sanitation**: Improve toilet access and hygiene infrastructure.")
        if 'electricity_availability' in district_data and district_data['electricity_availability'] < 70:
            recommendations.append("**Energy Access**: Expand electricity infrastructure and promote renewable energy solutions.")

        if not recommendations:
            recommendations.append("**Monitoring**: Maintain risk monitoring systems and update local adaptation plans regularly.")

        for i, rec in enumerate(recommendations, 1):
            st.markdown(f"{i}. {rec}")

        # Export district data
        st.markdown("### Export District Data")
        district_export = climate_df[climate_df['district'] == selected_district]
        st.download_button(
            f"Download {selected_district.title()} Profile",
            district_export.to_csv(index=False).encode('utf-8'),
            file_name=f"{selected_district.lower().replace(' ', '_')}_profile.csv",
            mime="text/csv"
        )

    with tabs[5]:
        df_to_show = climate_df
        if search_term:
            df_to_show = df_to_show[df_to_show['district'].str.contains(search_term, case=False)]
        st.dataframe(df_to_show, use_container_width=True, height=420)
    
    with main_tabs[5]:
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
