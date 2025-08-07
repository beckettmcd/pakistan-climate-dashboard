import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

# --------------
# App config
# --------------
st.set_page_config(
    page_title="Pakistan Climate Dashboard",
    page_icon="🌤️",
    layout="wide",
)

# --------------
# Helpers
# --------------
@st.cache_data(show_spinner=False)
def load_table(data_path: Path) -> pd.DataFrame:
    if not data_path.exists():
        return pd.DataFrame()
    if data_path.suffix.lower() in {".csv"}:
        return pd.read_csv(data_path)
    if data_path.suffix.lower() in {".parquet"}:
        return pd.read_parquet(data_path)
    if data_path.suffix.lower() in {".feather"}:
        return pd.read_feather(data_path)
    raise ValueError(f"Unsupported data format: {data_path.suffix}")


def normalize_name(name: Optional[str]) -> Optional[str]:
    if pd.isna(name) or name is None:
        return None
    s = str(name).strip().lower()
    for ch in ["-", ",", ".", "'", "`", "(", ")", "[", "]", "/", "\\"]:
        s = s.replace(ch, " ")
    s = " ".join(s.split())
    return s


def ensure_datetime(series: pd.Series) -> pd.Series:
    if np.issubdtype(series.dtype, np.datetime64):
        return series
    return pd.to_datetime(series, errors="coerce")


@st.cache_data(show_spinner=False)
def load_geojson(geojson_bytes: bytes) -> Dict:
    return json.loads(geojson_bytes.decode("utf-8"))


def build_color_continuous_scale() -> List[str]:
    # Teal to orange scale
    return [
        "#003f5c",
        "#2f4b7c",
        "#665191",
        "#a05195",
        "#d45087",
        "#f95d6a",
        "#ff7c43",
        "#ffa600",
    ]


def add_normalized_columns(df: pd.DataFrame, district_col: str, province_col: Optional[str]) -> pd.DataFrame:
    df = df.copy()
    df["district_norm"] = df[district_col].apply(normalize_name)
    if province_col and province_col in df.columns:
        df["province_norm"] = df[province_col].apply(normalize_name)
    else:
        df["province_norm"] = None
    return df


def aggregate_data(
    df: pd.DataFrame,
    date_col: str,
    district_col: str,
    province_col: Optional[str],
    value_col: str,
    agg: str,
) -> pd.DataFrame:
    df = df.copy()
    df[date_col] = ensure_datetime(df[date_col])
    df = df.dropna(subset=[date_col])

    if agg == "Daily":
        df["period"] = df[date_col].dt.date
    elif agg == "Monthly":
        df["period"] = df[date_col].dt.to_period("M").dt.to_timestamp()
    elif agg == "Seasonal":
        # DJF, MAM, JJA, SON by meteorological seasons
        month = df[date_col].dt.month
        season = pd.cut(
            month,
            bins=[0, 2, 5, 8, 11, 12],
            labels=["DJF", "MAM", "JJA", "SON", "DJF"],
            include_lowest=True,
            right=True,
        )
        year = df[date_col].dt.year
        # Adjust year for DJF
        year = np.where((month == 12), year + 1, year)
        df["period"] = pd.to_datetime(year.astype(str) + "-" + season.astype(str), errors="coerce")
        # Keep a display label too
        df["season_label"] = season
    elif agg == "Annual":
        df["period"] = pd.to_datetime(df[date_col].dt.year.astype(str) + "-01-01")
    else:
        df["period"] = df[date_col]

    group_cols = ["period", district_col]
    if province_col and province_col in df.columns:
        group_cols.append(province_col)

    agg_df = (
        df.groupby(group_cols, dropna=False)[value_col]
        .mean()
        .reset_index()
        .rename(columns={value_col: "value"})
    )
    return agg_df


def compute_kpis(filtered_df: pd.DataFrame) -> Tuple[Optional[float], Optional[float]]:
    if filtered_df.empty:
        return (None, None)
    return (
        float(np.nanmean(filtered_df["value"].to_numpy(), dtype=float)),
        float(np.nanmedian(filtered_df["value"].to_numpy(), dtype=float)),
    )


def join_values_to_geojson(geojson: Dict, values_df: pd.DataFrame, geojson_district_key: str, df_district_key: str) -> Dict:
    # Build lookup
    lookup = (
        values_df.dropna(subset=[df_district_key])
        .groupby(df_district_key)["value"]
        .mean()
        .to_dict()
    )

    for feature in geojson.get("features", []):
        props = feature.get("properties", {})
        district_value = props.get(geojson_district_key)
        district_norm = normalize_name(district_value)
        feature["properties"]["metric_value"] = lookup.get(district_norm)
    return geojson


# --------------
# Sidebar - Data inputs
# --------------
st.sidebar.title("Data Inputs")

DEFAULT_DATA_PATH = Path(__file__).parent / "data/Sample_District_Climate_Data.csv"
DEFAULT_GEOJSON_PATH = Path(__file__).parent / "data/geoBoundaries-PAK-ADM2 (1).geojson"

st.sidebar.subheader("Climate Data")
use_uploaded = st.sidebar.toggle("Upload data file now", value=False, help="You can also place a file at src/data/Sample_District_Climate_Data.csv")

uploaded_file = None
if use_uploaded:
    uploaded_file = st.sidebar.file_uploader("Upload CSV or Parquet", type=["csv", "parquet", "feather"], accept_multiple_files=False)

if uploaded_file is not None:
    if uploaded_file.name.endswith(".csv"):
        df_raw = pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith(".parquet"):
        df_raw = pd.read_parquet(uploaded_file)
    elif uploaded_file.name.endswith(".feather"):
        df_raw = pd.read_feather(uploaded_file)
    else:
        st.sidebar.warning("Unsupported file type. Please upload CSV, Parquet, or Feather.")
        df_raw = pd.DataFrame()
else:
    df_raw = load_table(DEFAULT_DATA_PATH)

st.sidebar.subheader("Boundary Geometry (GeoJSON)")
use_uploaded_geo = st.sidebar.toggle("Upload GeoJSON now", value=False, help="You can also place a file at src/data/geoBoundaries-PAK-ADM2 (1).geojson")
geojson_obj = None
if use_uploaded_geo:
    uploaded_geo = st.sidebar.file_uploader("Upload GeoJSON", type=["geojson", "json"], accept_multiple_files=False)
    if uploaded_geo is not None:
        geojson_obj = load_geojson(uploaded_geo.read())
else:
    if DEFAULT_GEOJSON_PATH.exists():
        geojson_obj = load_geojson(DEFAULT_GEOJSON_PATH.read_bytes())


# --------------
# Sidebar - Schema mapping
# --------------
st.sidebar.subheader("Column Mapping")

if df_raw.empty:
    st.info("Add your climate data to begin. Expected columns: a district name, optional province, a date, and one or more metric columns.")

columns = list(df_raw.columns) if not df_raw.empty else []

def_guess = {
    "district": next((c for c in columns if c.lower() in {"district", "district_name", "adm2", "adm_2"}), "district"),
    "province": next((c for c in columns if c.lower() in {"province", "state", "adm1", "adm_1"}), None),
    "date": next((c for c in columns if c.lower() in {"date", "day", "time", "timestamp"}), None),
}

metric_candidates = [c for c in columns if c.lower() in {"tavg", "tmin", "tmax", "temp", "temperature", "precip", "ppt", "rain", "avg_temperature_c", "avg_rainfall_mm", "drought_risk_index", "flood_risk_index"}]

col_district = st.sidebar.selectbox("District column", options=[None] + columns, index=(columns.index(def_guess["district"]) + 1) if def_guess["district"] in columns else 0)
col_province = st.sidebar.selectbox("Province column (optional)", options=[None] + columns, index=(columns.index(def_guess["province"]) + 1) if def_guess["province"] in columns else 0)
col_date = st.sidebar.selectbox("Date column", options=[None] + columns, index=(columns.index(def_guess["date"]) + 1) if def_guess["date"] in columns else 0)

col_metric = st.sidebar.selectbox("Metric column", options=[None] + (metric_candidates if metric_candidates else columns), index=(metric_candidates.index("avg_temperature_c") + 1) if "avg_temperature_c" in metric_candidates else 0)
metric_label = st.sidebar.text_input("Metric label (for charts)", value="Average Temperature (°C)" if col_metric == "avg_temperature_c" else (col_metric or "value"))

# GeoJSON district property
geojson_district_key = None
if geojson_obj is not None:
    example_props = geojson_obj.get("features", [{}])[0].get("properties", {}) if geojson_obj.get("features") else {}
    props_keys = list(example_props.keys())
    geojson_district_key = st.sidebar.selectbox(
    "GeoJSON district property",
    options=props_keys if props_keys else ["shapeName"],
    index=(props_keys.index("shapeName") if "shapeName" in props_keys else 0),
    help="Choose the property that contains the district name",
)

# --------------
# Main UI
# --------------
st.title("Pakistan Climate Dashboard")
st.caption("Explore climate metrics across districts of Pakistan. Upload your data and a districts GeoJSON to begin.")

if df_raw.empty or col_district is None or col_date is None or col_metric is None:
    st.warning("Waiting for data and column mapping...")
    st.stop()

# Filters
with st.sidebar:
    st.subheader("Filters")
    df_dates = ensure_datetime(df_raw[col_date])
    min_date = pd.to_datetime(df_dates.min()).date() if not df_raw.empty else None
    max_date = pd.to_datetime(df_dates.max()).date() if not df_raw.empty else None
    date_range = st.date_input("Date range", value=(min_date, max_date) if min_date and max_date else None)
    agg_grain = st.select_slider("Aggregation", options=["Daily", "Monthly", "Seasonal", "Annual"], value="Monthly")

    # district/province filter options
    df_norm = add_normalized_columns(df_raw, col_district, col_province)
    all_provinces = sorted(set([p for p in df_norm.get("province_norm", []).tolist() if p]))
    prov_choice = st.multiselect("Province(s)", options=all_provinces, default=[])

    # filter by province early for district listing
    if prov_choice:
        df_list = df_norm[df_norm["province_norm"].isin(prov_choice)]
    else:
        df_list = df_norm
    all_districts = sorted(set([d for d in df_list["district_norm"].tolist() if d]))
    dist_choice = st.multiselect("District(s)", options=all_districts, default=[])

# Apply filters
df = df_norm.copy()

# date filter
df[col_date] = ensure_datetime(df[col_date])
if isinstance(date_range, (list, tuple)) and len(date_range) == 2 and date_range[0] and date_range[1]:
    start_dt = pd.to_datetime(date_range[0])
    end_dt = pd.to_datetime(date_range[1])
    df = df[(df[col_date] >= start_dt) & (df[col_date] <= end_dt)]

# province filter
if prov_choice:
    df = df[df["province_norm"].isin(prov_choice)]
# district filter
if dist_choice:
    df = df[df["district_norm"].isin(dist_choice)]

# Aggregate
agg_df = aggregate_data(df, date_col=col_date, district_col="district_norm", province_col="province_norm", value_col=col_metric, agg=agg_grain)

# KPIs
kpi_mean, kpi_median = compute_kpis(agg_df)

k1, k2, k3 = st.columns(3)
with k1:
    st.metric(label=f"Average {metric_label}", value=(f"{kpi_mean:.2f}" if kpi_mean is not None else "-"))
with k2:
    st.metric(label=f"Median {metric_label}", value=(f"{kpi_median:.2f}" if kpi_median is not None else "-"))
with k3:
    st.metric(label="Observations", value=f"{len(df)}")

# Time series
st.subheader("Time Series")
try:
    ts = agg_df.groupby("period")["value"].mean().reset_index()
    fig_ts = px.line(ts, x="period", y="value", markers=True, labels={"value": metric_label, "period": "Period"})
    st.plotly_chart(fig_ts, use_container_width=True)
except Exception as e:
    st.info(f"Time series not available: {e}")

# Choropleth Map
st.subheader("District Choropleth")
if geojson_obj is None:
    st.info("Provide a districts GeoJSON to enable the map.")
else:
    if not geojson_district_key:
        st.info("Select the GeoJSON district property in the sidebar.")
    else:
        # Build a latest-period snapshot for mapping
        try:
            latest_period = agg_df["period"].max()
            map_df = agg_df[agg_df["period"] == latest_period]
            # Join values onto geojson
            geojson_with_values = join_values_to_geojson(geojson_obj, map_df.rename(columns={"district_norm": "district_norm"}), geojson_district_key=geojson_district_key, df_district_key="district_norm")

            # Color scale
            values = [f["properties"].get("metric_value") for f in geojson_with_values.get("features", [])]
            vmin = np.nanmin(values) if len(values) else None
            vmax = np.nanmax(values) if len(values) else None

            import pydeck as pdk

            tooltip = {
                "html": f"<b>{{{geojson_district_key}}}</b><br/>{metric_label}: {{{{metric_value}}}}",
                "style": {"backgroundColor": "steelblue", "color": "white"},
            }

            deck = pdk.Deck(
                map_style="mapbox://styles/mapbox/light-v11",
                initial_view_state=pdk.ViewState(latitude=30.3753, longitude=69.3451, zoom=4.5),
                layers=[
                    pdk.Layer(
                        "GeoJsonLayer",
                        geojson_with_values,
                        pickable=True,
                        stroked=True,
                        filled=True,
                        extruded=False,
                        get_fill_color="[255*(1 - (properties.metric_value - min_val) / (max_val - min_val + 1e-9)), 128, 255*(properties.metric_value - min_val) / (max_val - min_val + 1e-9), 180]",
                        get_line_color=[0, 0, 0, 80],
                        line_width_min_pixels=0.5,
                        parameters={"min_val": float(vmin) if vmin is not None else 0.0, "max_val": float(vmax) if vmax is not None else 1.0},
                    )
                ],
                tooltip=tooltip,
            )
            st.pydeck_chart(deck)
        except Exception as e:
            st.info(f"Map not available: {e}")

# Data table
st.subheader("Data Preview")
with st.expander("Show raw and aggregated data"):
    st.write("Raw data", df_raw.head(100))
    st.write("Aggregated data", agg_df.head(200))

# Footer help
st.markdown(
    """
    Tip: If district names don't match your GeoJSON, normalize them or provide a mapping table. Names are normalized by lowercase and punctuation removal.
    """
)
