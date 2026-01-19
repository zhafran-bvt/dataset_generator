import random
import pandas as pd
import json
import os
import multiprocessing as mp
import numpy as np
from shapely.geometry import shape, Point, Polygon, MultiPolygon
from shapely.ops import unary_union
from shapely.wkt import loads
from shapely.prepared import prep
from tqdm import tqdm
from shapely.strtree import STRtree
from datetime import datetime, timedelta
import geopandas as gpd
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
import math
from scipy.stats import multivariate_normal, gaussian_kde
import logging
import argparse
import psutil
import numba
from functools import lru_cache
from scipy import stats
import matplotlib.pyplot as plt

# Optional H3 support (not currently used by the generator but required by the API layer)
try:
    import h3  # noqa: F401

    H3_AVAILABLE = True
except ImportError:
    H3_AVAILABLE = False

# --- CONFIGURATION ---

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore')

GENDER_OPTIONS = ['Male', 'Female', 'Other']
OCCUPATION_OPTIONS = ['Employed', 'Unemployed', 'Student', 'Retired', 'Homemaker', 'Other']
EDUCATION_OPTIONS = ['Less than High School', 'High School', 'Some College', 'Associate Degree', "Bachelor's Degree", "Master's Degree", 'Doctorate']
EMPLOYMENT_STATUS_OPTIONS = ['Full-time', 'Part-time', 'Self-employed', 'Unemployed', 'Retired', 'Student', 'Other']
HEALTHCARE_ACCESS_OPTIONS = [True, False]

REALISTIC_LABELS = [
    "Population", "Birth Rate", "Death Rate", "Unemployment Rate",
    "Income per Capita", "GDP Growth", "Education Level",
    "Health Index", "Crime Rate", "Urbanization Rate",
    "Life Expectancy", "Poverty Rate", "Employment Rate",
    "Literacy Rate", "Housing Density", "Public Transport Usage",
    "Internet Penetration", "Energy Consumption",
    "Water Access", "Sanitation Access",
    "Median Age", "Household Size", "Housing Cost Index",
    "Hospital Beds per 1k", "Air Quality Index", "CO2 Emissions per Capita"
]
VALUE_RANGES = {
    "Population": (1000, 1000000),
    "Birth Rate": (5, 30),
    "Death Rate": (2, 15),
    "Unemployment Rate": (0, 20),
    "Income per Capita": (500, 50000),
    "GDP Growth": (-5, 10),
    "Education Level": (1, 10),
    "Health Index": (0, 100),
    "Crime Rate": (0, 50),
    "Urbanization Rate": (10, 100),
    "Life Expectancy": (50, 85),
    "Poverty Rate": (0, 50),
    "Employment Rate": (50, 100),
    "Literacy Rate": (50, 100),
    "Housing Density": (100, 10000),
    "Public Transport Usage": (0, 80),
    "Internet Penetration": (0, 100),
    "Energy Consumption": (100, 10000),
    "Water Access": (50, 100),
    "Sanitation Access": (50, 100),
    "Household Income": (0, 1000000),
    "Median Age": (15, 50),
    "Household Size": (1, 7),
    "Housing Cost Index": (50, 200),
    "Hospital Beds per 1k": (0.2, 10.0),
    "Air Quality Index": (0, 300),
    "CO2 Emissions per Capita": (0.5, 20.0)
}
VARIABLE_CORRELATIONS = [
    ("Income per Capita", "Education Level", 0.7),
    ("Income per Capita", "Household Income", 0.85),
    ("Household Income", "Education Level", 0.65),
    ("Income per Capita", "Poverty Rate", -0.8),
    ("Income per Capita", "Life Expectancy", 0.6),
    ("Income per Capita", "Health Index", 0.5),
    ("Health Index", "Life Expectancy", 0.75),
    ("Poverty Rate", "Health Index", -0.65),
    ("Water Access", "Health Index", 0.55),
    ("Sanitation Access", "Health Index", 0.6),
    ("Education Level", "Literacy Rate", 0.8),
    ("Education Level", "Employment Rate", 0.6),
    ("Education Level", "Birth Rate", -0.5),
    ("Urbanization Rate", "Internet Penetration", 0.65),
    ("Urbanization Rate", "Public Transport Usage", 0.7),
    ("Urbanization Rate", "Energy Consumption", 0.55),
    ("Urbanization Rate", "Housing Cost Index", 0.6),
    ("Population", "Housing Density", 0.6),
    ("Housing Density", "Public Transport Usage", 0.5),
    ("Housing Density", "Household Size", 0.45),
    ("Employment Rate", "Unemployment Rate", -0.9),
    ("GDP Growth", "Employment Rate", 0.6),
    ("GDP Growth", "Unemployment Rate", -0.5),
    ("Life Expectancy", "Median Age", 0.4),
    ("Birth Rate", "Median Age", -0.6),
    ("Health Index", "Hospital Beds per 1k", 0.5),
    ("Air Quality Index", "Health Index", -0.5),
    ("Energy Consumption", "CO2 Emissions per Capita", 0.7)
]
BOUNDING_BOXES = {
    1: {"name": "Jakarta", "lon_min": 106.65, "lon_max": 106.95, "lat_min": -6.35, "lat_max": -6.1},
    2: {"name": "Yogyakarta", "lon_min": 110.35, "lon_max": 110.5, "lat_min": -7.85, "lat_max": -7.75},
    3: {"name": "Indonesia", "lon_min": 95.0, "lon_max": 141.0, "lat_min": -11.0, "lat_max": 6.0},
    4: {"name": "Japan", "lon_min": 129.0, "lon_max": 146.0, "lat_min": 30.0, "lat_max": 45.0},
    5: {"name": "Vietnam", "lon_min": 102.0, "lon_max": 110.0, "lat_min": 8.0, "lat_max": 23.0}
}

def validate_input(prompt, valid_options, type_cast=str, allow_empty=False):
    while True:
        try:
            value = input(prompt).strip()
            if allow_empty and value == "":
                return value
            value = type_cast(value)
            if valid_options and value not in valid_options:
                logger.warning(f"Invalid input. Choose from: {valid_options}")
                continue
            return value
        except ValueError:
            logger.warning(f"Invalid input. Expected a {type_cast.__name__}.")

def load_config(config_path):
    with open(config_path, 'r') as f:
        return json.load(f)

class CorrelationEngine:
    def __init__(self, correlations, value_ranges, distribution_mode="uniform"):
        self.correlations = correlations
        self.value_ranges = value_ranges
        self.distribution_mode = (distribution_mode or "uniform").lower()
    def sample_value(self, min_val, max_val):
        unit_value = sample_unit_distribution(1, self.distribution_mode)[0]
        return unit_value * (max_val - min_val) + min_val
    def generate_batch_values(self, labels, batch_size=10000, seed=None):
        if seed is not None:
            np.random.seed(seed)
        values = {
            label: sample_unit_distribution(batch_size, self.distribution_mode)
            for label in labels if label in self.value_ranges
        }
        for _ in range(3):
            for var1, var2, corr_strength in self.correlations:
                if var1 in values and var2 in values:
                    val1 = values[var1]
                    target_val2 = val1 if corr_strength > 0 else 1 - val1
                    adjustment = (target_val2 - values[var2]) * abs(corr_strength) * 0.5
                    values[var2] = np.clip(values[var2] + adjustment, 0, 1)
        result = {}
        for label, norm_vals in values.items():
            min_val, max_val = self.value_ranges[label]
            result[label] = norm_vals * (max_val - min_val) + min_val
            if label == "Education Level":
                edu_levels = []
                for val in result[label]:
                    edu_num = int(val)
                    edu_num = max(1, min(edu_num, 10))
                    if edu_num <= 2:
                        edu_levels.append("Less than High School")
                    elif edu_num <= 4:
                        edu_levels.append("High School")
                    elif edu_num <= 5:
                        edu_levels.append("Some College")
                    elif edu_num <= 6:
                        edu_levels.append("Associate Degree")
                    elif edu_num <= 8:
                        edu_levels.append("Bachelor's Degree")
                    elif edu_num <= 9:
                        edu_levels.append("Master's Degree")
                    else:
                        edu_levels.append("Doctorate")
                result[label] = np.array(edu_levels)
            else:
                result[label] = np.round(result[label], 2)
        return result

def parse_datetime(value):
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        cleaned = value.strip()
        if cleaned.endswith("Z"):
            cleaned = cleaned[:-1]
        try:
            return datetime.fromisoformat(cleaned)
        except ValueError:
            logger.warning(f"Invalid datetime format: {value}. Using defaults.")
            return None
    return None

def generate_random_datetimes(n, start_dt=None, end_dt=None, seasonality="none"):
    start_dt = parse_datetime(start_dt) or datetime(2025, 1, 1, 0, 0, 0)
    end_dt = parse_datetime(end_dt) or datetime(2025, 12, 31, 23, 59, 59)
    if end_dt < start_dt:
        start_dt, end_dt = end_dt, start_dt
    start_date = start_dt.date()
    end_date = end_dt.date()
    total_days = (end_date - start_date).days + 1
    dates = [start_date + timedelta(days=i) for i in range(total_days)]
    weights = np.ones(total_days, dtype=float)
    if seasonality == "weekday":
        weights = np.array([1.0 if d.weekday() < 5 else 0.4 for d in dates], dtype=float)
    elif seasonality == "monthly":
        weights = np.array([1.5 if d.month in (6, 7, 8) else 1.0 for d in dates], dtype=float)
    weights /= weights.sum()
    day_indices = np.random.choice(total_days, size=n, p=weights)
    seconds = np.random.randint(0, 86400, size=n)
    dates_out = []
    for i in range(n):
        day = dates[day_indices[i]]
        dt = datetime.combine(day, datetime.min.time()) + timedelta(seconds=int(seconds[i]))
        dates_out.append(dt.strftime("%Y-%m-%dT%H:%M:%SZ"))
    return dates_out

@numba.njit
def is_point_in_polygon(polygon_coords, point):
    x, y = point
    n = len(polygon_coords)
    inside = False
    p1x, p1y = polygon_coords[0]
    for i in range(1, n):
        p2x, p2y = polygon_coords[i]
        if y > min(p1y, p2y) and y <= max(p1y, p2y) and x <= max(p1x, p2x):
            if p1y != p2y:
                xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                if p1x == p2x or x <= xinters:
                    inside = not inside
        p1x, p1y = p2x, p2y
    return inside

def get_random_points_in_bbox(lon_min, lon_max, lat_min, lat_max, count):
    lons = np.random.uniform(lon_min, lon_max, count)
    lats = np.random.uniform(lat_min, lat_max, count)
    return np.column_stack((lons, lats))

def extract_polygon_coords(land_geometry):
    if isinstance(land_geometry, Polygon):
        return [np.array(land_geometry.exterior.coords)]
    elif isinstance(land_geometry, MultiPolygon):
        return [np.array(geom.exterior.coords) for geom in land_geometry.geoms]
    return []

def random_sample_uniform(n, lon_min, lon_max, lat_min, lat_max):
    return get_random_points_in_bbox(lon_min, lon_max, lat_min, lat_max, n)

def _normalize_to_unit(values):
    min_val = float(np.min(values))
    max_val = float(np.max(values))
    if max_val == min_val:
        return np.zeros_like(values)
    return (values - min_val) / (max_val - min_val)

def sample_unit_distribution(n, mode):
    if mode == "beta":
        return np.random.beta(2.0, 5.0, n)
    if mode == "normal":
        return _normalize_to_unit(np.random.normal(0.0, 1.0, n))
    if mode == "lognormal":
        return _normalize_to_unit(np.random.lognormal(mean=0.0, sigma=1.0, size=n))
    return np.random.random(n)

def random_sample_in_geometry(
    n,
    polygons,
    lon_min,
    lon_max,
    lat_min,
    lat_max,
    batch_size=1000,
    land_shape=None,
    land_prepared=None,
    strict_land=False,
):
    points = []
    attempts = 0
    max_attempts = n * (50 if strict_land else 10)
    if land_prepared is None and land_shape is not None:
        land_prepared = prep(land_shape)
    while len(points) < n and attempts < max_attempts:
        candidates = get_random_points_in_bbox(lon_min, lon_max, lat_min, lat_max, min(batch_size, n - len(points)))
        for point in candidates:
            if len(points) >= n:
                break
            if land_prepared is not None:
                if land_prepared.covers(Point(point[0], point[1])):
                    points.append(point)
                    continue
            if land_prepared is None and polygons:
                for poly in polygons:
                    if is_point_in_polygon(poly, point):
                        points.append(point)
                        break
        attempts += len(candidates)
    if len(points) < n:
        if strict_land and (land_prepared is not None or polygons):
            raise RuntimeError("Unable to sample enough land points within max attempts.")
        remaining = n - len(points)
        points.extend(get_random_points_in_bbox(lon_min, lon_max, lat_min, lat_max, remaining))
    return np.array(points[:n])

def _point_in_any_polygon(polygons, point):
    for poly in polygons:
        if is_point_in_polygon(poly, point):
            return True
    return False

def _build_rect_around_point(lon, lat, width, height, lon_min, lon_max, lat_min, lat_max):
    half_w = width / 2.0
    half_h = height / 2.0
    min_x = max(lon_min, lon - half_w)
    max_x = min(lon_max, lon + half_w)
    min_y = max(lat_min, lat - half_h)
    max_y = min(lat_max, lat + half_h)
    if max_x <= min_x or max_y <= min_y:
        return None
    return [
        [min_x, min_y],
        [max_x, min_y],
        [max_x, max_y],
        [min_x, max_y],
        [min_x, min_y],
    ]

def generate_random_geom_batch(params):
    (
        geom_type,
        format_type,
        n,
        lon_min,
        lon_max,
        lat_min,
        lat_max,
        points,
        batch_id,
        polygons,
        land_shape,
        strict_land,
    ) = params
    land_prepared = prep(land_shape) if land_shape is not None else None
    geoms = []
    if geom_type == "POINT":
        if format_type == "WKT":
            for i in range(n):
                lon, lat = points[i]
                geoms.append(f"POINT ({lon:.6f} {lat:.6f})")
        else:
            for i in range(n):
                lon, lat = points[i]
                geoms.append(json.dumps({"type": "Point", "coordinates": [lon, lat]}))
    elif geom_type in ["POLYGON", "MULTIPOLYGON"]:
        lon_extent = lon_max - lon_min
        lat_extent = lat_max - lat_min
        for i in range(n):
            lon, lat = points[i]
            coords = None
            width = random.uniform(lon_extent * 0.005, lon_extent * 0.05)
            height = random.uniform(lat_extent * 0.005, lat_extent * 0.05)
            for _ in range(15):
                candidate = _build_rect_around_point(
                    lon, lat, width, height, lon_min, lon_max, lat_min, lat_max
                )
                if candidate is None:
                    width *= 0.5
                    height *= 0.5
                    continue
                candidate_ok = False
                if land_prepared is not None:
                    candidate_ok = land_prepared.covers(Polygon(candidate))
                if not candidate_ok and land_prepared is None and polygons:
                    candidate_ok = all(_point_in_any_polygon(polygons, pt) for pt in candidate)
                if land_prepared is None and not polygons:
                    candidate_ok = True
                if candidate_ok:
                    coords = candidate
                    break
                width *= 0.5
                height *= 0.5
            if coords is None:
                min_size = min(lon_extent, lat_extent) * 0.0005
                min_size = max(min_size, 1e-6)
                coords = _build_rect_around_point(
                    lon, lat, min_size, min_size, lon_min, lon_max, lat_min, lat_max
                )
                if coords is None or (strict_land and land_prepared is not None and not land_prepared.covers(Polygon(coords))):
                    raise RuntimeError("Unable to generate land-only polygon geometry.")
            if format_type == "WKT":
                coord_str = ", ".join([f"{x:.6f} {y:.6f}" for x, y in coords])
                if geom_type == "POLYGON":
                    geoms.append(f"POLYGON (({coord_str}))")
                else:
                    geoms.append(f"MULTIPOLYGON ((({coord_str})))")
            else:
                if geom_type == "POLYGON":
                    geoms.append(json.dumps({"type": "Polygon", "coordinates": [coords]}))
                else:
                    geoms.append(json.dumps({"type": "MultiPolygon", "coordinates": [[coords]]}))
    return geoms

def create_spatial_clustering_model(
    lon_min,
    lon_max,
    lat_min,
    lat_max,
    land_geometry,
    cluster_count=5,
    points_per_cluster=200,
    strict_land=False,
):
    land_shape = None
    if land_geometry is not None:
        polygons = extract_polygon_coords(land_geometry)
        land_shape = land_geometry
        land_prepared = prep(land_shape)
        centers = random_sample_in_geometry(
            cluster_count,
            polygons,
            lon_min,
            lon_max,
            lat_min,
            lat_max,
            land_shape=land_shape,
            land_prepared=land_prepared,
            strict_land=strict_land,
        )
    else:
        centers = get_random_points_in_bbox(lon_min, lon_max, lat_min, lat_max, cluster_count)
    samples = []
    weights = np.random.uniform(0.5, 1.5, cluster_count)
    weights = weights / np.sum(weights)
    lon_extent = lon_max - lon_min
    lat_extent = lat_max - lat_min
    avg_cluster_size = min(lon_extent, lat_extent) * 0.05
    for i, (center_x, center_y) in enumerate(centers):
        cluster_size_x = avg_cluster_size * random.uniform(0.5, 2.0)
        cluster_size_y = avg_cluster_size * random.uniform(0.5, 2.0)
        cluster_points = int(points_per_cluster * weights[i])
        x = np.random.normal(center_x, cluster_size_x, cluster_points)
        y = np.random.normal(center_y, cluster_size_y, cluster_points)
        x = np.clip(x, lon_min, lon_max)
        y = np.clip(y, lat_min, lat_max)
        samples.extend([[x[j], y[j]] for j in range(cluster_points)])
    samples = np.array(samples)
    kde_model = gaussian_kde(samples.T)
    return {
        'model': kde_model,
        'use_probability': 0.85,
        'centers': centers
    }

def generate_points_for_batch(params):
    (
        batch_size,
        lon_min,
        lon_max,
        lat_min,
        lat_max,
        use_spatial_clustering,
        spatial_clusters,
        polygons,
        land_shape,
        strict_land,
    ) = params
    land_prepared = prep(land_shape) if land_shape is not None else None
    if use_spatial_clustering and spatial_clusters and random.random() < spatial_clusters['use_probability']:
        samples = spatial_clusters['model'].resample(batch_size)
        points = np.column_stack((samples[0], samples[1]))
        points[:, 0] = np.clip(points[:, 0], lon_min, lon_max)
        points[:, 1] = np.clip(points[:, 1], lat_min, lat_max)
        if polygons or land_prepared is not None:
            for i in range(len(points)):
                is_valid = False
                if land_prepared is not None and land_prepared.covers(Point(points[i][0], points[i][1])):
                    is_valid = True
                elif land_prepared is None:
                    for poly in polygons:
                        if is_point_in_polygon(poly, points[i]):
                            is_valid = True
                            break
                if not is_valid:
                    valid_points = random_sample_in_geometry(
                        1,
                        polygons,
                        lon_min,
                        lon_max,
                        lat_min,
                        lat_max,
                        land_shape=land_shape,
                        land_prepared=land_prepared,
                        strict_land=strict_land,
                    )
                    points[i] = valid_points[0]
    elif polygons or land_shape is not None:
        points = random_sample_in_geometry(
            batch_size,
            polygons,
            lon_min,
            lon_max,
            lat_min,
            lat_max,
            land_shape=land_shape,
            land_prepared=land_prepared,
            strict_land=strict_land,
        )
    else:
        points = random_sample_uniform(batch_size, lon_min, lon_max, lat_min, lat_max)
    return points

def generate_batch_data(batch_params):
    (
        start_id,
        batch_size,
        geom_type,
        format_type,
        lon_min,
        lon_max,
        lat_min,
        lat_max,
        polygons,
        land_shape,
        labels,
        include_demographic,
        include_economic,
        spatial_clusters,
        correlation_engine,
        use_spatial_clustering,
        strict_land,
        date_start,
        date_end,
        seasonality,
        seed,
        batch_id,
    ) = batch_params
    if seed is not None:
        batch_seed = seed + batch_id
        random.seed(batch_seed)
        np.random.seed(batch_seed)
    points = generate_points_for_batch((
        batch_size, lon_min, lon_max, lat_min, lat_max,
        use_spatial_clustering, spatial_clusters, polygons, land_shape, strict_land
    ))
    geoms = generate_random_geom_batch((
        geom_type, format_type, batch_size, lon_min, lon_max, lat_min, lat_max, 
        points, batch_id, polygons, land_shape, strict_land
    ))
    ids = list(range(start_id, start_id + batch_size))
    dates = generate_random_datetimes(batch_size, date_start, date_end, seasonality)
    correlated_values = correlation_engine.generate_batch_values(labels, batch_size, seed=start_id)
    data = []
    for i in range(batch_size):
        row = {"id": ids[i], "geom": geoms[i], "date_created": dates[i]}
        if include_demographic:
            row["Gender"] = random.choice(GENDER_OPTIONS)
            row["Occupation"] = random.choice(OCCUPATION_OPTIONS)
            if "Education Level" in correlated_values:
                row["Education Level"] = correlated_values["Education Level"][i]
            else:
                row["Education Level"] = random.choice(EDUCATION_OPTIONS)
        if include_economic:
            if "Household Income" in correlated_values:
                row["Household Income"] = correlated_values["Household Income"][i]
            row["Employment Status"] = random.choice(EMPLOYMENT_STATUS_OPTIONS)
            row["Access to Healthcare"] = random.choice(HEALTHCARE_ACCESS_OPTIONS)
        for label in labels:
            if label not in row and label not in ["id", "geom", "date_created"]:
                if label in correlated_values:
                    row[label] = correlated_values[label][i]
                else:
                    min_val, max_val = VALUE_RANGES.get(label, (0, 100))
                    row[label] = round(correlation_engine.sample_value(min_val, max_val), 2)
        data.append(row)
    return data

def save_files_chunked(df, filename_prefix, chunk_size=100000):
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    total_rows = len(df)
    if total_rows <= chunk_size:
        csv_file = os.path.join(output_dir, f"{filename_prefix}.csv")
        df.to_csv(csv_file, index=False, encoding='utf-8')
        logger.info(f"Saved: {csv_file} ({os.path.getsize(csv_file) / 1024 / 1024:.2f} MB)")
        try:
            if total_rows <= 1000000:
                excel_file = os.path.join(output_dir, f"{filename_prefix}.xlsx")
                with pd.ExcelWriter(excel_file, engine='xlsxwriter') as writer:
                    df.to_excel(writer, index=False, sheet_name='Data')
                logger.info(f"Saved: {excel_file} ({os.path.getsize(excel_file) / 1024 / 1024:.2f} MB)")
            else:
                logger.warning("Dataset too large for Excel (> 1M rows). Excel file not created.")
        except Exception as e:
            logger.error(f"Error saving Excel file: {e}")
    else:
        logger.info(f"Dataset is large ({total_rows} rows). Saving in chunks...")
        num_chunks = math.ceil(total_rows / chunk_size)
        for i in range(num_chunks):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, total_rows)
            chunk = df.iloc[start_idx:end_idx]
            chunk_file = os.path.join(output_dir, f"{filename_prefix}_part{i+1}.csv")
            if i == 0:
                chunk.to_csv(chunk_file, index=False, encoding='utf-8')
            else:
                chunk.to_csv(chunk_file, index=False, encoding='utf-8', header=False)
            logger.info(f"Saved chunk {i+1}/{num_chunks}: {chunk_file} ({os.path.getsize(chunk_file) / 1024 / 1024:.2f} MB)")
        logger.info(f"CSV data saved in {num_chunks} chunks in the '{output_dir}' directory.")
        logger.warning("Excel file not created for chunked data (dataset too large).")

def apply_data_realism_options(
    df,
    noise_level=0.0,
    outlier_rate=0.0,
    outlier_scale=2.5,
    missing_rate=0.0,
):
    if noise_level > 0:
        for col in df.columns:
            if col in ["id", "geom", "date_created"]:
                continue
            if not pd.api.types.is_numeric_dtype(df[col]):
                continue
            if col in VALUE_RANGES:
                min_val, max_val = VALUE_RANGES[col]
                scale = (max_val - min_val) * noise_level
            else:
                scale = float(df[col].std() or 0)
            if scale > 0:
                df[col] = df[col] + np.random.normal(0, scale, len(df))
            if col in VALUE_RANGES:
                min_val, max_val = VALUE_RANGES[col]
                df[col] = df[col].clip(min_val, max_val)
    if outlier_rate > 0:
        outlier_rate = min(max(outlier_rate, 0.0), 1.0)
        for col in df.columns:
            if col in ["id", "geom", "date_created"]:
                continue
            if not pd.api.types.is_numeric_dtype(df[col]):
                continue
            count = int(len(df) * outlier_rate)
            if count <= 0:
                continue
            idx = np.random.choice(df.index, size=count, replace=False)
            if col in VALUE_RANGES:
                min_val, max_val = VALUE_RANGES[col]
                span = max_val - min_val
                direction = np.random.choice([-1.0, 1.0], size=count)
                df.loc[idx, col] = df.loc[idx, col] + direction * span * outlier_scale
                df.loc[idx, col] = df.loc[idx, col].clip(min_val, max_val)
            else:
                scale = float(df[col].std() or 0)
                if scale > 0:
                    df.loc[idx, col] = df.loc[idx, col] + np.random.normal(0, scale * outlier_scale, count)
    if missing_rate > 0:
        missing_rate = min(max(missing_rate, 0.0), 1.0)
        for col in df.columns:
            if col in ["id", "geom", "date_created"]:
                continue
            mask = np.random.random(len(df)) < missing_rate
            if pd.api.types.is_numeric_dtype(df[col]):
                df.loc[mask, col] = np.nan
            else:
                df.loc[mask, col] = None
    return df

def analyze_correlations(df, sample_size=10000):
    if len(df) > sample_size:
        logger.info(f"Sampling {sample_size:,} rows for correlation analysis")
        df = df.sample(sample_size)
    numeric_df = df.select_dtypes(include=['number'])
    if len(numeric_df.columns) < 2:
        logger.warning("Not enough numeric columns for correlation analysis.")
        return
    corr_matrix = numeric_df.corr()
    logger.info("\nCorrelation analysis for key variable pairs:")
    logger.info("-" * 50)
    logger.info(f"{'Variable Pair':<40} | {'Target':<8} | {'Actual':<8}")
    logger.info("-" * 50)
    for var1, var2, target_corr in VARIABLE_CORRELATIONS:
        if var1 in corr_matrix.columns and var2 in corr_matrix.columns:
            actual_corr = corr_matrix.loc[var1, var2]
            logger.info(f"{var1} vs {var2:<20} | {target_corr:8.2f} | {actual_corr:8.2f}")
    logger.info("-" * 50)

def generate_parallel_dataframe(
    rows,
    cols,
    geom_type,
    format_type,
    lon_min,
    lon_max,
    lat_min,
    lat_max,
    land_geometry=None,
    include_demographic=False,
    include_economic=False,
    use_spatial_clustering=False,
    cluster_count=5,
    points_per_cluster=200,
    strict_land=False,
    distribution_mode="uniform",
    noise_level=0.0,
    outlier_rate=0.0,
    outlier_scale=2.5,
    missing_rate=0.0,
    spatial_weighting="none",
    seed=None,
    date_start=None,
    date_end=None,
    seasonality="none",
):
    fixed_columns = ["id", "geom", "date_created"]
    if include_demographic:
        fixed_columns += ["Gender", "Occupation", "Education Level"]
    if include_economic:
        fixed_columns += ["Household Income", "Employment Status", "Access to Healthcare"]
    max_cols = len(REALISTIC_LABELS) + len(fixed_columns)
    if cols > max_cols:
        cols = max_cols
    num_additional = cols - len(fixed_columns)
    additional_labels = random.sample(REALISTIC_LABELS, min(num_additional, len(REALISTIC_LABELS))) if num_additional > 0 else []
    labels = fixed_columns + additional_labels
    polygons = []
    land_shape = None
    if land_geometry is not None:
        land_geom_obj = land_geometry
        if hasattr(land_geometry, "geometry"):
            land_geom_obj = unary_union(land_geometry["geometry"].values)
        polygons = extract_polygon_coords(land_geom_obj)
        land_shape = land_geom_obj
        logger.info(f"Extracted {len(polygons)} polygon(s) for efficient point-in-polygon testing")
    spatial_weighting = (spatial_weighting or "none").lower()
    if spatial_weighting == "urban_bias":
        use_spatial_clustering = True
        cluster_count = max(2, cluster_count // 2)
        points_per_cluster = max(points_per_cluster, 300)
    elif spatial_weighting == "rural_bias":
        use_spatial_clustering = True
        cluster_count = max(6, cluster_count * 2)
        points_per_cluster = max(50, points_per_cluster // 2)
    spatial_clusters = None
    if use_spatial_clustering:
        logger.info("Creating spatial clustering model...")
        spatial_clusters = create_spatial_clustering_model(
            lon_min, lon_max, lat_min, lat_max, land_geometry, 
            cluster_count, points_per_cluster, strict_land
        )
        logger.info(f"Created model with {len(spatial_clusters['centers'])} population clusters")
    system_cores = mp.cpu_count()
    recommended_cores = max(1, system_cores - 1)
    if rows < 10000:
        recommended_cores = max(1, min(2, recommended_cores))
    num_cores = recommended_cores
    logger.info(f"Using {num_cores} CPU cores for parallel processing")
    mem_per_core = psutil.virtual_memory().available / (num_cores * 1.5)
    est_row_size = 1000
    batch_size_by_mem = int(mem_per_core / est_row_size)
    batch_size = min(max(100, rows // (num_cores * 2)), batch_size_by_mem)
    if rows > 1000000:
        batch_size = min(50000, batch_size)
    logger.info(f"Using batch size of {batch_size:,} rows")
    distribution_mode = (distribution_mode or "uniform").lower()
    seasonality = (seasonality or "none").lower()
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    correlation_engine = CorrelationEngine(
        VARIABLE_CORRELATIONS,
        VALUE_RANGES,
        distribution_mode=distribution_mode,
    )
    parsed_start = parse_datetime(date_start)
    parsed_end = parse_datetime(date_end)
    num_batches = math.ceil(rows / batch_size)
    batch_params = [
        (i * batch_size + 1,
         min(batch_size, rows - i * batch_size),
         geom_type, 
         format_type, 
         lon_min, 
         lon_max, 
         lat_min, 
         lat_max,
         polygons,
         land_shape,
         labels, 
         include_demographic, 
         include_economic,
         spatial_clusters,
         correlation_engine,
        use_spatial_clustering,
        strict_land,
        parsed_start,
        parsed_end,
        seasonality,
        seed,
        i)
        for i in range(num_batches)
    ]
    all_data = []
    with tqdm(total=rows, desc="Generating data") as progress:
        try:
            with ProcessPoolExecutor(max_workers=num_cores) as executor:
                futures = [executor.submit(generate_batch_data, params) for params in batch_params]
                for future in as_completed(futures):
                    try:
                        batch_data = future.result()
                        all_data.extend(batch_data)
                        progress.update(len(batch_data))
                    except Exception as e:
                        logger.error(f"Error in batch processing: {e}")
        except PermissionError:
            logger.warning("Process pool not permitted in this environment; falling back to sequential generation.")
            for params in batch_params:
                batch_data = generate_batch_data(params)
                all_data.extend(batch_data)
                progress.update(len(batch_data))
    logger.info(f"Converting {len(all_data):,} rows to DataFrame")
    df = pd.DataFrame(all_data, columns=labels)
    for col in df.columns:
        if col in ["id"]:
            df[col] = df[col].astype('int32')
        elif col in ["Household Income", "Population"]:
            df[col] = df[col].astype('float32')
    df = apply_data_realism_options(
        df,
        noise_level=noise_level,
        outlier_rate=outlier_rate,
        outlier_scale=outlier_scale,
        missing_rate=missing_rate,
    )
    logger.info("Analyzing correlations in generated data...")
    analyze_correlations(df)
    return df

# --- DATA VALIDATOR CLASS ---

class DataValidator:
    def __init__(self, validation_options=None):
        self.validation_options = validation_options or {
            'correlation_tolerance': 0.15,
            'outlier_threshold': 3,
            'min_valid_geometries': 0.99,
            'repair_geometries': True,
            'generate_reports': True,
            'report_dir': 'validation_reports'
        }
        if self.validation_options['generate_reports']:
            os.makedirs(self.validation_options['report_dir'], exist_ok=True)
    def validate_statistical_properties(self, df, expected_correlations, value_ranges):
        validation_results = {
            'passed': True,
            'correlation_issues': [],
            'distribution_issues': [],
            'range_issues': [],
            'outliers': {}
        }
        numeric_df = df.select_dtypes(include=['number'])
        if len(numeric_df.columns) >= 2:
            corr_matrix = numeric_df.corr()
            for var1, var2, expected_corr in expected_correlations:
                if var1 in corr_matrix.columns and var2 in corr_matrix.columns:
                    actual_corr = corr_matrix.loc[var1, var2]
                    diff = abs(actual_corr - expected_corr)
                    if diff > self.validation_options['correlation_tolerance']:
                        validation_results['passed'] = False
                        validation_results['correlation_issues'].append({
                            'variables': (var1, var2),
                            'expected': expected_corr,
                            'actual': actual_corr,
                            'difference': diff
                        })
        for col in df.columns:
            if col in value_ranges and pd.api.types.is_numeric_dtype(df[col]):
                min_val, max_val = value_ranges[col]
                actual_min, actual_max = df[col].min(), df[col].max()
                if actual_min < min_val or actual_max > max_val:
                    validation_results['passed'] = False
                    validation_results['range_issues'].append({
                        'variable': col,
                        'expected_range': (min_val, max_val),
                        'actual_range': (actual_min, actual_max)
                    })
                z_scores = stats.zscore(df[col])
                outliers = np.abs(z_scores) > self.validation_options['outlier_threshold']
                if outliers.any():
                    outlier_count = outliers.sum()
                    outlier_percentage = (outlier_count / len(df)) * 100
                    validation_results['outliers'][col] = {
                        'count': int(outlier_count),
                        'percentage': float(outlier_percentage),
                        'indices': df.index[outliers].tolist()[:100]
                    }
        if self.validation_options['generate_reports']:
            self._generate_statistical_report(df, validation_results, expected_correlations)
        return validation_results
    def validate_geometries(self, df, geom_column='geom', format_type='WKT'):
        import shapely
        from shapely.validation import explain_validity
        validation_results = {
            'passed': True,
            'total_geometries': len(df),
            'valid_count': 0,
            'invalid_count': 0,
            'repaired_count': 0,
            'invalid_samples': [],
            'bounds_issues': [],
        }
        bounds = None
        if hasattr(self, 'lon_min') and hasattr(self, 'lon_max') and hasattr(self, 'lat_min') and hasattr(self, 'lat_max'):
            bounds = (self.lon_min, self.lat_min, self.lon_max, self.lat_max)
        for i, geom_str in enumerate(df[geom_column]):
            try:
                if format_type == 'WKT':
                    geom = shapely.wkt.loads(geom_str)
                else:
                    geom = shape(json.loads(geom_str))
                if geom.is_valid:
                    validation_results['valid_count'] += 1
                else:
                    reason = explain_validity(geom)
                    validation_results['invalid_count'] += 1
                    if len(validation_results['invalid_samples']) < 10:
                        validation_results['invalid_samples'].append({
                            'index': i,
                            'geometry': geom_str,
                            'reason': reason
                        })
                    if self.validation_options['repair_geometries']:
                        try:
                            repaired = geom.buffer(0)
                            if repaired.is_valid:
                                validation_results['repaired_count'] += 1
                                if format_type == 'WKT':
                                    df.at[i, geom_column] = repaired.wkt
                                else:
                                    df.at[i, geom_column] = json.dumps(shapely.geometry.mapping(repaired))
                        except Exception as e:
                            logger.warning(f"Failed to repair geometry at index {i}: {e}")
                if bounds and not geom.is_empty:
                    geom_bounds = geom.bounds
                    if (geom_bounds[0] < bounds[0] or geom_bounds[2] > bounds[2] or 
                        geom_bounds[1] < bounds[1] or geom_bounds[3] > bounds[3]):
                        validation_results['bounds_issues'].append({
                            'index': i,
                            'geometry_bounds': geom_bounds,
                            'expected_bounds': bounds
                        })
            except Exception as e:
                validation_results['invalid_count'] += 1
                if len(validation_results['invalid_samples']) < 10:
                    validation_results['invalid_samples'].append({
                        'index': i,
                        'geometry': geom_str,
                        'error': str(e)
                    })
        validity_percentage = (validation_results['valid_count'] / 
                               validation_results['total_geometries']) if validation_results['total_geometries'] > 0 else 0
        validation_results['validity_percentage'] = validity_percentage * 100
        validation_results['passed'] = (validity_percentage >= 
                                       self.validation_options['min_valid_geometries'])
        if self.validation_options['generate_reports']:
            self._generate_geometry_report(df, validation_results, format_type)
        return validation_results
    def _generate_statistical_report(self, df, validation_results, expected_correlations):
        report_time = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_path = os.path.join(self.validation_options['report_dir'], f'stat_report_{report_time}.pdf')
        plt.figure(figsize=(15, 12))
        numeric_df = df.select_dtypes(include=['number'])
        if len(numeric_df.columns) >= 2:
            plt.subplot(2, 2, 1)
            corr_matrix = numeric_df.corr()
            plt.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
            plt.colorbar()
            plt.title('Correlation Matrix')
            plt.xticks(range(len(corr_matrix.columns)), corr_matrix.columns, rotation=90)
            plt.yticks(range(len(corr_matrix.columns)), corr_matrix.columns)
            for issue in validation_results['correlation_issues']:
                var1, var2 = issue['variables']
                if var1 in corr_matrix.columns and var2 in corr_matrix.columns:
                    i = list(corr_matrix.columns).index(var1)
                    j = list(corr_matrix.columns).index(var2)
                    plt.plot(j, i, 'rx', markersize=10)
        key_vars = ['Population', 'Income per Capita', 'Education Level', 'Health Index']
        key_vars = [var for var in key_vars if var in df.columns]
        for i, var in enumerate(key_vars[:3]):
            plt.subplot(2, 2, i+2)
            if pd.api.types.is_numeric_dtype(df[var]):
                df[var].hist(bins=30)
                plt.title(f'Distribution of {var}')
                plt.xlabel(var)
                plt.ylabel('Frequency')
                plt.axvline(df[var].mean(), color='r', linestyle='--', label=f'Mean: {df[var].mean():.2f}')
                plt.axvline(df[var].median(), color='g', linestyle='-', label=f'Median: {df[var].median():.2f}')
                plt.legend()
        plt.tight_layout()
        plt.savefig(report_path)
        plt.close()
        logger.info(f"Statistical validation report saved to {report_path}")
    def _generate_geometry_report(self, df, validation_results, format_type):
        import json
        report_time = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_path = os.path.join(self.validation_options['report_dir'], 
                                   f'geom_report_{report_time}.json')
        report = {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_geometries': validation_results['total_geometries'],
                'valid_percentage': validation_results['validity_percentage'],
                'invalid_count': validation_results['invalid_count'],
                'repaired_count': validation_results['repaired_count'],
                'passed_validation': validation_results['passed']
            },
            'invalid_samples': validation_results['invalid_samples'][:10],
            'bounds_issues': validation_results['bounds_issues'][:10]
        }
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        logger.info(f"Geometry validation report saved to {report_path}")
    def validate_dataset(self, df, expected_correlations, value_ranges, geom_column='geom', format_type='WKT'):
        sample_size = 100000
        validate_df = df if len(df) <= sample_size else df.sample(sample_size)
        logger.info(f"Starting dataset validation on {len(validate_df):,} rows")
        logger.info("Validating statistical properties...")
        stat_results = self.validate_statistical_properties(
            validate_df, expected_correlations, value_ranges)
        logger.info("Validating geometries...")
        geom_results = self.validate_geometries(validate_df, geom_column, format_type)
        results = {
            'passed': stat_results['passed'] and geom_results['passed'],
            'statistical_validation': stat_results,
            'geometry_validation': geom_results,
            'timestamp': datetime.now().isoformat()
        }
        logger.info("\n=== Validation Summary ===")
        logger.info(f"Overall validation: {'PASSED' if results['passed'] else 'FAILED'}")
        logger.info(f"Statistical validation: {'PASSED' if stat_results['passed'] else 'FAILED'}")
        logger.info(f"- Correlation issues: {len(stat_results['correlation_issues'])}")
        logger.info(f"- Distribution issues: {len(stat_results['distribution_issues'])}")
        logger.info(f"- Range issues: {len(stat_results['range_issues'])}")
        logger.info(f"- Variables with outliers: {len(stat_results['outliers'])}")
        logger.info(f"Geometry validation: {'PASSED' if geom_results['passed'] else 'FAILED'}")
        logger.info(f"- Valid geometries: {geom_results['valid_count']:,} ({geom_results['validity_percentage']:.2f}%)")
        logger.info(f"- Invalid geometries: {geom_results['invalid_count']:,}")
        logger.info(f"- Repaired geometries: {geom_results['repaired_count']:,}")
        logger.info(f"- Bounds issues: {len(geom_results['bounds_issues'])}")
        logger.info("==========================")
        if self.validation_options['generate_reports']:
            report_path = os.path.join(
                self.validation_options['report_dir'], 
                f'validation_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
            )
            with open(report_path, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Comprehensive validation report saved to {report_path}")
        return results

def validate_generated_data(df, geom_type, format_type, lon_min, lon_max, lat_min, lat_max):
    logger.info("Starting data validation...")
    validator = DataValidator({
        'correlation_tolerance': 0.45,
        'outlier_threshold': 3.0,
        'min_valid_geometries': 0.99,
        'repair_geometries': True,
        'generate_reports': True,
        'report_dir': 'validation_reports'
    })
    validator.lon_min = lon_min
    validator.lon_max = lon_max
    validator.lat_min = lat_min
    validator.lat_max = lat_max
    results = validator.validate_dataset(
        df, 
        VARIABLE_CORRELATIONS,
        VALUE_RANGES,
        'geom',
        'WKT' if format_type == "WKT" else 'GeoJSON'
    )
    if results['geometry_validation']['repaired_count'] > 0:
        logger.info(f"Repaired {results['geometry_validation']['repaired_count']} geometries")
        return df, results
    return df, results

# --- MAIN ENTRYPOINT ---

if __name__ == "__main__":
    logger.info("Dataset Generator with Validation")
    logger.info("==============================================")
    logger.info(f"Last update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("Author: zhafran-bvt")
    logger.info("==============================================")
    parser = argparse.ArgumentParser(description="Dataset Generator")
    parser.add_argument('--config', help="Path to JSON config file")
    args = parser.parse_args()
    if args.config:
        config = load_config(args.config)
        format_choice = config['format_choice']
        include_demographic = config['include_demographic'].lower() == 'yes'
        include_economic = config['include_economic'].lower() == 'yes'
        num_columns = config['num_columns']
        use_spatial_clustering = config['use_spatial_clustering'].lower() == 'yes'
        area_choice = config['area_choice']
        geojson_path = config.get('geojson_path', '')
        strict_land = config.get('strict_land', False)
        if isinstance(strict_land, str):
            strict_land = strict_land.lower() == 'yes'
        else:
            strict_land = bool(strict_land)
        distribution_mode = config.get('distribution_mode', 'uniform')
        noise_level = float(config.get('noise_level', 0.0))
        outlier_rate = float(config.get('outlier_rate', 0.0))
        outlier_scale = float(config.get('outlier_scale', 2.5))
        missing_rate = float(config.get('missing_rate', 0.0))
        spatial_weighting = config.get('spatial_weighting', 'none')
        seed = config.get('seed', None)
        if seed is not None:
            seed = int(seed)
        date_start = config.get('date_start', None)
        date_end = config.get('date_end', None)
        seasonality = config.get('seasonality', 'none')
        num_rows = config['num_rows']
        use_chunking = config['use_chunking'].lower() == 'yes'
        geometry_type = config['geometry_type']
    else:
        format_choice = validate_input("Choose format: 1 for WKT, 2 for GeoJSON\nEnter choice: ", [1, 2], int)
        include_demographic = validate_input("Include demographic columns (Gender, Occupation, Education Level)? (yes/no): ", ['yes', 'no']).lower() == 'yes'
        include_economic = validate_input("Include economic columns (Household Income, Employment Status, Access to Healthcare)? (yes/no): ", ['yes', 'no']).lower() == 'yes'
        min_cols = 3
        if include_demographic:
            min_cols += 3
        if include_economic:
            min_cols += 3
        max_cols = min_cols + len(REALISTIC_LABELS)
        num_columns = validate_input(f"Enter number of columns (min {min_cols}, max {max_cols}): ", range(min_cols, max_cols + 1), int)
        use_spatial_clustering = validate_input("Use spatial clustering to create realistic population distributions? (yes/no): ", ['yes', 'no']).lower() == 'yes'
        area_choice = validate_input("Choose area: 1 for Jakarta, 2 for Yogyakarta, 3 for Indonesia, 4 for Japan, 5 for Vietnam\nEnter choice: ", [1, 2, 3, 4, 5], int)
        geojson_path = ""
        if area_choice in [1, 3, 4, 5]:
            geojson_path = validate_input(f"Enter path to GeoJSON file for {BOUNDING_BOXES[area_choice]['name']} land boundaries (e.g., geojson/id-jk.min.geojson): ", None, str, allow_empty=True)
        strict_land = False
        if geojson_path:
            strict_land = validate_input("Strictly constrain to land only (no ocean fallback)? (yes/no): ", ['yes', 'no']).lower() == 'yes'
        distribution_mode = "uniform"
        noise_level = 0.0
        outlier_rate = 0.0
        outlier_scale = 2.5
        missing_rate = 0.0
        spatial_weighting = "none"
        seed = None
        date_start = None
        date_end = None
        seasonality = "none"
        num_rows = validate_input("Enter number of rows: ", None, int)
        chunking_threshold = 100000
        use_chunking = True
        if num_rows > chunking_threshold:
            use_chunking = validate_input(f"Large dataset detected ({num_rows} rows). Use chunked file output? (yes/no, default: yes): ", ['yes', 'no', ''], str).lower() != 'no'
        geometry_type = validate_input("Choose geometry type: 1 for POINT, 2 for POLYGON, 3 for MULTIPOLYGON\nEnter choice: ", [1, 2, 3], int)
    format_type = "WKT" if format_choice == 1 else "GeoJSON"
    area = BOUNDING_BOXES[area_choice]
    lon_min, lon_max = area["lon_min"], area["lon_max"]
    lat_min, lat_max = area["lat_min"], area["lat_max"]
    area_name = area["name"]
    geom_type = "POINT" if geometry_type == 1 else "POLYGON" if geometry_type == 2 else "MULTIPOLYGON"
    start_time = datetime.now()
    logger.info(f"Generating {num_rows:,} rows of {geom_type} geometries for {area_name}...")
    land_geometry = None
    if geojson_path:
        land_geometry = gpd.read_file(geojson_path)
        if land_geometry.empty:
            logger.error("Failed to load land geometry. Exiting.")
            exit(1)
        land_geometry = unary_union(land_geometry['geometry'].values)
    df = generate_parallel_dataframe(
        num_rows, num_columns, geom_type, format_type, lon_min, lon_max, lat_min, lat_max, 
        land_geometry, include_demographic, include_economic, use_spatial_clustering,
        strict_land=strict_land,
        distribution_mode=distribution_mode,
        noise_level=noise_level,
        outlier_rate=outlier_rate,
        outlier_scale=outlier_scale,
        missing_rate=missing_rate,
        spatial_weighting=spatial_weighting,
        seed=seed,
        date_start=date_start,
        date_end=date_end,
        seasonality=seasonality,
    )
    df, validation_results = validate_generated_data(
        df, geom_type, format_type, lon_min, lon_max, lat_min, lat_max
    )
    filename_prefix = f"{area_name.lower()}_data_{num_rows}r_{num_columns}c_{geom_type.lower()}_{format_type.lower()}"
    if land_geometry is not None:
        filename_prefix += "_land"
    if not validation_results['passed']:
        filename_prefix += "_needs_review"
    end_time = datetime.now()
    elapsed_time = end_time - start_time
    logger.info(f"Data generation completed in {elapsed_time.total_seconds():.2f} seconds")
    if use_chunking:
        save_files_chunked(df, filename_prefix)
    else:
        output_dir = "output"
        os.makedirs(output_dir, exist_ok=True)
        csv_file = os.path.join(output_dir, f"{filename_prefix}.csv")
        excel_file = os.path.join(output_dir, f"{filename_prefix}.xlsx")
        df.to_csv(csv_file, index=False, encoding='utf-8')
        logger.info(f"Saved: {csv_file} ({os.path.getsize(csv_file) / 1024 / 1024:.2f} MB)")
        try:
            if num_rows <= 1000000:
                with pd.ExcelWriter(excel_file, engine='xlsxwriter') as writer:
                    df.to_excel(writer, index=False, sheet_name='Data')
                logger.info(f"Saved: {excel_file} ({os.path.getsize(excel_file) / 1024 / 1024:.2f} MB)")
            else:
                logger.warning("Dataset too large for Excel (> 1M rows). Excel file not created.")
        except Exception as e:
            logger.error(f"Error saving Excel file: {e}")
    logger.info("Process completed successfully.")
