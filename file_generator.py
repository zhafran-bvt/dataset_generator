import random
import pandas as pd
import json
import os
import multiprocessing as mp
import numpy as np
from shapely.geometry import shape, Point, Polygon, MultiPolygon
from shapely.ops import unary_union
from shapely.wkt import loads
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

# H3 library import with availability check
try:
    import h3
    H3_AVAILABLE = True
except ImportError:
    H3_AVAILABLE = False
    h3 = None

# --- CONFIGURATION ---

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore')

# H3 configuration constants
H3_RESOLUTION = 9  # Fixed resolution level

def generate_h3_cells_batch(points, resolution=None, lon_min=None, lon_max=None, lat_min=None, lat_max=None, land_geometry=None):
    """
    Convert point arrays to H3 cells with optional filtering.
    
    Args:
        points: NumPy array of shape (n, 2) containing [longitude, latitude] coordinates
        resolution: H3 resolution level (ignored, always uses H3_RESOLUTION = 9)
        lon_min, lon_max, lat_min, lat_max: Optional bounding box for filtering cell centers
        land_geometry: Optional Shapely geometry for filtering cell centers within land boundaries
    
    Returns:
        List of H3 cell identifiers as hexadecimal strings
    """
    # Always use fixed resolution 9
    resolution = H3_RESOLUTION
    if not H3_AVAILABLE:
        logger.error("H3 library is not available. Cannot generate H3 cells.")
        return []
    
    # Extract land polygons if provided
    land_polys = []
    if land_geometry:
        land_polys = extract_polygon_coords(land_geometry)
    
    h3_cells = []
    for i in range(len(points)):
        lon, lat = points[i]
        try:
            # h3 v4 API uses latlng_to_cell (lat, lng order)
            h3_cell = h3.latlng_to_cell(lat, lon, resolution)
            
            # Get cell center for filtering
            center_lat, center_lon = h3.cell_to_latlng(h3_cell)
            
            # Filter by bounding box if specified
            if lon_min is not None and lon_max is not None and lat_min is not None and lat_max is not None:
                if center_lon < lon_min or center_lon > lon_max or center_lat < lat_min or center_lat > lat_max:
                    logger.debug(f"Filtered H3 cell {h3_cell} with center ({center_lon:.6f}, {center_lat:.6f}) outside bounding box")
                    continue
            
            # Filter by land boundaries if specified
            if land_polys:
                point_in_land = False
                for poly in land_polys:
                    if is_point_in_polygon(poly, (center_lon, center_lat)):
                        point_in_land = True
                        break
                
                if not point_in_land:
                    logger.debug(f"Filtered H3 cell {h3_cell} with center ({center_lon:.6f}, {center_lat:.6f}) outside land boundaries")
                    continue
            
            h3_cells.append(h3_cell)
        except Exception as e:
            logger.warning(f"Failed to convert point ({lon:.6f}, {lat:.6f}) to H3 cell: {e}")
            continue
    
    return h3_cells

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
    "Water Access", "Sanitation Access"
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
    "Household Income": (0, 1000000)
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
    ("Population", "Housing Density", 0.6),
    ("Housing Density", "Public Transport Usage", 0.5),
    ("Employment Rate", "Unemployment Rate", -0.9),
    ("GDP Growth", "Employment Rate", 0.6),
    ("GDP Growth", "Unemployment Rate", -0.5)
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
    def __init__(self, correlations, value_ranges):
        self.correlations = correlations
        self.value_ranges = value_ranges
    def generate_batch_values(self, labels, batch_size=10000, seed=None):
        if seed is not None:
            np.random.seed(seed)
        values = {label: np.random.random(batch_size) for label in labels if label in self.value_ranges}
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

def generate_random_datetimes(n):
    start_ts = datetime(2025, 1, 1, 0, 0, 0).timestamp()
    end_ts = datetime(2025, 12, 31, 23, 59, 59).timestamp()
    random_ts = np.random.uniform(start_ts, end_ts, n)
    dates = []
    for ts in random_ts:
        dt = datetime.fromtimestamp(ts)
        dates.append(dt.strftime("%Y-%m-%dT%H:%M:%SZ"))
    return dates

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

def random_sample_in_geometry(n, polygons, lon_min, lon_max, lat_min, lat_max, batch_size=1000):
    # Strictly sample points inside provided polygons; never fall back to bbox.
    # If no polygons provided, fall back to uniform bbox sampling.
    if not polygons:
        return get_random_points_in_bbox(lon_min, lon_max, lat_min, lat_max, n)

    points = []
    attempts = 0
    # Increase attempts generously to avoid premature fallback for coastal/fragmented shapes
    max_attempts = max(n * 100, 5000)
    while len(points) < n and attempts < max_attempts:
        # Sample in larger chunks to improve hit rate
        want = n - len(points)
        candidates = get_random_points_in_bbox(
            lon_min, lon_max, lat_min, lat_max, min(max(batch_size, want * 5), want * 20)
        )
        for point in candidates:
            if len(points) >= n:
                break
            for poly in polygons:
                if is_point_in_polygon(poly, point):
                    points.append(point)
                    break
        attempts += len(candidates)
    # Return only the valid, in-geometry points collected (may be < n in pathological cases)
    return np.array(points[:n])

def generate_random_geom_batch(params):
    geom_type, format_type, n, lon_min, lon_max, lat_min, lat_max, points, batch_id, land_geometry, h3_resolution = params
    geoms = []
    if geom_type == "H3":
        # Generate H3 cells from points with filtering (always uses resolution 9)
        h3_cells = generate_h3_cells_batch(
            points, 
            lon_min=lon_min, lon_max=lon_max, 
            lat_min=lat_min, lat_max=lat_max,
            land_geometry=land_geometry
        )
        return h3_cells
    elif geom_type == "POINT":
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
        existing_geoms = []
        tree = STRtree(existing_geoms) if existing_geoms else None
        land_polys = extract_polygon_coords(land_geometry) if land_geometry else []
        for i in range(n):
            max_attempts = 250
            last_valid_coords = None
            for attempt in range(max_attempts):
                lon, lat = points[i]
                logger.debug(f"Attempt {attempt} for index {i}: lon={lon:.6f}, lat={lat:.6f}")
                # Size range with progressive reduction and minimum threshold
                size_factor = max(0.0025, 0.025 * (1.0 / (attempt // 20 + 1))) if attempt >= 20 else random.uniform(0.005, 0.05)
                width = lon_extent * size_factor
                height = lat_extent * size_factor
                # Find the containing land polygon
                containing_poly = None
                for poly in land_polys:
                    if Point(lon, lat).within(Polygon(poly)):
                        containing_poly = Polygon(poly)
                        break
                if not containing_poly:
                    continue
                # Generate initial coordinates
                coords = [
                    [lon - width/2, lat - height/2],
                    [lon + width/2, lat - height/2],
                    [lon + width/2, lat + height/2],
                    [lon - width/2, lat + height/2],
                    [lon - width/2, lat - height/2]
                ]
                initial_poly = Polygon(coords)
                # Clip to the containing land polygon
                clipped_geom = initial_poly.intersection(containing_poly)
                if not clipped_geom.is_valid or clipped_geom.is_empty:
                    adjustment_factor = min(0.1, 0.01 * (attempt // 50 + 1)) * lon_extent
                    adjustment = random.uniform(0.001, adjustment_factor)
                    lon += adjustment if random.choice([True, False]) else -adjustment
                    lat += adjustment if random.choice([True, False]) else -adjustment
                    continue
                # Handle MultiPolygon case
                if isinstance(clipped_geom, MultiPolygon):
                    max_area = 0
                    best_poly = None
                    for poly in clipped_geom.geoms:
                        if poly.area > max_area:
                            max_area = poly.area
                            best_poly = poly
                    if best_poly:
                        clipped_poly = best_poly
                    else:
                        adjustment_factor = min(0.1, 0.01 * (attempt // 50 + 1)) * lon_extent
                        adjustment = random.uniform(0.001, adjustment_factor)
                        lon += adjustment if random.choice([True, False]) else -adjustment
                        lat += adjustment if random.choice([True, False]) else -adjustment
                        continue
                else:
                    clipped_poly = clipped_geom
                # Validate that all coordinates are within land
                all_within = True
                coords_to_check = list(clipped_poly.exterior.coords)
                if coords_to_check[0] != coords_to_check[-1]:
                    coords_to_check.append(coords_to_check[0])
                for x, y in coords_to_check[:-1]:
                    if not any(Point(x, y).within(Polygon(poly)) for poly in land_polys):
                        all_within = False
                        break
                if not all_within:
                    adjustment_factor = min(0.1, 0.01 * (attempt // 50 + 1)) * lon_extent
                    adjustment = random.uniform(0.001, adjustment_factor)
                    lon += adjustment if random.choice([True, False]) else -adjustment
                    lat += adjustment if random.choice([True, False]) else -adjustment
                    continue
                # Check for overlap with existing geometries
                if tree:
                    new_multi = MultiPolygon([clipped_poly])
                    if any(new_multi.intersects(loads(g) if format_type == "WKT" else shape(json.loads(g))) for g in geoms):
                        adjustment_factor = min(0.1, 0.01 * (attempt // 50 + 1)) * lon_extent
                        adjustment = random.uniform(0.001, adjustment_factor)
                        lon += adjustment if random.choice([True, False]) else -adjustment
                        lat += adjustment if random.choice([True, False]) else -adjustment
                        continue
                # Ensure closed coordinates for output
                last_valid_coords = list(clipped_poly.exterior.coords)
                if last_valid_coords[0] != last_valid_coords[-1]:
                    last_valid_coords.append(last_valid_coords[0])
                break
            if last_valid_coords is None:
                logger.error(f"Failed to generate non-overlapping {geom_type} at index {i} after {max_attempts} attempts (including fallback). Skipping index {i}.")
                continue  # Skip to the next index instead of raising an exception
            if format_type == "WKT":
                coord_str = ", ".join([f"{x:.6f} {y:.6f}" for x, y in last_valid_coords])
                if geom_type == "POLYGON":
                    geoms.append(f"POLYGON (({coord_str}))")
                else:  # MULTIPOLYGON
                    geoms.append(f"MULTIPOLYGON ((({coord_str})))")
            else:
                if geom_type == "POLYGON":
                    geoms.append(json.dumps({"type": "Polygon", "coordinates": [last_valid_coords]}))
                else:  # MULTIPOLYGON
                    geoms.append(json.dumps({"type": "MultiPolygon", "coordinates": [[last_valid_coords]]}))
            existing_geoms.append(Polygon(last_valid_coords))
            tree = STRtree(existing_geoms)
    return geoms

def create_spatial_clustering_model(lon_min, lon_max, lat_min, lat_max, land_geometry, cluster_count=5, points_per_cluster=200):
    if land_geometry:
        polygons = extract_polygon_coords(land_geometry)
        centers = random_sample_in_geometry(cluster_count, polygons, lon_min, lon_max, lat_min, lat_max)
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
    batch_size, lon_min, lon_max, lat_min, lat_max, use_spatial_clustering, spatial_clusters, polygons = params
    if use_spatial_clustering and spatial_clusters and random.random() < spatial_clusters['use_probability']:
        samples = spatial_clusters['model'].resample(batch_size)
        points = np.column_stack((samples[0], samples[1]))
        points[:, 0] = np.clip(points[:, 0], lon_min, lon_max)
        points[:, 1] = np.clip(points[:, 1], lat_min, lat_max)
        if polygons:
            for i in range(len(points)):
                is_valid = False
                for poly in polygons:
                    if is_point_in_polygon(poly, points[i]):
                        is_valid = True
                        break
                if not is_valid:
                    # Attempt to replace with a valid in-geometry point
                    replaced = False
                    for _ in range(10):
                        valid_points = random_sample_in_geometry(1, polygons, lon_min, lon_max, lat_min, lat_max)
                        if valid_points.shape[0] >= 1:
                            points[i] = valid_points[0]
                            replaced = True
                            break
                    if not replaced:
                        # Try a small batch replacement to increase odds
                        valid_points = random_sample_in_geometry(50, polygons, lon_min, lon_max, lat_min, lat_max)
                        if valid_points.shape[0] >= 1:
                            points[i] = valid_points[0]
    elif polygons:
        points = random_sample_in_geometry(batch_size, polygons, lon_min, lon_max, lat_min, lat_max)
    else:
        points = random_sample_uniform(batch_size, lon_min, lon_max, lat_min, lat_max)
    return points

def generate_batch_data(batch_params):
    (start_id, batch_size, geom_type, format_type, lon_min, lon_max, lat_min, lat_max, 
     polygons, labels, include_demographic, include_economic, spatial_clusters, 
     correlation_engine, use_spatial_clustering, batch_id, land_geometry, h3_resolution) = batch_params
    points = generate_points_for_batch((
        batch_size, lon_min, lon_max, lat_min, lat_max,
        use_spatial_clustering, spatial_clusters, polygons
    ))
    try:
        geoms = generate_random_geom_batch((
            geom_type, format_type, batch_size, lon_min, lon_max, lat_min, lat_max, 
            points, batch_id, land_geometry, h3_resolution
        ))
    except RuntimeError as e:
        logger.error(f"Geometry generation failed: {e}. Using available geometries.")
        geoms = []  # Handle partial failure by using any generated geometries
    ids = list(range(start_id, start_id + batch_size))
    dates = generate_random_datetimes(batch_size)
    correlated_values = correlation_engine.generate_batch_values(labels, batch_size, seed=start_id)
    
    # Generate building count values for all rows (INT column)
    np.random.seed(start_id)  # Use start_id as seed for reproducibility within batch
    building_counts = np.random.randint(1, 500, size=batch_size)
    
    data = []
    for i in range(batch_size):
        if i < len(geoms):  # Only include rows where geometries were successfully generated
            row = {"id": ids[i], "geom": geoms[i], "date_created": dates[i]}
            
            # Add latitude and longitude for H3 geometry type
            if geom_type == "H3" and H3_AVAILABLE:
                try:
                    lat, lon = h3.cell_to_latlng(geoms[i])
                    row["latitude"] = lat
                    row["longitude"] = lon
                except Exception as e:
                    logger.warning(f"Failed to extract coordinates from H3 cell {geoms[i]}: {e}")
                    row["latitude"] = None
                    row["longitude"] = None
            
            # Add building count (INT column)
            row["building_count"] = int(building_counts[i])
            
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
                if label not in row and label not in ["id", "geom", "date_created", "latitude", "longitude", "building_count"]:
                    if label in correlated_values:
                        row[label] = correlated_values[label][i]
                    else:
                        min_val, max_val = VALUE_RANGES.get(label, (0, 100))
                        row[label] = round(random.uniform(min_val, max_val), 2)
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

def generate_parallel_dataframe(rows, cols, geom_type, format_type, lon_min, lon_max, lat_min, lat_max, 
                              land_geometry=None, include_demographic=False, include_economic=False, 
                              use_spatial_clustering=False, cluster_count=5, points_per_cluster=200, h3_resolution=None):
    fixed_columns = ["id", "geom", "date_created"]
    if include_demographic:
        fixed_columns += ["Gender", "Occupation", "Education Level"]
    if include_economic:
        fixed_columns += ["Household Income", "Employment Status", "Access to Healthcare"]
    
    # Filter out any REALISTIC_LABELS that are already in fixed_columns to prevent duplicates
    available_labels = [label for label in REALISTIC_LABELS if label not in fixed_columns]
    
    max_cols = len(available_labels) + len(fixed_columns)
    if cols > max_cols:
        cols = max_cols
    num_additional = cols - len(fixed_columns)
    additional_labels = random.sample(available_labels, min(num_additional, len(available_labels))) if num_additional > 0 else []
    labels = fixed_columns + additional_labels
    polygons = []
    if land_geometry:
        polygons = extract_polygon_coords(land_geometry)
        logger.info(f"Extracted {len(polygons)} polygon(s) for efficient point-in-polygon testing")
    spatial_clusters = None
    if use_spatial_clustering:
        logger.info("Creating spatial clustering model...")
        spatial_clusters = create_spatial_clustering_model(
            lon_min, lon_max, lat_min, lat_max, land_geometry, 
            cluster_count, points_per_cluster
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
    correlation_engine = CorrelationEngine(VARIABLE_CORRELATIONS, VALUE_RANGES)
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
         labels, 
         include_demographic, 
         include_economic,
         spatial_clusters,
         correlation_engine,
         use_spatial_clustering,
         i,
         land_geometry,
         h3_resolution)
        for i in range(num_batches)
    ]
    all_data = []
    with tqdm(total=rows, desc="Generating data") as progress:
        with ProcessPoolExecutor(max_workers=num_cores) as executor:
            futures = [executor.submit(generate_batch_data, params) for params in batch_params]
            for future in as_completed(futures):
                try:
                    batch_data = future.result()
                    all_data.extend(batch_data)
                    progress.update(len(batch_data))
                except Exception as e:
                    logger.error(f"Error in batch processing: {e}")
                    # Continue with available data even if an error occurs
    logger.info(f"Converting {len(all_data):,} rows to DataFrame")
    # Don't restrict columns - let DataFrame include all columns from the data
    df = pd.DataFrame(all_data)
    
    # Reorder columns to have a logical order
    # Start with fixed columns, then H3-specific columns if present, then the rest
    fixed_cols = ["id", "geom"]
    if "latitude" in df.columns and "longitude" in df.columns:
        fixed_cols.extend(["latitude", "longitude"])
    fixed_cols.append("date_created")
    if "building_count" in df.columns:
        fixed_cols.append("building_count")
    
    # Get remaining columns in their original order
    remaining_cols = [col for col in df.columns if col not in fixed_cols]
    df = df[fixed_cols + remaining_cols]
    
    for col in df.columns:
        if col in ["id", "building_count"]:
            df[col] = df[col].astype('int32')
        elif col in ["Household Income", "Population"]:
            df[col] = df[col].astype('float32')
        elif col in ["latitude", "longitude"]:
            df[col] = df[col].astype('float64')
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
    def validate_h3_cells(self, df, h3_column='geom', resolution=None, 
                          lon_min=None, lon_max=None, lat_min=None, lat_max=None,
                          land_geometry=None):
        """
        Validate H3 cell identifiers in the dataset.
        
        Args:
            df: DataFrame containing H3 cells
            h3_column: Column name containing H3 cell identifiers
            resolution: Expected H3 resolution (optional)
            lon_min, lon_max, lat_min, lat_max: Bounding box coordinates (optional)
            land_geometry: Shapely geometry for land boundaries (optional)
        
        Returns:
            dict: Validation results including:
                - passed: bool
                - total_cells: int
                - valid_count: int
                - invalid_count: int
                - resolution_mismatches: int
                - out_of_bounds_count: int
                - invalid_samples: list
        """
        if not H3_AVAILABLE:
            logger.error("H3 library is not available. Cannot validate H3 cells.")
            return {
                'passed': False,
                'total_cells': 0,
                'valid_count': 0,
                'invalid_count': 0,
                'resolution_mismatches': 0,
                'out_of_bounds_count': 0,
                'invalid_samples': [],
                'error': 'H3 library not available'
            }
        
        validation_results = {
            'passed': True,
            'total_cells': len(df),
            'valid_count': 0,
            'invalid_count': 0,
            'resolution_mismatches': 0,
            'out_of_bounds_count': 0,
            'out_of_land_count': 0,
            'invalid_samples': []
        }
        
        # Extract land polygons if provided
        land_polys = []
        if land_geometry:
            land_polys = extract_polygon_coords(land_geometry)
        
        for i, h3_cell in enumerate(df[h3_column]):
            try:
                # Validate H3 cell identifier
                if not h3.is_valid_cell(h3_cell):
                    validation_results['invalid_count'] += 1
                    if len(validation_results['invalid_samples']) < 10:
                        validation_results['invalid_samples'].append({
                            'index': i,
                            'cell': h3_cell,
                            'reason': 'Invalid H3 cell identifier'
                        })
                    continue
                
                validation_results['valid_count'] += 1
                
                # Check resolution if specified
                if resolution is not None:
                    actual_resolution = h3.get_resolution(h3_cell)
                    if actual_resolution != resolution:
                        validation_results['resolution_mismatches'] += 1
                        if len(validation_results['invalid_samples']) < 10:
                            validation_results['invalid_samples'].append({
                                'index': i,
                                'cell': h3_cell,
                                'reason': f'Resolution mismatch: expected {resolution}, got {actual_resolution}'
                            })
                
                # Check bounding box if specified
                if lon_min is not None and lon_max is not None and lat_min is not None and lat_max is not None:
                    lat, lon = h3.cell_to_latlng(h3_cell)
                    if lon < lon_min or lon > lon_max or lat < lat_min or lat > lat_max:
                        validation_results['out_of_bounds_count'] += 1
                        if len(validation_results['invalid_samples']) < 10:
                            validation_results['invalid_samples'].append({
                                'index': i,
                                'cell': h3_cell,
                                'center': (lon, lat),
                                'reason': f'Cell center ({lon:.6f}, {lat:.6f}) outside bounding box'
                            })
                
                # Check land boundaries if provided
                if land_polys:
                    lat, lon = h3.cell_to_latlng(h3_cell)
                    point_in_land = False
                    for poly in land_polys:
                        if is_point_in_polygon(poly, (lon, lat)):
                            point_in_land = True
                            break
                    
                    if not point_in_land:
                        validation_results['out_of_land_count'] += 1
                        if len(validation_results['invalid_samples']) < 10:
                            validation_results['invalid_samples'].append({
                                'index': i,
                                'cell': h3_cell,
                                'center': (lon, lat),
                                'reason': f'Cell center ({lon:.6f}, {lat:.6f}) outside land boundaries'
                            })
                
            except Exception as e:
                validation_results['invalid_count'] += 1
                if len(validation_results['invalid_samples']) < 10:
                    validation_results['invalid_samples'].append({
                        'index': i,
                        'cell': h3_cell,
                        'error': str(e)
                    })
        
        # Calculate validity percentage
        validity_percentage = (validation_results['valid_count'] / 
                              validation_results['total_cells']) if validation_results['total_cells'] > 0 else 0
        validation_results['validity_percentage'] = validity_percentage * 100
        
        # Determine if validation passed
        validation_results['passed'] = (
            validity_percentage >= self.validation_options['min_valid_geometries'] and
            validation_results['resolution_mismatches'] == 0
        )
        
        return validation_results
    
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
    def validate_dataset(self, df, expected_correlations, value_ranges, geom_column='geom', format_type='WKT', 
                        geom_type=None, h3_resolution=None, land_geometry=None):
        sample_size = 100000
        validate_df = df if len(df) <= sample_size else df.sample(sample_size)
        logger.info(f"Starting dataset validation on {len(validate_df):,} rows")
        logger.info("Validating statistical properties...")
        stat_results = self.validate_statistical_properties(
            validate_df, expected_correlations, value_ranges)
        
        # Detect H3 geometry type and use appropriate validation
        is_h3 = geom_type == "H3" if geom_type else False
        
        if is_h3:
            logger.info("Validating H3 cells...")
            geom_results = self.validate_h3_cells(
                validate_df, 
                h3_column=geom_column,
                resolution=h3_resolution,
                lon_min=getattr(self, 'lon_min', None),
                lon_max=getattr(self, 'lon_max', None),
                lat_min=getattr(self, 'lat_min', None),
                lat_max=getattr(self, 'lat_max', None),
                land_geometry=land_geometry
            )
        else:
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
        
        if is_h3:
            logger.info(f"H3 validation: {'PASSED' if geom_results['passed'] else 'FAILED'}")
            logger.info(f"- Valid H3 cells: {geom_results['valid_count']:,} ({geom_results['validity_percentage']:.2f}%)")
            logger.info(f"- Invalid H3 cells: {geom_results['invalid_count']:,}")
            logger.info(f"- Resolution mismatches: {geom_results['resolution_mismatches']:,}")
            logger.info(f"- Out of bounds: {geom_results['out_of_bounds_count']:,}")
            if land_geometry:
                logger.info(f"- Out of land boundaries: {geom_results['out_of_land_count']:,}")
        else:
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

def validate_generated_data(df, geom_type, format_type, lon_min, lon_max, lat_min, lat_max, 
                           h3_resolution=None, land_geometry=None):
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
        'WKT' if format_type == "WKT" else 'GeoJSON',
        geom_type=geom_type,
        h3_resolution=h3_resolution,
        land_geometry=land_geometry
    )
    if 'repaired_count' in results['geometry_validation'] and results['geometry_validation']['repaired_count'] > 0:
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
        num_rows = config['num_rows']
        use_chunking = config['use_chunking'].lower() == 'yes'
        geometry_type = config['geometry_type']
        
        # Handle H3 resolution from config (always uses fixed resolution 9)
        h3_resolution = None
        if geometry_type == 4:
            if not H3_AVAILABLE:
                logger.error("Error: H3 library is not installed. Please install it using:")
                logger.error("pip install h3")
                exit(1)
            h3_resolution = H3_RESOLUTION
            logger.info(f"Using H3 resolution {h3_resolution} (~0.1 km² hexagons)")
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
            geojson_path = validate_input(f"Enter path to GeoJSON file for {BOUNDING_BOXES[area_choice]['name']} land boundaries (e.g., geojson/jkt.geojson): ", None, str, allow_empty=True)
        num_rows = validate_input("Enter number of rows: ", None, int)
        chunking_threshold = 100000
        use_chunking = True
        if num_rows > chunking_threshold:
            use_chunking = validate_input(f"Large dataset detected ({num_rows} rows). Use chunked file output? (yes/no, default: yes): ", ['yes', 'no', ''], str).lower() != 'no'
        
        # Geometry type selection with H3 support
        if H3_AVAILABLE:
            geometry_type = validate_input("Choose geometry type: 1 for POINT, 2 for POLYGON, 3 for MULTIPOLYGON, 4 for H3\nEnter choice: ", [1, 2, 3, 4], int)
        else:
            geometry_type = validate_input("Choose geometry type: 1 for POINT, 2 for POLYGON, 3 for MULTIPOLYGON\nEnter choice: ", [1, 2, 3], int)
        
        # H3 resolution is fixed at level 9
        h3_resolution = None
        if geometry_type == 4:
            if not H3_AVAILABLE:
                logger.error("Error: H3 library is not installed. Please install it using:")
                logger.error("pip install h3")
                exit(1)
            h3_resolution = H3_RESOLUTION
            logger.info(f"Using H3 resolution {h3_resolution} (~0.1 km² hexagons)")
    format_type = "WKT" if format_choice == 1 else "GeoJSON"
    area = BOUNDING_BOXES[area_choice]
    lon_min, lon_max = area["lon_min"], area["lon_max"]
    lat_min, lat_max = area["lat_min"], area["lat_max"]
    area_name = area["name"]
    geom_type = "POINT" if geometry_type == 1 else "POLYGON" if geometry_type == 2 else "MULTIPOLYGON" if geometry_type == 3 else "H3"
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
        h3_resolution=h3_resolution if geometry_type == 4 else None
    )
    df, validation_results = validate_generated_data(
        df, geom_type, format_type, lon_min, lon_max, lat_min, lat_max,
        h3_resolution=h3_resolution if geometry_type == 4 else None,
        land_geometry=land_geometry
    )
    
    # Generate filename with H3 support
    if geom_type == "H3":
        filename_prefix = f"{area_name.lower()}_data_{num_rows}r_{num_columns}c_h3_res{h3_resolution}"
    else:
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
