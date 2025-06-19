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

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Define options for new categorical and economic columns
GENDER_OPTIONS = ['Male', 'Female', 'Other']
OCCUPATION_OPTIONS = ['Employed', 'Unemployed', 'Student', 'Retired', 'Homemaker', 'Other']
EDUCATION_OPTIONS = ['Less than High School', 'High School', 'Some College', 'Associate Degree', "Bachelor's Degree", "Master's Degree", 'Doctorate']
EMPLOYMENT_STATUS_OPTIONS = ['Full-time', 'Part-time', 'Self-employed', 'Unemployed', 'Retired', 'Student', 'Other']
HEALTHCARE_ACCESS_OPTIONS = [True, False]

# Constants for realistic data generation
REALISTIC_LABELS = [
    "Population", "Birth Rate", "Death Rate", "Unemployment Rate",
    "Income per Capita", "GDP Growth", "Education Level",
    "Health Index", "Crime Rate", "Urbanization Rate",
    "Life Expectancy", "Poverty Rate", "Employment Rate",
    "Literacy Rate", "Housing Density", "Public Transport Usage",
    "Internet Penetration", "Energy Consumption",
    "Water Access", "Sanitation Access"
]

# Define realistic ranges for each label
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

# Define education level to numeric mapping
EDUCATION_LEVEL_MAPPING = {
    'Less than High School': 1,
    'High School': 2,
    'Some College': 3,
    'Associate Degree': 4,
    "Bachelor's Degree": 5,
    "Master's Degree": 6,
    'Doctorate': 7
}

# Define correlations between variables
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

# Define bounding boxes for each area
BOUNDING_BOXES = {
    1: {"name": "Jakarta", "lon_min": 106.65, "lon_max": 106.95, "lat_min": -6.35, "lat_max": -6.1},
    2: {"name": "Yogyakarta", "lon_min": 110.35, "lon_max": 110.5, "lat_min": -7.85, "lat_max": -7.75},
    3: {"name": "Indonesia", "lon_min": 95.0, "lon_max": 141.0, "lat_min": -11.0, "lat_max": 6.0},
    4: {"name": "Japan", "lon_min": 129.0, "lon_max": 146.0, "lat_min": 30.0, "lat_max": 45.0},
    5: {"name": "Vietnam", "lon_min": 102.0, "lon_max": 110.0, "lat_min": 8.0, "lat_max": 23.0}
}

def validate_input(prompt, valid_options, type_cast=str, allow_empty=False):
    """Validate user input against valid options."""
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
    """Load configuration from a JSON file."""
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading config file: {e}")
        raise

class CorrelationEngine:
    """Handle generation of correlated values for dataset labels."""
    def __init__(self, correlations, value_ranges):
        self.correlations = correlations
        self.value_ranges = value_ranges

    def generate_values(self, labels, seed=None):
        if seed is not None:
            np.random.seed(seed)
        normalized_values = {label: random.random() for label in labels if label in self.value_ranges}
        for _ in range(3):
            for var1, var2, corr_strength in self.correlations:
                if var1 in normalized_values and var2 in normalized_values:
                    val1 = normalized_values[var1]
                    target_val2 = val1 if corr_strength > 0 else 1 - val1
                    adjustment = (target_val2 - normalized_values[var2]) * abs(corr_strength) * 0.5
                    normalized_values[var2] = np.clip(normalized_values[var2] + adjustment, 0, 1)
        result = {}
        for label, norm_val in normalized_values.items():
            min_val, max_val = self.value_ranges[label]
            result[label] = round(norm_val * (max_val - min_val) + min_val, 2)
        if "Education Level" in labels and "Education Level" in result:
            edu_level_num = int(result["Education Level"])
            edu_level_num = max(1, min(edu_level_num, 10))
            if edu_level_num <= 2:
                result["Education Level"] = "Less than High School"
            elif edu_level_num <= 4:
                result["Education Level"] = "High School"
            elif edu_level_num <= 5:
                result["Education Level"] = "Some College"
            elif edu_level_num <= 6:
                result["Education Level"] = "Associate Degree"
            elif edu_level_num <= 8:
                result["Education Level"] = "Bachelor's Degree"
            elif edu_level_num <= 9:
                result["Education Level"] = "Master's Degree"
            else:
                result["Education Level"] = "Doctorate"
        return result

def generate_random_datetime():
    """Generate a random datetime in 2025 in ISO 8601 format."""
    start = datetime(2025, 1, 1, 0, 0, 0)
    end = datetime(2025, 12, 31, 23, 59, 59)
    delta = end - start
    random_seconds = random.randint(0, int(delta.total_seconds()))
    random_dt = start + timedelta(seconds=random_seconds)
    return random_dt.strftime("%Y-%m-%dT%H:%M:%SZ")

def normalize_value(value, min_val, max_val):
    """Normalize a value to be between 0 and 1."""
    return (value - min_val) / (max_val - min_val)

def denormalize_value(normalized_value, min_val, max_val):
    """Convert a normalized value back to its original range."""
    return normalized_value * (max_val - min_val) + min_val

def random_point_in_geometry(geometry):
    """Generate a random point within a given geometry."""
    minx, miny, maxx, maxy = geometry.bounds
    while True:
        point = Point(random.uniform(minx, maxx), random.uniform(miny, maxy))
        if geometry.contains(point):
            return point

def load_land_geometry(geojson_path, simplify_tolerance=0.0001):
    """Load land geometry from a GeoJSON file, combining all features."""
    try:
        gdf = gpd.read_file(geojson_path)
        if gdf.empty:
            logger.error("No geometries found in the GeoJSON file.")
            return None
        file_size_mb = os.path.getsize(geojson_path) / (1024 * 1024)
        if file_size_mb > 10:
            logger.info(f"Large GeoJSON file detected ({file_size_mb:.2f} MB). Simplifying geometries...")
            gdf['geometry'] = gdf['geometry'].simplify(simplify_tolerance, preserve_topology=True)
        combined_geometry = unary_union(gdf['geometry'].values)
        return combined_geometry
    except Exception as e:
        logger.error(f"Error loading GeoJSON file: {e}")
        return None

def generate_random_geom(geom_type, format_type, lon_min, lon_max, lat_min, lat_max, land_geometry=None, spatial_clusters=None):
    """Generate random geometry within bounds or land geometry."""
    if spatial_clusters is not None and random.random() < spatial_clusters['use_probability']:
        x, y = spatial_clusters['model'].resample(1).T[0]
        lon = np.clip(x, lon_min, lon_max)
        lat = np.clip(y, lat_min, lat_max)
        if land_geometry and not land_geometry.contains(Point(lon, lat)):
            point = random_point_in_geometry(land_geometry)
            lon, lat = point.x, point.y
    elif land_geometry:
        point = random_point_in_geometry(land_geometry)
        lon, lat = point.x, point.y
    else:
        lon = random.uniform(lon_min, lon_max)
        lat = random.uniform(lat_min, lat_max)
    
    if geom_type == "POINT":
        if format_type == "WKT":
            return f"POINT ({lon:.6f} {lat:.6f})"
        return json.dumps({"type": "Point", "coordinates": [lon, lat]})
    
    elif geom_type in ["POLYGON", "MULTIPOLYGON"]:
        lon_extent = lon_max - lon_min
        lat_extent = lat_max - lat_min
        width = random.uniform(lon_extent * 0.005, lon_extent * 0.05)
        height = random.uniform(lat_extent * 0.005, lat_extent * 0.05)
        coords = [
            [lon, lat],
            [lon + width, lat],
            [lon + width, lat + height],
            [lon, lat + height],
            [lon, lat]
        ]
        initial_polygon = Polygon(coords)
        if land_geometry:
            clipped_geometry = land_geometry.intersection(initial_polygon)
            if clipped_geometry.is_empty:
                width = lon_extent * 0.005
                height = lat_extent * 0.005
                coords = [
                    [lon, lat],
                    [lon + width, lat],
                    [lon + width, lat + height],
                    [lon, lat + height],
                    [lon, lat]
                ]
                clipped_geometry = land_geometry.intersection(Polygon(coords))
            if isinstance(clipped_geometry, Polygon):
                coords = list(clipped_geometry.exterior.coords)
            elif isinstance(clipped_geometry, MultiPolygon):
                coords = list(clipped_geometry.geoms[0].exterior.coords)
            else:
                coords = [[lon, lat], [lon, lat], [lon, lat], [lon, lat], [lon, lat]]
        else:
            clipped_geometry = initial_polygon
            coords = list(clipped_geometry.exterior.coords)
        if format_type == "WKT":
            coord_str = ", ".join([f"{x:.6f} {y:.6f}" for x, y in coords])
            if geom_type == "POLYGON":
                return f"POLYGON (({coord_str}))"
            return f"MULTIPOLYGON ((({coord_str})))"
        if geom_type == "POLYGON":
            return json.dumps({"type": "Polygon", "coordinates": [coords]})
        return json.dumps({"type": "MultiPolygon", "coordinates": [[coords]]})

def create_spatial_clustering_model(lon_min, lon_max, lat_min, lat_max, land_geometry, cluster_count=5, points_per_cluster=200):
    """Create a spatial clustering model for realistic population distributions."""
    centers = []
    for _ in range(cluster_count):
        if land_geometry:
            point = random_point_in_geometry(land_geometry)
            centers.append((point.x, point.y))
        else:
            lon = random.uniform(lon_min, lon_max)
            lat = random.uniform(lat_min, lat_max)
            centers.append((lon, lat))
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
    return {
        'model': gaussian_kde(samples.T),
        'use_probability': 0.85,
        'centers': centers
    }

def generate_batch_rows(batch_params):
    """Generate a batch of rows for parallel processing."""
    start_id, batch_size, geom_type, format_type, lon_min, lon_max, lat_min, lat_max, land_geometry, labels, include_demographic, include_economic, spatial_clusters, correlation_engine = batch_params
    data = []
    existing_geometries = []
    tree = STRtree(existing_geometries) if existing_geometries else None
    attempts = 0
    max_attempts = 1000
    while len(data) < batch_size and attempts < max_attempts:
        geom = generate_random_geom(geom_type, format_type, lon_min, lon_max, lat_min, lat_max, land_geometry, spatial_clusters)
        if format_type == "WKT":
            current_geometry = loads(geom)
        else:
            current_geometry = shape(json.loads(geom))
        if tree and tree.query(current_geometry).size > 0:
            attempts += 1
            continue
        existing_geometries.append(current_geometry)
        tree = STRtree(existing_geometries)
        row = {"id": start_id + len(data) + 1}
        row["geom"] = geom
        row["date_created"] = generate_random_datetime()
        correlated_values = correlation_engine.generate_values(labels, seed=row["id"])
        if include_demographic:
            row["Gender"] = random.choice(GENDER_OPTIONS)
            row["Occupation"] = random.choice(OCCUPATION_OPTIONS)
            if "Education Level" not in correlated_values:
                row["Education Level"] = random.choice(EDUCATION_OPTIONS)
            else:
                row["Education Level"] = correlated_values["Education Level"]
        if include_economic:
            if "Household Income" in correlated_values:
                row["Household Income"] = correlated_values["Household Income"]
            row["Employment Status"] = random.choice(EMPLOYMENT_STATUS_OPTIONS)
            row["Access to Healthcare"] = random.choice(HEALTHCARE_ACCESS_OPTIONS)
        for label in labels:
            if label not in row and label not in ["id", "geom", "date_created"]:
                if label in correlated_values:
                    row[label] = correlated_values[label]
                else:
                    min_val, max_val = VALUE_RANGES.get(label, (0, 100))
                    row[label] = round(random.uniform(min_val, max_val), 2)
        data.append(row)
        attempts = 0
    return data

def save_files_chunked(df, filename_prefix, chunk_size=100000):
    """Save DataFrame to CSV and Excel files with chunking for large files."""
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    csv_file = os.path.join(output_dir, f"{filename_prefix}.csv")
    excel_file = os.path.join(output_dir, f"{filename_prefix}.xlsx")
    total_rows = len(df)
    if total_rows <= chunk_size:
        df.to_csv(csv_file, index=False, encoding='utf-8')
        logger.info(f"Saved: {csv_file} ({os.path.getsize(csv_file) / 1024 / 1024:.2f} MB)")
        try:
            if total_rows <= 1000000:
                df.to_excel(excel_file, index=False)
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

def generate_parallel_dataframe(rows, cols, geom_type, format_type, lon_min, lon_max, lat_min, lat_max, land_geometry=None, include_demographic=False, include_economic=False, use_spatial_clustering=False, cluster_count=5, points_per_cluster=200):
    """Generate a DataFrame with random data using parallel processing."""
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
    spatial_clusters = None
    if use_spatial_clustering:
        logger.info("Creating spatial clustering model...")
        spatial_clusters = create_spatial_clustering_model(lon_min, lon_max, lat_min, lat_max, land_geometry, cluster_count, points_per_cluster)
        logger.info(f"Created model with {len(spatial_clusters['centers'])} population clusters")
    num_cores = max(1, mp.cpu_count() - 1)
    logger.info(f"Using {num_cores} CPU cores for parallel processing")
    batch_size = max(100, rows // num_cores)
    num_batches = math.ceil(rows / batch_size)
    correlation_engine = CorrelationEngine(VARIABLE_CORRELATIONS, VALUE_RANGES)
    batch_params = [
        (i * batch_size, 
         min(batch_size, rows - i * batch_size),
         geom_type, 
         format_type, 
         lon_min, 
         lon_max, 
         lat_min, 
         lat_max, 
         land_geometry, 
         labels, 
         include_demographic, 
         include_economic,
         spatial_clusters,
         correlation_engine) 
        for i in range(num_batches)
    ]
    progress = tqdm(total=rows, desc="Generating rows")
    all_data = []
    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        future_to_batch = {executor.submit(generate_batch_rows, params): params for params in batch_params}
        for future in as_completed(future_to_batch):
            batch_data = future.result()
            all_data.extend(batch_data)
            progress.update(len(batch_data))
    progress.close()
    logger.info(f"Generated {len(all_data)} rows of data")
    df = pd.DataFrame(all_data, columns=labels)
    logger.info("Analyzing correlations in generated data...")
    analyze_correlations(df)
    return df

def analyze_correlations(df):
    """Analyze and log correlations between variables."""
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

if __name__ == "__main__":
    logger.info("Dataset Generator with Performance Improvements")
    logger.info("==============================================")
    logger.info("Last update: 2025-06-19 13:34:10")
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
        
        land_geometry = None
        if area_choice in [1, 3, 4, 5]:
            geojson_path = validate_input(f"Enter path to GeoJSON file for {BOUNDING_BOXES[area_choice]['name']} land boundaries (e.g., geojson/id-jk.min.geojson): ", None, str, allow_empty=True)
            if geojson_path:
                land_geometry = load_land_geometry(geojson_path)
                if land_geometry is None:
                    logger.error("Failed to load land geometry. Exiting.")
                    exit(1)
        
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
    logger.info(f"Generating {num_rows} rows of {geom_type} geometries for {area_name}...")
    logger.info(f"Using {'spatial clustering' if use_spatial_clustering else 'uniform distribution'} for point placement")
    logger.info(f"Generating correlated values based on {len(VARIABLE_CORRELATIONS)} defined relationships")

    if area_choice in [1, 3, 4, 5] and land_geometry:
        df = generate_parallel_dataframe(num_rows, num_columns, geom_type, format_type, lon_min, lon_max, lat_min, lat_max, 
                                        land_geometry, include_demographic, include_economic, use_spatial_clustering)
        filename_prefix = f"{area_name.lower()}_data_{num_rows}r_{num_columns}c_{geom_type.lower()}_{format_type.lower()}_land"
    else:
        df = generate_parallel_dataframe(num_rows, num_columns, geom_type, format_type, lon_min, lon_max, lat_min, lat_max, 
                                        include_demographic=include_demographic, include_economic=include_economic, 
                                        use_spatial_clustering=use_spatial_clustering)
        filename_prefix = f"{area_name.lower()}_data_{num_rows}r_{num_columns}c_{geom_type.lower()}_{format_type.lower()}"

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
                df.to_excel(excel_file, index=False)
                logger.info(f"Saved: {excel_file} ({os.path.getsize(excel_file) / 1024 / 1024:.2f} MB)")
            else:
                logger.warning("Dataset too large for Excel (> 1M rows). Excel file not created.")
        except Exception as e:
            logger.error(f"Error saving Excel file: {e}")

    logger.info("Process completed successfully.")