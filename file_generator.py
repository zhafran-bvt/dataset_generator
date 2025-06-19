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
    "Household Income": (0, 1000000)  # Added for economic indicator
}

# Define education level to numeric mapping for correlation calculations
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
# Format: (variable1, variable2, correlation_strength)
# Correlation strength: -1.0 to 1.0 where:
# - Positive values indicate positive correlation
# - Negative values indicate negative correlation
# - Magnitude indicates strength (0 = no correlation, 1/-1 = perfect correlation)
VARIABLE_CORRELATIONS = [
    # Economic correlations
    ("Income per Capita", "Education Level", 0.7),
    ("Income per Capita", "Household Income", 0.85),
    ("Household Income", "Education Level", 0.65),
    ("Income per Capita", "Poverty Rate", -0.8),
    ("Income per Capita", "Life Expectancy", 0.6),
    ("Income per Capita", "Health Index", 0.5),
    
    # Health correlations
    ("Health Index", "Life Expectancy", 0.75),
    ("Poverty Rate", "Health Index", -0.65),
    ("Water Access", "Health Index", 0.55),
    ("Sanitation Access", "Health Index", 0.6),
    
    # Education correlations
    ("Education Level", "Literacy Rate", 0.8),
    ("Education Level", "Employment Rate", 0.6),
    ("Education Level", "Birth Rate", -0.5),
    
    # Infrastructure correlations
    ("Urbanization Rate", "Internet Penetration", 0.65),
    ("Urbanization Rate", "Public Transport Usage", 0.7),
    ("Urbanization Rate", "Energy Consumption", 0.55),
    
    # Population correlations
    ("Population", "Housing Density", 0.6),
    ("Housing Density", "Public Transport Usage", 0.5),
    
    # Employment correlations
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

def generate_random_datetime():
    """
    Generate a random datetime in 2025 in ISO 8601 format (e.g., 2025-06-17T13:00:00Z).
    """
    start = datetime(2025, 1, 1, 0, 0, 0)
    end = datetime(2025, 12, 31, 23, 59, 59)
    delta = end - start
    random_seconds = random.randint(0, int(delta.total_seconds()))
    random_dt = start + timedelta(seconds=random_seconds)
    return random_dt.strftime("%Y-%m-%dT%H:%M:%SZ")

def normalize_value(value, min_val, max_val):
    """
    Normalize a value to be between 0 and 1 based on its min/max range.
    """
    return (value - min_val) / (max_val - min_val)

def denormalize_value(normalized_value, min_val, max_val):
    """
    Convert a normalized value (0-1) back to its original range.
    """
    return normalized_value * (max_val - min_val) + min_val

def random_point_in_geometry(geometry):
    """
    Generate a random point within a given geometry.
    """
    minx, miny, maxx, maxy = geometry.bounds
    while True:
        point = Point(random.uniform(minx, maxx), random.uniform(miny, maxy))
        if geometry.contains(point):
            return point

def load_land_geometry(geojson_path):
    """
    Load land geometry from a GeoJSON file, combining all features into a single geometry.
    Uses geopandas for more efficient loading of large GeoJSON files.
    """
    try:
        # Use GeoPandas for more efficient loading
        gdf = gpd.read_file(geojson_path)
        
        # Check if GeoDataFrame is empty
        if gdf.empty:
            print("No geometries found in the GeoJSON file.")
            return None
        
        # Simplify geometries for better performance if the file is large (> 10MB)
        file_size_mb = os.path.getsize(geojson_path) / (1024 * 1024)
        if file_size_mb > 10:
            print(f"Large GeoJSON file detected ({file_size_mb:.2f} MB). Simplifying geometries...")
            tolerance = 0.0001  # Adjust tolerance based on your data
            gdf['geometry'] = gdf['geometry'].simplify(tolerance, preserve_topology=True)
        
        # Combine all geometries into a single multipolygon
        combined_geometry = unary_union(gdf['geometry'].values)
        return combined_geometry
    
    except Exception as e:
        print(f"Error loading GeoJSON file: {e}")
        return None

def generate_random_geom(geom_type, format_type, lon_min, lon_max, lat_min, lat_max, land_geometry=None, spatial_clusters=None):
    """
    Generate random geometry (POINT, POLYGON, or MULTIPOLYGON) within specified bounds or land geometry.
    If spatial_clusters is provided, use it to determine the location based on the clustering model.
    """
    if spatial_clusters is not None and random.random() < spatial_clusters['use_probability']:
        # Use spatial clustering to determine point location
        x, y = spatial_clusters['model'].resample(1).T[0]
        lon = np.clip(x, lon_min, lon_max)
        lat = np.clip(y, lat_min, lat_max)
        
        # If land_geometry is provided, ensure the point is within it
        if land_geometry:
            point = Point(lon, lat)
            if not land_geometry.contains(point):
                # Fall back to random point in geometry if the clustered point is not within land
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
        else:
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
            else:
                return f"MULTIPOLYGON ((({coord_str})))"
        else:
            if geom_type == "POLYGON":
                return json.dumps({"type": "Polygon", "coordinates": [coords]})
            else:
                return json.dumps({"type": "MultiPolygon", "coordinates": [[coords]]})

def create_spatial_clustering_model(lon_min, lon_max, lat_min, lat_max, land_geometry, cluster_count=5):
    """
    Create a spatial clustering model to generate realistic population distributions.
    Returns a model that can generate points with spatial clustering.
    """
    # Generate random cluster centers within the bounds
    centers = []
    for _ in range(cluster_count):
        if land_geometry:
            point = random_point_in_geometry(land_geometry)
            centers.append((point.x, point.y))
        else:
            lon = random.uniform(lon_min, lon_max)
            lat = random.uniform(lat_min, lat_max)
            centers.append((lon, lat))
    
    # Generate sample points from each cluster center
    samples = []
    points_per_cluster = 500  # Number of sample points to create the distribution
    
    # Random weights for each cluster to make some clusters more dense than others
    weights = np.random.uniform(0.5, 1.5, cluster_count)
    weights = weights / np.sum(weights)
    
    lon_extent = lon_max - lon_min
    lat_extent = lat_max - lat_min
    
    # Average cluster size (standard deviation) as a fraction of the area extent
    avg_cluster_size = min(lon_extent, lat_extent) * 0.05
    
    for i, (center_x, center_y) in enumerate(centers):
        # Make cluster sizes vary
        cluster_size_x = avg_cluster_size * random.uniform(0.5, 2.0)
        cluster_size_y = avg_cluster_size * random.uniform(0.5, 2.0)
        
        # Generate points for this cluster
        cluster_points = int(points_per_cluster * weights[i])
        x = np.random.normal(center_x, cluster_size_x, cluster_points)
        y = np.random.normal(center_y, cluster_size_y, cluster_points)
        
        # Clip points to bounds
        x = np.clip(x, lon_min, lon_max)
        y = np.clip(y, lat_min, lat_max)
        
        # Add to samples
        for j in range(cluster_points):
            samples.append([x[j], y[j]])
    
    # Convert to numpy array
    samples = np.array(samples)

    # Create KDE model from the samples
    kde = gaussian_kde(samples.T)
    
    return {
        'model': kde,
        'use_probability': 0.85,  # 85% chance of using the cluster model vs. uniform random
        'centers': centers
    }

def generate_correlated_values(selected_labels, seed=None):
    """
    Generate a set of correlated values for the given labels based on the defined correlations.
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Create a dictionary to store normalized values (0-1 range)
    normalized_values = {}
    
    # First pass: generate initial random values for all labels
    for label in selected_labels:
        if label in ["id", "geom", "date_created", "Gender", "Occupation", "Employment Status", "Access to Healthcare"]:
            continue  # Skip non-numeric or special fields
            
        normalized_values[label] = random.random()
    
    # Second pass: apply correlations (multiple iterations to strengthen the effect)
    for _ in range(3):  # Apply correlation adjustments multiple times
        for var1, var2, corr_strength in VARIABLE_CORRELATIONS:
            if var1 in normalized_values and var2 in normalized_values:
                # Get current values
                val1 = normalized_values[var1]
                val2 = normalized_values[var2]
                
                # Calculate target value based on correlation
                target_val2 = val1 if corr_strength > 0 else 1 - val1
                
                # Apply correlation with some randomness
                # Higher correlation strength means closer to target
                adjustment = (target_val2 - val2) * abs(corr_strength) * 0.5
                normalized_values[var2] = np.clip(val2 + adjustment, 0, 1)
    
    # Convert normalized values back to their original ranges
    result = {}
    for label, norm_val in normalized_values.items():
        if label in VALUE_RANGES:
            min_val, max_val = VALUE_RANGES[label]
            result[label] = round(denormalize_value(norm_val, min_val, max_val), 2)
    
    # Handle categorical variables that have correlations
    if "Education Level" in selected_labels and "Education Level" in result:
        # Map the education level number to a category
        edu_level_num = int(result["Education Level"])
        edu_level_num = max(1, min(edu_level_num, 10))  # Ensure in range 1-10
        
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

def generate_batch_rows(batch_params):
    """
    Generate a batch of rows for parallel processing with correlated values and spatial clustering.
    """
    start_id, batch_size, geom_type, format_type, lon_min, lon_max, lat_min, lat_max, land_geometry, labels, include_demographic, include_economic, spatial_clusters = batch_params
    
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
        
        # Generate correlated values for this row
        correlated_values = generate_correlated_values(labels, seed=row["id"])
        
        # Add fixed categorical variables
        if include_demographic:
            row["Gender"] = random.choice(GENDER_OPTIONS)
            row["Occupation"] = random.choice(OCCUPATION_OPTIONS)
            
            # Education Level is handled by the correlated values if it's in labels
            if "Education Level" not in correlated_values:
                row["Education Level"] = random.choice(EDUCATION_OPTIONS)
            else:
                row["Education Level"] = correlated_values["Education Level"]
        
        if include_economic:
            if "Household Income" in correlated_values:
                row["Household Income"] = correlated_values["Household Income"]
            
            row["Employment Status"] = random.choice(EMPLOYMENT_STATUS_OPTIONS)
            row["Access to Healthcare"] = random.choice(HEALTHCARE_ACCESS_OPTIONS)
        
        # Add all other variables, using correlated values when available
        for label in labels:
            if label not in row and label not in ["id", "geom", "date_created"]:
                if label in correlated_values:
                    row[label] = correlated_values[label]
                else:
                    row[label] = generate_realistic_value(label)
            
        data.append(row)
        attempts = 0
        
    return data

def save_files_chunked(df, filename_prefix, chunk_size=100000):
    """
    Save DataFrame to CSV and Excel files with chunking for large files.
    """
    # Create directory for output if it doesn't exist
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    
    csv_file = os.path.join(output_dir, f"{filename_prefix}.csv")
    excel_file = os.path.join(output_dir, f"{filename_prefix}.xlsx")
    
    # Get total number of rows
    total_rows = len(df)
    
    # If the dataset is small, save it directly
    if total_rows <= chunk_size:
        df.to_csv(csv_file, index=False, encoding='utf-8')
        print(f"Saved: {csv_file} ({os.path.getsize(csv_file) / 1024 / 1024:.2f} MB)")
        
        try:
            # For Excel files, limit to 1 million rows (Excel's limitation)
            if total_rows <= 1000000:
                df.to_excel(excel_file, index=False)
                print(f"Saved: {excel_file} ({os.path.getsize(excel_file) / 1024 / 1024:.2f} MB)")
            else:
                print(f"Dataset too large for Excel (> 1M rows). Excel file not created.")
        except Exception as e:
            print(f"Error saving Excel file: {e}")
    else:
        # For large datasets, save in chunks
        print(f"Dataset is large ({total_rows} rows). Saving in chunks...")
        
        # CSV chunks
        num_chunks = math.ceil(total_rows / chunk_size)
        for i in range(num_chunks):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, total_rows)
            
            chunk = df.iloc[start_idx:end_idx]
            chunk_file = os.path.join(output_dir, f"{filename_prefix}_part{i+1}.csv")
            
            # First chunk: write with headers, others: append without headers
            if i == 0:
                chunk.to_csv(chunk_file, index=False, encoding='utf-8')
            else:
                chunk.to_csv(chunk_file, index=False, encoding='utf-8', header=False)
                
            print(f"Saved chunk {i+1}/{num_chunks}: {chunk_file} ({os.path.getsize(chunk_file) / 1024 / 1024:.2f} MB)")
        
        print(f"CSV data saved in {num_chunks} chunks in the '{output_dir}' directory.")
        print("Note: Excel file not created for chunked data (dataset too large).")

def generate_parallel_dataframe(rows, cols, geom_type, format_type, lon_min, lon_max, lat_min, lat_max, land_geometry=None, include_demographic=False, include_economic=False, use_spatial_clustering=False):
    """
    Generate a DataFrame with random data using parallel processing for better performance.
    Includes options for correlated values and spatial clustering.
    """
    # Calculate fixed columns
    fixed_columns = ["id", "geom", "date_created"]
    if include_demographic:
        fixed_columns += ["Gender", "Occupation", "Education Level"]
    if include_economic:
        fixed_columns += ["Household Income", "Employment Status", "Access to Healthcare"]
    
    # Adjust max columns based on fixed columns
    max_cols = len(REALISTIC_LABELS) + len(fixed_columns)
    if cols > max_cols:
        cols = max_cols
    
    # Select random columns from REALISTIC_LABELS
    num_additional = cols - len(fixed_columns)
    additional_labels = random.sample(REALISTIC_LABELS, min(num_additional, len(REALISTIC_LABELS))) if num_additional > 0 else []
    labels = fixed_columns + additional_labels
    
    # Create spatial clustering model if requested
    spatial_clusters = None
    if use_spatial_clustering:
        print("Creating spatial clustering model...")
        spatial_clusters = create_spatial_clustering_model(lon_min, lon_max, lat_min, lat_max, land_geometry)
        print(f"Created model with {len(spatial_clusters['centers'])} population clusters")
    
    # Determine number of CPU cores to use (leave one free for system)
    num_cores = max(1, mp.cpu_count() - 1)
    print(f"Using {num_cores} CPU cores for parallel processing")
    
    # Calculate batch size for each process
    batch_size = max(100, rows // num_cores)
    num_batches = math.ceil(rows / batch_size)
    
    # Prepare batch parameters
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
         spatial_clusters) 
        for i in range(num_batches)
    ]
    
    # Create a progress bar
    progress = tqdm(total=rows, desc="Generating rows")
    
    # Generate data in parallel
    all_data = []
    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        # Submit all batch jobs
        future_to_batch = {executor.submit(generate_batch_rows, params): params for params in batch_params}
        
        # Process results as they complete
        for future in as_completed(future_to_batch):
            batch_data = future.result()
            all_data.extend(batch_data)
            progress.update(len(batch_data))
    
    progress.close()
    
    # Convert to DataFrame
    print(f"Generated {len(all_data)} rows of data")
    df = pd.DataFrame(all_data, columns=labels)
    
    # Analyze correlations for verification
    print("\nAnalyzing correlations in generated data:")
    analyze_correlations(df)
    
    return df

def analyze_correlations(df):
    """
    Analyze and print the correlations between variables in the generated dataset.
    """
    # Get numeric columns only
    numeric_df = df.select_dtypes(include=['number'])
    
    if len(numeric_df.columns) < 2:
        print("Not enough numeric columns for correlation analysis.")
        return
    
    # Calculate correlation matrix
    corr_matrix = numeric_df.corr()
    
    # Print correlation analysis for defined variable pairs
    print("\nCorrelation analysis for key variable pairs:")
    print("-" * 50)
    print(f"{'Variable Pair':<40} | {'Target':<8} | {'Actual':<8}")
    print("-" * 50)
    
    for var1, var2, target_corr in VARIABLE_CORRELATIONS:
        if var1 in corr_matrix.columns and var2 in corr_matrix.columns:
            actual_corr = corr_matrix.loc[var1, var2]
            print(f"{var1} vs {var2:<20} | {target_corr:8.2f} | {actual_corr:8.2f}")
    
    print("-" * 50)

if __name__ == "__main__":
    print("Dataset Generator with Performance Improvements")
    print("==============================================")
    print("Last update: 2025-06-19 00:43:10")
    print("Author: zhafran-bvt")
    print("==============================================\n")
    
    print("Choose format: 1 for WKT, 2 for GeoJSON")
    format_choice = int(input("Enter choice: "))
    format_type = "WKT" if format_choice == 1 else "GeoJSON"
    
    # Prompt for demographic and economic columns
    include_demographic = input("Include demographic columns (Gender, Occupation, Education Level)? (yes/no): ").lower() == 'yes'
    include_economic = input("Include economic columns (Household Income, Employment Status, Access to Healthcare)? (yes/no): ").lower() == 'yes'
    
    # Calculate minimum columns
    min_cols = 3  # id, geom, date_created
    if include_demographic:
        min_cols += 3
    if include_economic:
        min_cols += 3
    
    cols = int(input(f"Enter number of columns (min {min_cols}, max {len(REALISTIC_LABELS) + min_cols}): "))
    while cols < min_cols:
        print(f"Number of columns must be at least {min_cols}.")
        cols = int(input(f"Enter number of columns (min {min_cols}, max {len(REALISTIC_LABELS) + min_cols}): "))
    
    # New option for spatial clustering
    use_spatial_clustering = input("Use spatial clustering to create realistic population distributions? (yes/no): ").lower() == 'yes'
    
    print("Choose area: 1 for Jakarta, 2 for Yogyakarta, 3 for Indonesia, 4 for Japan, 5 for Vietnam")
    area_choice = int(input("Enter choice: "))
    
    if area_choice not in BOUNDING_BOXES:
        print("Invalid area choice. Exiting.")
        exit()
    
    area = BOUNDING_BOXES[area_choice]
    lon_min, lon_max = area["lon_min"], area["lon_max"]
    lat_min, lat_max = area["lat_min"], area["lat_max"]
    area_name = area["name"]
    
    land_geometry = None
    if area_choice in [1, 3, 4, 5]:
        geojson_path = input(f"Enter path to GeoJSON file for {area_name} land boundaries (e.g., geojson/id-jk.min.geojson for Jakarta, geojson/id.json for Indonesia): ")
        land_geometry = load_land_geometry(geojson_path)
        if land_geometry is None:
            print("Failed to load land geometry. Exiting.")
            exit()
    
    rows = int(input("Enter number of rows: "))
    
    # Ask about chunking for large datasets
    chunking_threshold = 100000
    use_chunking = False
    if rows > chunking_threshold:
        use_chunking = input(f"Large dataset detected ({rows} rows). Use chunked file output? (yes/no, default: yes): ").lower() != 'no'
    
    print("Choose geometry type: 1 for POINT, 2 for POLYGON, 3 for MULTIPOLYGON")
    geom_choice = int(input("Enter choice: "))
    geom_type = "POINT" if geom_choice == 1 else "POLYGON" if geom_choice == 2 else "MULTIPOLYGON"
    
    # Start timing
    start_time = datetime.now()
    
    # Generate data using parallel processing
    print(f"\nGenerating {rows} rows of {geom_type} geometries for {area_name}...")
    print(f"Using {'spatial clustering' if use_spatial_clustering else 'uniform distribution'} for point placement")
    print(f"Generating correlated values based on {len(VARIABLE_CORRELATIONS)} defined relationships")
    
    if area_choice == 1:
        df = generate_parallel_dataframe(rows, cols, geom_type, format_type, lon_min, lon_max, lat_min, lat_max, 
                                         land_geometry, include_demographic, include_economic, use_spatial_clustering)
        filename_prefix = f"jakarta_data_{rows}r_{cols}c_{geom_type.lower()}_{format_type.lower()}_land"
    elif area_choice in [3, 4, 5]:
        df = generate_parallel_dataframe(rows, cols, geom_type, format_type, lon_min, lon_max, lat_min, lat_max, 
                                         land_geometry, include_demographic, include_economic, use_spatial_clustering)
        filename_prefix = f"{area_name.lower()}_data_{rows}r_{cols}c_{geom_type.lower()}_{format_type.lower()}_land"
    else:  # area_choice == 2
        df = generate_parallel_dataframe(rows, cols, geom_type, format_type, lon_min, lon_max, lat_min, lat_max, 
                                        land_geometry=None, include_demographic=include_demographic, 
                                        include_economic=include_economic, use_spatial_clustering=use_spatial_clustering)
        filename_prefix = f"{area_name.lower()}_data_{rows}r_{cols}c_{geom_type.lower()}_{format_type.lower()}"
    
    # End timing
    end_time = datetime.now()
    elapsed_time = end_time - start_time
    print(f"\nData generation completed in {elapsed_time.total_seconds():.2f} seconds")
    
    # Save files (with chunking if needed)
    if use_chunking:
        save_files_chunked(df, filename_prefix)
    else:
        # Create directory for output if it doesn't exist
        output_dir = "output"
        os.makedirs(output_dir, exist_ok=True)
        
        csv_file = os.path.join(output_dir, f"{filename_prefix}.csv")
        excel_file = os.path.join(output_dir, f"{filename_prefix}.xlsx")
        
        df.to_csv(csv_file, index=False, encoding='utf-8')
        print(f"Saved: {csv_file} ({os.path.getsize(csv_file) / 1024 / 1024:.2f} MB)")
        
        try:
            # For Excel files, limit to 1 million rows (Excel's limitation)
            if rows <= 1000000:
                df.to_excel(excel_file, index=False)
                print(f"Saved: {excel_file} ({os.path.getsize(excel_file) / 1024 / 1024:.2f} MB)")
            else:
                print(f"Dataset too large for Excel (> 1M rows). Excel file not created.")
        except Exception as e:
            print(f"Error saving Excel file: {e}")
    
    print("\nProcess completed successfully.")