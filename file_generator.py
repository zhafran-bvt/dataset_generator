import random
import pandas as pd
import json
import os
from shapely.geometry import shape, Point, Polygon, MultiPolygon
from shapely.ops import unary_union
from tqdm import tqdm
from shapely.strtree import STRtree  # Added for spatial indexing

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
    "Sanitation Access": (50, 100)
}

# Define bounding boxes for each area
BOUNDING_BOXES = {
    1: {"name": "Jakarta", "lon_min": 106.65, "lon_max": 106.95, "lat_min": -6.35, "lat_max": -6.1},
    2: {"name": "Yogyakarta", "lon_min": 110.35, "lon_max": 110.5, "lat_min": -7.85, "lat_max": -7.75},
    3: {"name": "Indonesia", "lon_min": 95.0, "lon_max": 141.0, "lat_min": -11.0, "lat_max": 6.0},
    4: {"name": "Japan", "lon_min": 129.0, "lon_max": 146.0, "lat_min": 30.0, "lat_max": 45.0},
    5: {"name": "Vietnam", "lon_min": 102.0, "lon_max": 110.0, "lat_min": 8.0, "lat_max": 23.0}
}

def generate_realistic_value(label):
    """
    Generate a realistic value for a given label based on predefined ranges.
    """
    min_val, max_val = VALUE_RANGES.get(label, (0, 100))
    return round(random.uniform(min_val, max_val), 2)

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
    """
    try:
        with open(geojson_path, 'r') as f:
            data = json.load(f)
        
        if data['type'] == 'FeatureCollection':
            geometries = [shape(feature['geometry']) for feature in data['features'] if 'geometry' in feature]
            if not geometries:
                print("No valid geometries found in FeatureCollection.")
                return None
            combined_geometry = unary_union(geometries)
            return combined_geometry
        elif data['type'] == 'Polygon':
            return Polygon(data['coordinates'])
        elif data['type'] == 'MultiPolygon':
            return MultiPolygon(data['coordinates'])
        else:
            print("Unsupported GeoJSON type.")
            return None
    except Exception as e:
        print(f"Error loading GeoJSON file: {e}")
        return None

def generate_random_geom(geom_type, format_type, lon_min, lon_max, lat_min, lat_max, land_geometry=None):
    """
    Generate random geometry (POINT, POLYGON, or MULTIPOLYGON) within specified bounds or land geometry.
    If land_geometry is provided, the geometry is clipped to ensure it stays within land boundaries.
    The size of POLYGON and MULTIPOLYGON is scaled based on the area's longitudinal and latitudinal extent.
    """
    if land_geometry:
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

def generate_random_dataframe(rows, cols, geom_type, format_type, lon_min, lon_max, lat_min, lat_max, land_geometry=None):
    """
    Generate a DataFrame with random data and non-intersecting geometries.
    Stops when the desired number of rows is reached or no more non-intersecting geometries can be placed.
    """
    from shapely.wkt import loads  # Import WKT parser
    if cols > len(REALISTIC_LABELS) + 2:
        cols = len(REALISTIC_LABELS) + 2
    labels = ["id", "geom"] + random.sample(REALISTIC_LABELS, cols - 2)
    data = []
    existing_geometries = []
    tree = STRtree(existing_geometries)  # Initialize with empty list

    with tqdm(total=rows, desc="Generating rows") as pbar:
        attempts = 0
        max_attempts = 1000  # Maximum attempts to place a geometry
        while len(data) < rows and attempts < max_attempts:
            geom = generate_random_geom(geom_type, format_type, lon_min, lon_max, lat_min, lat_max, land_geometry)
            # Convert geom to shapely geometry based on format_type
            if format_type == "WKT":
                current_geometry = loads(geom)  # Parse WKT string
            else:
                current_geometry = shape(json.loads(geom))  # Parse GeoJSON

            # Check for intersection with existing geometries
            if existing_geometries and tree.query(current_geometry).size > 0:
                attempts += 1
                continue

            # Add geometry if no intersection
            existing_geometries.append(current_geometry)
            tree = STRtree(existing_geometries)  # Update the spatial index
            row = {"id": len(data) + 1}
            row["geom"] = geom
            for label in labels[2:]:
                row[label] = generate_realistic_value(label)
            data.append(row)
            pbar.update(1)
            attempts = 0  # Reset attempts on success

        if len(data) < rows:
            print(f"Warning: Only {len(data)} geometries could be placed without intersection. Maximum attempts reached.")

    return pd.DataFrame(data, columns=labels)

def save_files(df, filename_prefix):
    """
    Save DataFrame to CSV and Excel files.
    """
    csv_file = f"{filename_prefix}.csv"
    excel_file = f"{filename_prefix}.xlsx"
    
    df.to_csv(csv_file, index=False)
    print(f"Saved: {csv_file} ({os.path.getsize(csv_file) / 1024 / 1024:.2f} MB)")
    
    try:
        df.to_excel(excel_file, index=False)
        print(f"Saved: {excel_file} ({os.path.getsize(excel_file) / 1024 / 1024:.2f} MB)")
    except Exception as e:
        print(f"Error saving Excel file: {e}")

if __name__ == "__main__":
    print("Choose format: 1 for WKT, 2 for GeoJSON")
    format_choice = int(input("Enter choice: "))
    format_type = "WKT" if format_choice == 1 else "GeoJSON"
    
    cols = int(input(f"Enter number of columns (max {len(REALISTIC_LABELS) + 2}): "))
    
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
    print("Choose geometry type: 1 for POINT, 2 for POLYGON, 3 for MULTIPOLYGON")
    geom_choice = int(input("Enter choice: "))
    geom_type = "POINT" if geom_choice == 1 else "POLYGON" if geom_choice == 2 else "MULTIPOLYGON"
    
    if area_choice == 1:
        df = generate_random_dataframe(rows, cols, geom_type, format_type, lon_min, lon_max, lat_min, lat_max, land_geometry)
        filename_prefix = f"jakarta_data_{rows}r_{cols}c_{geom_type.lower()}_{format_type.lower()}_land"
    elif area_choice in [3, 4, 5]:
        df = generate_random_dataframe(rows, cols, geom_type, format_type, lon_min, lon_max, lat_min, lat_max, land_geometry)
        filename_prefix = f"{area_name.lower()}_data_{rows}r_{cols}c_{geom_type.lower()}_{format_type.lower()}_land"
    else:  # area_choice == 2
        df = generate_random_dataframe(rows, cols, geom_type, format_type, lon_min, lon_max, lat_min, lat_max)
        filename_prefix = f"{area_name.lower()}_data_{rows}r_{cols}c_{geom_type.lower()}_{format_type.lower()}"
    
    save_files(df, filename_prefix)