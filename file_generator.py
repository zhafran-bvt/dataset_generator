import random
import pandas as pd
import json
import os
from shapely.geometry import shape, Point
from tqdm import tqdm

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

def load_country_land_geometry(geojson_path):
    """
    Load a country's land geometry from a GeoJSON file.
    """
    try:
        with open(geojson_path, 'r') as f:
            data = json.load(f)
        
        if data['type'] == 'FeatureCollection':
            features = data['features']
            if not features:
                print("No features found in FeatureCollection.")
                return None
            geometry = shape(features[0]['geometry'])
            return geometry
            
        elif data['type'] in ['Polygon', 'MultiPolygon']:
            geometry = shape(data)
            return geometry
            
        print("Unsupported GeoJSON type or no features found.")
        return None
        
    except Exception as e:
        print(f"Error loading GeoJSON file: {e}")
        return None

def load_jakarta_districts(geojson_path, format_type):
    """
    Load Jakarta district geometries from a GeoJSON file.
    """
    try:
        with open(geojson_path, 'r') as f:
            data = json.load(f)
        
        districts = []
        if data['type'] == 'FeatureCollection':
            for feature in data['features']:
                geom = shape(feature['geometry'])
                district_name = feature['properties'].get('name', 'Unknown')
                if format_type == "WKT":
                    geom_str = geom.wkt
                else:
                    geom_str = json.dumps(feature['geometry'])
                districts.append({"district_name": district_name, "geom": geom_str})
        return districts
    except Exception as e:
        print(f"Error loading Jakarta districts: {e}")
        return []

def generate_random_geom(geom_type, format_type, lon_min, lon_max, lat_min, lat_max, land_geometry=None):
    """
    Generate random geometry (POINT, POLYGON, or MULTIPOLYGON) within specified bounds or land geometry.
    If land_geometry is provided, points are generated within it to avoid sea areas.
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
        width = random.uniform(0.001, 0.01)
        height = random.uniform(0.001, 0.01)
        coords = [
            [lon, lat],
            [lon + width, lat],
            [lon + width, lat + height],
            [lon, lat + height],
            [lon, lat]
        ]
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
    Generate a DataFrame with random data and geometries.
    Passes land_geometry to generate_random_geom for land-only point generation.
    Fixed to prevent geom column overwrite.
    """
    if cols > len(REALISTIC_LABELS) + 2:
        cols = len(REALISTIC_LABELS) + 2
    labels = ["id", "geom"] + random.sample(REALISTIC_LABELS, cols - 2)
    data = []
    for i in tqdm(range(rows), desc="Generating rows"):
        row = {"id": i + 1}
        row["geom"] = generate_random_geom(geom_type, format_type, lon_min, lon_max, lat_min, lat_max, land_geometry)
        for label in labels[2:]:
            row[label] = generate_realistic_value(label)
        data.append(row)
    return pd.DataFrame(data, columns=labels)

def generate_district_dataframe(cols, format_type, geojson_path):
    """
    Generate a DataFrame using Jakarta district geometries from a GeoJSON file.
    Assumes load_jakarta_districts is defined elsewhere.
    """
    if cols > len(REALISTIC_LABELS) + 3:
        cols = len(REALISTIC_LABELS) + 3
    labels = ["id", "district_name", "geom"] + random.sample(REALISTIC_LABELS, cols - 3)
    districts = load_jakarta_districts(geojson_path, format_type)
    if not districts:
        print("No district data loaded. Exiting.")
        exit()
    data = []
    for i, district in tqdm(enumerate(districts), total=len(districts), desc="Generating district rows"):
        row = {
            "id": i + 1,
            "district_name": district["district_name"],
            "geom": district["geom"]
        }
        for label in labels[3:]:
            row[label] = generate_realistic_value(label)
        data.append(row)
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
    if area_choice == 1:
        use_districts = input("Use actual district geometries for Jakarta? (y/n): ").lower()
        if use_districts == 'y':
            geojson_path = input("Enter path to GeoJSON file for Jakarta districts (e.g., geojson/jakarta_districts.json): ")
            df = generate_district_dataframe(cols, format_type, geojson_path)
            filename_prefix = f"jakarta_district_data_{format_type.lower()}"
        else:
            rows = int(input("Enter number of rows: "))
            print("Choose geometry type: 1 for POINT, 2 for POLYGON, 3 for MULTIPOLYGON")
            geom_choice = int(input("Enter choice: "))
            geom_type = "POINT" if geom_choice == 1 else "POLYGON" if geom_choice == 2 else "MULTIPOLYGON"
            df = generate_random_dataframe(rows, cols, geom_type, format_type, lon_min, lon_max, lat_min, lat_max)
            filename_prefix = f"jakarta_data_{rows}r_{cols}c_{geom_type.lower()}_{format_type.lower()}"
    else:
        if area_choice in [3, 4, 5]:
            use_land = input("Use land boundaries to avoid sea areas? (y/n): ").lower()
            if use_land == 'y':
                geojson_path = input(f"Enter path to GeoJSON file for {area_name} land boundaries (e.g., geojson/{area_name.lower()}.json): ")
                land_geometry = load_country_land_geometry(geojson_path)
                if land_geometry is None:
                    print("Failed to load land geometry. Exiting.")
                    exit()
                filename_prefix = f"{area_name.lower()}_data_{rows}r_{cols}c_{geom_type.lower()}_{format_type.lower()}_land"
            else:
                filename_prefix = f"{area_name.lower()}_data_{rows}r_{cols}c_{geom_type.lower()}_{format_type.lower()}"
        else:
            filename_prefix = f"{area_name.lower()}_data_{rows}r_{cols}c_{geom_type.lower()}_{format_type.lower()}"
        
        rows = int(input("Enter number of rows: "))
        print("Choose geometry type: 1 for POINT, 2 for POLYGON, 3 for MULTIPOLYGON")
        geom_choice = int(input("Enter choice: "))
        geom_type = "POINT" if geom_choice == 1 else "POLYGON" if geom_choice == 2 else "MULTIPOLYGON"
        df = generate_random_dataframe(rows, cols, geom_type, format_type, lon_min, lon_max, lat_min, lat_max, land_geometry)
    
    save_files(df, filename_prefix)