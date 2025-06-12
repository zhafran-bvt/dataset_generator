import random
import pandas as pd
import json
import os
from shapely.geometry import shape, Point
from shapely.wkt import dumps as wkt_dumps

# Define realistic labels for dataset columns
REALISTIC_LABELS = [
    "Family Identity Card", "Migration", "Mortality", "People Density",
    "Population", "Registered Residents", "Birth Rate", "Death Rate",
    "Unemployment Rate", "Income per Capita", "Households", "Literacy Rate",
    "School Enrollment", "Employment Rate", "Water Access", "Electricity Access",
    "Health Facilities", "Internet Access", "Average Age", "Vehicle Ownership"
]

def generate_realistic_value(label):
    """
    Generate realistic values based on the label type.
    Returns percentages, counts, or categorical values as appropriate.
    """
    if "Rate" in label or label in ["Migration", "Mortality", "Water Access", "Electricity Access", "Internet Access", "School Enrollment", "Literacy Rate"]:
        return round(random.uniform(0, 100), 2)  # Percentages or rates
    elif label in ["Population", "Registered Residents", "Households", "Family Identity Card", "Health Facilities"]:
        return random.randint(1000, 1000000)  # Counts
    elif label == "Income per Capita":
        return random.randint(1000000, 15000000)  # IDR (adjust for other currencies if needed)
    elif label == "Average Age":
        return random.randint(20, 80)  # Age range
    elif label == "People Density":
        return random.randint(1000, 25000)  # People per km²
    elif label == "Vehicle Ownership":
        return random.randint(0, 5)  # Vehicles per household
    else:
        return random.choice(["Low", "Medium", "High"])  # Fallback

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
        else:  # GeoJSON
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
        else:  # GeoJSON
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
    for i in range(rows):
        row = {"id": i + 1}
        row["geom"] = generate_random_geom(geom_type, format_type, lon_min, lon_max, lat_min, lat_max, land_geometry)
        for label in labels[2:]:  # Start from index 2 to skip id and geom
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
    for i, district in enumerate(districts):
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
    Save the DataFrame to CSV and Excel files, printing file sizes.
    """
    csv_file = f"{filename_prefix}.csv"
    excel_file = f"{filename_prefix}.xlsx"
    df.to_csv(csv_file, index=False)
    df.to_excel(excel_file, index=False)
    print(f"✅ Saved: {csv_file} ({os.path.getsize(csv_file) / 1024 / 1024:.2f} MB)")
    print(f"✅ Saved: {excel_file} ({os.path.getsize(excel_file) / 1024 / 1024:.2f} MB)")

def load_country_land_geometry(geojson_path):
    """
    Load a country's land geometry from a GeoJSON file.
    Supports FeatureCollection or direct Polygon/MultiPolygon formats.
    """
    try:
        with open(geojson_path, 'r') as f:
            data = json.load(f)
        if data["type"] == "FeatureCollection":
            features = data["features"]
            if features:
                geometry = shape(features[0]['geometry'])
                return geometry
        elif data["type"] in ["Polygon", "MultiPolygon"]:
            geometry = shape(data)
            return geometry
        print("Unsupported GeoJSON type or no features found.")
        return None
    except Exception as e:
        print(f"Error loading GeoJSON file: {e}")
        return None

def random_point_in_geometry(geometry):
    """
    Generate a random point within the provided geometry, weighted by polygon area.
    Ensures points are on land when using land geometry.
    Compatible with Shapely 1.x and 2.x.
    """
    if geometry.geom_type == 'MultiPolygon':
        polygons = list(geometry.geoms)  # Use .geoms for Shapely 2.x compatibility
    elif geometry.geom_type == 'Polygon':
        polygons = [geometry]
    else:
        raise ValueError("Geometry must be Polygon or MultiPolygon")
    areas = [poly.area for poly in polygons]
    total_area = sum(areas)
    if total_area == 0:
        raise ValueError("Geometry has zero area")
    weights = [area / total_area for area in areas]
    chosen_poly = random.choices(polygons, weights=weights)[0]
    minx, miny, maxx, maxy = chosen_poly.bounds
    while True:
        x = random.uniform(minx, maxx)
        y = random.uniform(miny, maxy)
        point = Point(x, y)
        if chosen_poly.contains(point):
            return point

# Note: load_jakarta_districts is assumed to be defined elsewhere
# def load_jakarta_districts(geojson_path, format_type):
#     # Implement loading districts from GeoJSON
#     pass

if __name__ == "__main__":
    land_geometry = None
    print("Choose format: 1 for WKT, 2 for GeoJSON")
    format_choice = input("Enter choice: ")
    format_type = "WKT" if format_choice == "1" else "GeoJSON"
    cols = int(input(f"Enter number of columns (max {len(REALISTIC_LABELS) + 2}): "))
    print("Choose area: 1 for Jakarta, 2 for Yogyakarta, 3 for Indonesia, 4 for Japan, 5 for Vietnam")
    area_choice = input("Enter choice: ")
    
    if area_choice == "1":
        area = "Jakarta"
        lon_min, lon_max = 106.53, 106.98
        lat_min, lat_max = -6.38, -5.53
        use_districts = input("Use actual district geometries for Jakarta? (y/n): ").lower() == 'y'
    elif area_choice == "2":
        area = "Yogyakarta"
        lon_min, lon_max = 109.6667, 111.0
        lat_min, lat_max = -8.5, -7.3333
        use_districts = False
    elif area_choice == "3":
        area = "Indonesia"
        lon_min, lon_max = 95.0, 141.0
        lat_min, lat_max = -11.0, 6.0
        use_districts = False
        use_land_boundaries = input("Use land boundaries to avoid sea areas? (y/n): ").lower() == 'y'
        if use_land_boundaries:
            geojson_path = input("Enter path to GeoJSON file for Indonesia land boundaries (e.g., path/to/id.json): ")
            land_geometry = load_country_land_geometry(geojson_path)
            if land_geometry is None:
                print("Failed to load land geometry. Exiting.")
                exit()
    elif area_choice == "4":
        area = "Japan"
        lon_min, lon_max = 122.938, 149.5
        lat_min, lat_max = 24.249, 45.523
        use_districts = False
        use_land_boundaries = input("Use land boundaries to avoid sea areas? (y/n): ").lower() == 'y'
        if use_land_boundaries:
            geojson_path = input("Enter path to GeoJSON file for Japan land boundaries (e.g., path/to/jp.json): ")
            land_geometry = load_country_land_geometry(geojson_path)
            if land_geometry is None:
                print("Failed to load land geometry. Exiting.")
                exit()
    elif area_choice == "5":
        area = "Vietnam"
        lon_min, lon_max = 102.144, 109.464
        lat_min, lat_max = 8.559, 23.393
        use_districts = False
        use_land_boundaries = input("Use land boundaries to avoid sea areas? (y/n): ").lower() == 'y'
        if use_land_boundaries:
            geojson_path = input("Enter path to GeoJSON file for Vietnam land boundaries (e.g., path/to/vietnam.json): ")
            land_geometry = load_country_land_geometry(geojson_path)
            if land_geometry is None:
                print("Failed to load land geometry. Exiting.")
                exit()
    else:
        print("Invalid choice")
        exit()
    
    if use_districts and area == "Jakarta":
        geojson_path = input("Enter path to GeoJSON file for Jakarta districts: ")
        df = generate_district_dataframe(cols, format_type, geojson_path)
        filename_prefix = f"jakarta_districts_data_{cols}c_{format_type.lower()}"
    else:
        rows = int(input("Enter number of rows: "))
        print("Choose geometry type: 1 for POINT, 2 for POLYGON, 3 for MULTIPOLYGON")
        geom_choice = input("Enter choice: ")
        geom_map = {"1": "POINT", "2": "POLYGON", "3": "MULTIPOLYGON"}
        geom_type = geom_map.get(geom_choice, "POINT")
        df = generate_random_dataframe(rows, cols, geom_type, format_type, lon_min, lon_max, lat_min, lat_max, land_geometry)
        filename_prefix = f"{area.lower()}_data_{rows}r_{cols}c_{geom_type.lower()}_{format_type.lower()}"
        if area in ["Indonesia", "Japan", "Vietnam"] and use_land_boundaries:
            filename_prefix += "_land_only"
    
    save_files(df, filename_prefix)