# 🗂️ Dataset Generator

This repository provides a Python script (`file_generator.py`) to generate **synthetic geospatial and demographic datasets** for regions including Jakarta, Yogyakarta, Indonesia, Japan, and Vietnam.

It supports geometry types in **Well-Known Text (WKT)** or **GeoJSON** formats and produces **realistic, randomized, and correlated** data suitable for simulation, visualization, or testing geospatial applications. A test suite (`test_file_generator.py`) is included to ensure reliability.

---

## 📌 Features

- Realistic, correlated labels such as:
  - Population, Birth Rate, Income per Capita, Health Index, Literacy Rate, etc.
- **Optional Demographic Columns**: Gender, Occupation, Education Level  
- **Optional Economic Columns**: Household Income, Employment Status, Access to Healthcare
- Supports geometry types:
  - `POINT`, `POLYGON`, `MULTIPOLYGON`, `H3` (Hexagonal Hierarchical Spatial Index)
- Optional spatial clustering for realistic distributions
- Optional use of real **GeoJSON boundaries** to avoid placing geometries in oceans
- **Output formats**:
  - CSV
  - Excel (XLSX) – limited to 1 million rows
- Chunked output support for large datasets
- Unit tests for validating dataset logic

---

## ⚙️ Requirements

Ensure the following are installed:

```bash
pip install pandas geopandas shapely openpyxl tqdm numpy scipy h3
```

Supports **Python 3.x**

**Note**: The `h3` library is required for H3 hexagonal spatial indexing support. If you don't plan to use H3 geometry type, you can skip installing it.

---

## 🚀 Usage

Run the generator:

```bash
python file_generator.py
```

### You will be prompted to input:

- **Format**:  
  `1 = WKT`, `2 = GeoJSON`

- **Include Demographic Columns**:  
  `yes` / `no`

- **Include Economic Columns**:  
  `yes` / `no`

- **Number of Columns**:  
  Minimum `3` (or `6/9` with demographic/economic), up to `29`

- **Use Spatial Clustering**:  
  `yes` / `no`

- **Area**:  
  ```
  1 = Jakarta  
  2 = Yogyakarta  
  3 = Indonesia  
  4 = Japan  
  5 = Vietnam
  ```

- **GeoJSON Path** *(for areas 1, 3, 4, 5)*:  
  e.g., `geojson/id.json`

- **Number of Rows**:  
  Total number of data rows

- **Use Chunked Output** *(for >100,000 rows)*:  
  `yes` / `no`

- **Geometry Type**:  
  ```
  1 = POINT  
  2 = POLYGON  
  3 = MULTIPOLYGON
  4 = H3 (requires h3 library)
  ```

- **H3 Resolution** *(if H3 geometry type is selected)*:  
  Resolution level from `0` (largest hexagons, ~4,250 km²) to `15` (smallest hexagons, ~0.9 m²)  
  Default: `9` (~0.1 km², approximately 174 meters across)  
  
  **Recommended values**:
  - Resolution 5: ~252 km² per hexagon (regional analysis)
  - Resolution 9: ~0.1 km² per hexagon (city-level analysis, default)
  - Resolution 12: ~307 m² per hexagon (neighborhood analysis)

### Example: Generating H3 Dataset

To generate a dataset with H3 hexagonal cells for Jakarta at resolution 9:

```bash
python file_generator.py
```

When prompted:
- Format: `1` (WKT) or `2` (GeoJSON)
- Include Demographic: `yes` or `no`
- Include Economic: `yes` or `no`
- Number of Columns: e.g., `10`
- Use Spatial Clustering: `yes` or `no`
- Area: `1` (Jakarta)
- GeoJSON Path: `geoJson/jkt.geojson`
- Number of Rows: e.g., `1000`
- Use Chunked Output: `no`
- Geometry Type: `4` (H3)
- H3 Resolution: `9` (or press Enter for default)

This will generate files like:
```text
output/jakarta_data_1000r_10c_h3_res9.csv
output/jakarta_data_1000r_10c_h3_res9.xlsx
```

Each row will contain an H3 cell identifier (e.g., `8928308280fffff`) in the `geom` column instead of traditional point/polygon geometries.

### Example Outputs:

```text
output/indonesia_data_100r_20c_point_wkt.csv  
output/indonesia_data_100r_20c_point_wkt.xlsx  
output/indonesia_data_100r_20c_h3_res9.csv
output/indonesia_data_100r_20c_h3_res9.xlsx
output/indonesia_data_1000000r_20c_point_wkt_part1.csv
output/indonesia_data_1000000r_20c_point_wkt_part2.csv
```

---

## 📂 GeoJSON Directory

Make sure the following files are in `geojson/`:

- `id.json` – Indonesia
- `jp.json` – Japan
- `vn.json` – Vietnam
- `jakarta_districts.json` – Jakarta districts
- Optional: Minified versions like `id-jk.min.geojson`

---

## 🧪 Dataset Columns

### Always included:
- `id`: Unique identifier  
- `geom`: Geometry (WKT, GeoJSON, or H3 cell identifier)  
- `date_created`: Random datetime in 2025 (ISO 8601)

**Note**: When using H3 geometry type, the `geom` column contains H3 cell identifiers as hexadecimal strings (e.g., `8928308280fffff`) instead of traditional WKT or GeoJSON geometries.

### Optional Demographic:
- `Gender`: Male, Female, Other  
- `Occupation`: Employed, Unemployed, Student, etc.  
- `Education Level`: High School, Bachelor’s, etc.

### Optional Economic:
- `Household Income`: 0–1,000,000  
- `Employment Status`: Full-time, Self-employed, etc.  
- `Access to Healthcare`: True / False

### Additional Columns (up to 20):
- `Population`, `Birth Rate`, `Death Rate`, `Unemployment Rate`, `Income per Capita`  
- `GDP Growth`, `Health Index`, `Urbanization Rate`, `Poverty Rate`, `Energy Consumption`, etc.

> Data values are intelligently randomized with realistic ranges and correlations.

---

## 🧪 Testing

Run the built-in unit tests:

```bash
python -m unittest test_file_generator.py -v
```

Tests cover:
- Date generation
- Geometry creation (all types)
- Clustering logic
- Correlation handling
- File writing and chunking

---

## 🤖 GitHub Actions Workflow

This repo includes a GitHub Actions workflow to automate dataset generation.

**Workflow File**: `.github/workflows/generate_dataset.yaml`

### Trigger Steps:

1. Go to the **Actions** tab
2. Select **Generate Dataset**
3. Click **Run workflow**
4. Fill in the inputs:

```yaml
format_choice: 1                  # 1=WKT, 2=GeoJSON
include_demographic: no           # yes/no
include_economic: no              # yes/no
num_columns: 9                    # 3–29
use_spatial_clustering: no        # yes/no
area_choice: 3                    # 1=Jakarta, ..., 5=Vietnam
geojson_path: geojson/id.json     # required if area ≠ 2
num_rows: 10                      # total rows
use_chunking: yes                 # yes/no
geometry_type: 1                  # 1=POINT, 2=POLYGON, 3=MULTIPOLYGON
```

### What It Does:

- Checks out the repo
- Installs Python dependencies
- Runs unit tests
- Parses workflow inputs
- Executes `file_generator.py`
- Uploads generated `.csv` and `.xlsx` files as artifacts

---

## ✅ Output Example

**POINT Geometry:**

| id | geom | date_created | Population | Income per Capita | Literacy Rate | Gender | Household Income |
|----|------|---------------|------------|--------------------|----------------|--------|-------------------|
| 1  | POINT(106.82 -6.17) | 2025-06-17T13:00:00Z | 54321 | 25000.50 | 85.0 | Male | 450000.75 |

**H3 Geometry:**

| id | geom | date_created | Population | Income per Capita | Literacy Rate | Gender | Household Income |
|----|------|---------------|------------|--------------------|----------------|--------|-------------------|
| 1  | 8928308280fffff | 2025-06-17T13:00:00Z | 54321 | 25000.50 | 85.0 | Male | 450000.75 |

---

## 📬 Contributing

Contributions are welcome!

- Fork the repo
- Create a new branch
- Submit a PR
- Ensure your changes pass all tests (`test_file_generator.py`)

Feel free to open issues to request features or report bugs.

---
