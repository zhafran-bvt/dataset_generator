# Dataset Generator

This repository provides a Python script (`file_generator.py`) to generate synthetic geospatial and demographic datasets for regions including Jakarta, Yogyakarta, Indonesia, Japan, and Vietnam. It supports geometry types in Well-Known Text (WKT) or GeoJSON formats and produces realistic, randomized data columns suitable for simulation, visualization, or testing geospatial applications.

---

## 📌 Features

- Generates data with realistic labels like:
  - *Population*, *Migration*, *Mortality*, *Income per Capita*, etc.
- Geometry types supported:
  - `POINT`, `POLYGON`, and `MULTIPOLYGON`
- Optional use of actual **Jakarta district boundaries** via GeoJSON
- Optional filtering using **land boundary GeoJSONs** to avoid placing points in oceans (for Indonesia, Japan, Vietnam)
- Output formats:
  - **CSV**
  - **Excel (XLSX)**

---

## ⚙️ Requirements

Make sure the following are installed:

- Python 3.x
- Required packages:
  ```bash
  pip install pandas geopandas shapely openpyxl tqdm
  ```

---

## 🚀 Usage

Run the script using:

```bash
python file_generator.py
```

You’ll be prompted to input:

1. **Format**:  
   `1` = WKT, `2` = GeoJSON

2. **Number of Columns**:  
   Up to 20 (limited by available labels)

3. **Area**:  
   - `1` = Jakarta  
   - `2` = Yogyakarta  
   - `3` = Indonesia  
   - `4` = Japan  
   - `5` = Vietnam  

4. **GeoJSON Path**:  
   If using land boundaries or Jakarta districts, provide the corresponding GeoJSON file (e.g., `geojson/id.json`)

5. **Number of Rows**:  
   Total data rows to generate

6. **Geometry Type**:  
   - `1` = POINT  
   - `2` = POLYGON  
   - `3` = MULTIPOLYGON

After execution, output files will be saved in the current directory with filenames like:

```
indonesia_data_100r_20c_point_wkt.csv  
indonesia_data_100r_20c_point_wkt.xlsx
```

---

## 📂 GeoJSON Directory

Ensure the following files exist under the `geojson/` directory:

- `id.json` – Land boundaries of Indonesia  
- `jp.json` – Land boundaries of Japan  
- `vn.json` – Land boundaries of Vietnam  
- `jakarta_districts.json` – District geometries for Jakarta  
- Optionally compressed versions like `id-jk.min.geojson`

---

## 🧪 Dataset Columns

Each dataset includes:

- `id`: Unique identifier
- `geom`: Geometry (WKT or GeoJSON)
- Additional demographic columns (max 20) selected from:

```text
Family Identity Card, Migration, Mortality, People Density,
Population, Registered Residents, Birth Rate, Death Rate,
Unemployment Rate, Income per Capita, Households, Literacy Rate,
School Enrollment, Employment Rate, Water Access, Electricity Access,
Health Facilities, Internet Access, Average Age, Vehicle Ownership
```

Values are intelligently randomized:
- Rates/percentages (0–100) for columns like `Literacy Rate`
- Integer counts for `Population`, `Mortality`, etc.
- Random floats where appropriate

---

## 🤖 GitHub Actions Workflow

Automate dataset creation via GitHub Actions.

**Workflow File:** `.github/workflows/generate_dataset.yaml`

### How to Trigger:

1. Go to the **Actions** tab
2. Select **Generate Dataset**
3. Click **Run workflow**
4. Fill out inputs:

```yaml
format_choice: 1                  # 1=WKT, 2=GeoJSON
num_columns: 22                  # up to 22 columns
area_choice: 3                   # 1=Jakarta, 2=Yogyakarta, 3=Indonesia, etc.
geojson_path: geojson/id.json    # required for area 1, 3, 4, 5
num_rows: 10                     # number of rows
geometry_type: 1                 # 1=POINT, 2=POLYGON, 3=MULTIPOLYGON
```

### Workflow Logic:

```yaml
- Setup Python and dependencies
- Prepare input file from dispatch inputs
- Validate required fields based on area
- Pipe the inputs into `file_generator.py`
- Upload generated `.csv` and `.xlsx` as artifacts
```

---

## ✅ Output Example

| id | geom | Population | Birth Rate | Internet Access | ... |
|----|------|------------|------------|------------------|-----|
| 1  | POINT(...) | 54321 | 2.5 | 75.0 | ... |

---

## 📬 Contributing

Fork, create a branch, and submit PRs or open issues to improve the tool.
