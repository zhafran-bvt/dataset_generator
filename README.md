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

4. **Use Jakarta District Geometry** (for Jakarta):  
   - `y` or `n`  
   - If yes, input path to a GeoJSON file (e.g., `geojson/jakarta_districts.json`)

5. **Use Land Boundaries** (for Indonesia/Japan/Vietnam):  
   - `y` or `n`  
   - If yes, input path to the GeoJSON file (e.g., `geojson/id.json`)

6. **Number of Rows**:  
   Total data rows to generate

7. **Geometry Type**:  
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

Make sure the following files exist under the `geojson/` directory:

- `id.json` – Land boundaries of Indonesia  
- `jp.json` – Land boundaries of Japan  
- `vn.json` – Land boundaries of Vietnam  
- `jakarta_districts.json` – Detailed district geometries for Jakarta

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

## 🤖 GitHub Actions Workflow (Optional)

Automate dataset creation via GitHub Actions:

**Workflow File:** `.github/workflows/generate_dataset.yml`

### How to Trigger:

1. Go to **Actions** tab on GitHub
2. Choose **Generate Dataset**
3. Click **Run workflow**
4. Fill out form inputs (format, region, columns, etc.)
5. Submit and download results from **Artifacts**

---

## ✅ Output Example

| id | geom | Population | Birth Rate | Internet Access | ... |
|----|------|------------|------------|------------------|-----|
| 1  | POINT(...) | 54321 | 2.5 | 75.0 | ... |

---

## 📄 License

This project is released under the MIT License.

---

## 📬 Contributing

Feel free to fork, submit PRs, or raise issues to improve or extend functionality.
