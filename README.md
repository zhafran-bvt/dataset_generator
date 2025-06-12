Dataset Generator
This repository contains a Python script to generate synthetic geospatial and demographic datasets for various regions, including Jakarta, Yogyakarta, Indonesia, Japan, and Vietnam. The datasets can be generated in Well-Known Text (WKT) or GeoJSON formats and include random numerical and categorical data alongside geographic geometries, suitable for data analysis, visualization, or simulation purposes.
Features

Generates datasets with realistic demographic labels, such as population, birth rate, and income per capita.
Supports POINT, POLYGON, and MULTIPOLYGON geometries.
Allows use of actual district geometries for Jakarta via GeoJSON files.
Enables land boundary constraints to avoid generating points over sea areas for larger regions.
Outputs datasets in both CSV and Excel formats for easy integration with analysis tools.

Requirements
To run the script, ensure the following are installed:

Python 3.x
Required packages: pandas, geopandas, shapely, openpyxl

Install the dependencies using the following command:
pip install pandas geopandas shapely openpyxl

Usage
To execute the script locally, run:
python file_generator.py

The script will prompt for the following inputs:

Format: Choose 1 for WKT or 2 for GeoJSON.
Number of Columns: Enter a number up to 22, corresponding to available demographic labels.
Area: Select 1 for Jakarta, 2 for Yogyakarta, 3 for Indonesia, 4 for Japan, or 5 for Vietnam.
Jakarta District Geometries: For Jakarta, choose whether to use actual district geometries (y/n). If yes, provide the path to a GeoJSON file (e.g., geojson/jakarta_districts.json).
Land Boundaries: For Indonesia, Japan, or Vietnam, choose whether to use land boundaries to avoid sea areas (y/n). If yes, provide the path to a GeoJSON file (e.g., geojson/id.json).
Number of Rows: Specify the number of data rows to generate.
Geometry Type: Choose 1 for POINT, 2 for POLYGON, or 3 for MULTIPOLYGON.

Upon completion, the script generates CSV and Excel files with the dataset, saved in the current directory with names reflecting the configuration (e.g., indonesia_data_10r_22c_point_wkt.csv).
GeoJSON Files
The repository includes a geojson directory containing GeoJSON files for land boundaries of Indonesia (id.json), Japan (jp.json), and Vietnam (vn.json). For Jakarta, a GeoJSON file for district geometries is also provided (jakarta_districts.json). Ensure these files are present in the geojson directory when using district or land boundary options.
GitHub Actions Workflow
A GitHub Actions workflow is configured to automate dataset generation, allowing users to generate datasets without local execution. The workflow is defined in .github/workflows/generate_dataset.yml and can be triggered manually with customizable inputs.
Using the GitHub Actions Workflow

Navigate to the Actions tab in the repository at Dataset Generator.
Select the "Generate Dataset" workflow.
Click "Run workflow."
Provide the required inputs, mirroring the script’s prompts (format, columns, area, etc.).
Submit the workflow.
Upon completion, download the generated CSV and Excel files from the artifacts section.

Dataset Structure
The generated dataset includes the following columns:

id: A unique identifier for each row.
geom: The geometric data in WKT or GeoJSON format.
Additional columns corresponding to selected demographic labels, such as "Population," "Birth Rate," or "Income per Capita."

Each row represents a geographic entity with associated demographic data, suitable for geospatial analysis or visualization.
