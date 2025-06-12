# Dataset Generator

This repository provides a Python script to generate synthetic geospatial and demographic datasets for regions including Jakarta, Yogyakarta, Indonesia, Japan, and Vietnam. 
The datasets are suitable for data analysis, visualization, and simulation, and can be exported in Well-Known Text (WKT) or GeoJSON formats.

## Features

- Generates datasets with realistic demographic labels (e.g., *Population*, *Birth Rate*, *Income per Capita*).
- Supports `POINT`, `POLYGON`, and `MULTIPOLYGON` geometries.
- Option to use actual district geometries for Jakarta via GeoJSON files.
- Supports land-boundary filtering to prevent generating points over sea areas.
- Outputs in both **CSV** and **Excel** formats for compatibility with analysis tools.

## Requirements

Ensure you have Python 3.x installed. Install dependencies via pip:

```bash
pip install pandas geopandas shapely openpyxl
