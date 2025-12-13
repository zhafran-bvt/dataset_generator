#!/usr/bin/env python3
"""
Script to enhance H3 dataset with additional columns:
- INT column with random integer values
- Longitude and Latitude columns extracted from H3 cell centers
"""

import pandas as pd
import numpy as np
import h3


def enhance_h3_dataset(input_file, output_file):
    """
    Enhance H3 dataset with additional columns

    Args:
        input_file: Path to input CSV file
        output_file: Path to output CSV file
    """
    # Read the dataset
    print(f"Reading dataset from {input_file}...")
    df = pd.read_csv(input_file)

    print(f"Original dataset has {len(df)} rows and {len(df.columns)} columns")

    # Add INT column with random values (e.g., building count or population density index)
    print("Adding 'building_count' column (INT)...")
    np.random.seed(42)  # For reproducibility
    df["building_count"] = np.random.randint(1, 500, size=len(df))

    # Extract longitude and latitude from H3 cells
    print("Extracting longitude and latitude from H3 cells...")
    latitudes = []
    longitudes = []

    for h3_cell in df["geom"]:
        try:
            # Get the center coordinates of the H3 cell
            lat, lon = h3.cell_to_latlng(h3_cell)
            latitudes.append(lat)
            longitudes.append(lon)
        except Exception as e:
            print(f"Warning: Could not convert H3 cell {h3_cell}: {e}")
            latitudes.append(None)
            longitudes.append(None)

    # Add the coordinate columns
    df["latitude"] = latitudes
    df["longitude"] = longitudes

    # Reorder columns to put coordinates after geom
    cols = df.columns.tolist()
    # Remove latitude and longitude from their current position
    cols.remove("latitude")
    cols.remove("longitude")
    # Find the position of 'geom' column
    geom_idx = cols.index("geom")
    # Insert latitude and longitude right after geom
    cols.insert(geom_idx + 1, "latitude")
    cols.insert(geom_idx + 2, "longitude")
    # Reorder the dataframe
    df = df[cols]

    # Save the enhanced dataset
    print(f"Saving enhanced dataset to {output_file}...")
    df.to_csv(output_file, index=False)

    # Also save as Excel
    excel_file = output_file.replace(".csv", ".xlsx")
    df.to_excel(excel_file, index=False)

    print(f"\nEnhanced dataset summary:")
    print(f"- Total rows: {len(df)}")
    print(f"- Total columns: {len(df.columns)}")
    print(
        f"- New columns added: building_count (INT), latitude (FLOAT), longitude (FLOAT)"
    )
    print(f"\nFiles saved:")
    print(f"- CSV: {output_file}")
    print(f"- Excel: {excel_file}")

    # Show sample data
    print(f"\nSample data (first 5 rows):")
    print(df[["id", "geom", "latitude", "longitude", "building_count"]].head())

    # Show data types
    print(f"\nData types for new columns:")
    print(f"- building_count: {df['building_count'].dtype}")
    print(f"- latitude: {df['latitude'].dtype}")
    print(f"- longitude: {df['longitude'].dtype}")

    return df


if __name__ == "__main__":
    input_file = "output/yogyakarta_data_1000r_15c_h3_res9.csv"
    output_file = "output/yogyakarta_data_1000r_15c_h3_res9_enhanced.csv"

    enhance_h3_dataset(input_file, output_file)
