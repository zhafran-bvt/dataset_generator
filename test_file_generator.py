import unittest
import os
import sys
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, Polygon
from unittest.mock import patch, MagicMock
import tempfile
import json
from datetime import datetime

# Import the generator module
import file_generator as fg

class TestFileGenerator(unittest.TestCase):
    def setUp(self):
        self.test_polygon = Polygon([
            (100.0, 0.0), (101.0, 0.0), (101.0, 1.0), (100.0, 1.0), (100.0, 0.0)
        ])
        self.temp_dir = tempfile.TemporaryDirectory()
        self.geojson_path = os.path.join(self.temp_dir.name, "test.geojson")
        with open(self.geojson_path, 'w') as f:
            json.dump({
                "type": "FeatureCollection",
                "features": [{
                    "type": "Feature",
                    "properties": {},
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[[100.0, 0.0], [101.0, 0.0], [101.0, 1.0], [100.0, 1.0], [100.0, 0.0]]]
                    }
                }]
            }, f)
        self.config_path = os.path.join(self.temp_dir.name, "test_config.json")
        with open(self.config_path, 'w') as f:
            json.dump({
                "format_choice": 1,
                "include_demographic": "yes",
                "include_economic": "no",
                "num_columns": 9,
                "use_spatial_clustering": "no",
                "area_choice": 2,
                "geojson_path": "",
                "num_rows": 10,
                "use_chunking": "no",
                "geometry_type": 1
            }, f)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_generate_random_datetime(self):
        for _ in range(10):
            dt_str = fg.generate_random_datetime()
            dt = datetime.strptime(dt_str, "%Y-%m-%dT%H:%M:%SZ")
            self.assertEqual(dt.year, 2025)

    def test_normalize_denormalize_value(self):
        min_val, max_val = 10, 50
        value = 30
        norm = fg.normalize_value(value, min_val, max_val)
        self.assertTrue(0 <= norm <= 1)
        denorm = fg.denormalize_value(norm, min_val, max_val)
        self.assertAlmostEqual(value, denorm)

    def test_random_point_in_geometry(self):
        for _ in range(5):
            pt = fg.random_point_in_geometry(self.test_polygon)
            self.assertTrue(self.test_polygon.contains(pt))
            self.assertIsInstance(pt, Point)

    @patch('geopandas.read_file')
    def test_load_land_geometry_mocked(self, mock_read_file):
        mock_gdf = gpd.GeoDataFrame({'geometry': [self.test_polygon]})
        mock_read_file.return_value = mock_gdf
        geometry = fg.load_land_geometry('geoJson/jkt.geojson')
        self.assertIsNotNone(geometry)
        self.assertTrue(geometry.is_valid)
        self.assertFalse(geometry.is_empty)

    def test_generate_random_geom_point_wkt(self):
        geom = fg.generate_random_geom("POINT", "WKT", 100.0, 101.0, 0.0, 1.0)
        self.assertTrue(geom.startswith("POINT ("))
        coords = geom.replace("POINT (", "").replace(")", "").split()
        lon, lat = float(coords[0]), float(coords[1])
        self.assertTrue(100.0 <= lon <= 101.0)
        self.assertTrue(0.0 <= lat <= 1.0)

    def test_generate_random_geom_point_geojson(self):
        geom = fg.generate_random_geom("POINT", "GeoJSON", 100.0, 101.0, 0.0, 1.0)
        geojson = json.loads(geom)
        self.assertEqual(geojson["type"], "Point")
        lon, lat = geojson["coordinates"]
        self.assertTrue(100.0 <= lon <= 101.0)
        self.assertTrue(0.0 <= lat <= 1.0)

    def test_generate_random_geom_with_land_geometry(self):
        geom = fg.generate_random_geom("POINT", "WKT", 99.0, 102.0, -1.0, 2.0, self.test_polygon)
        coords = geom.replace("POINT (", "").replace(")", "").split()
        lon, lat = float(coords[0]), float(coords[1])
        pt = Point(lon, lat)
        self.assertTrue(self.test_polygon.contains(pt))

    def test_generate_random_geom_with_spatial_clusters(self):
        class DummyKDE:
            def resample(self, n):
                return np.array([[100.5, 0.5]] * n).T
        spatial_clusters = {
            'model': DummyKDE(),
            'use_probability': 1.0,
            'centers': [(100.5, 0.5)]
        }
        geom = fg.generate_random_geom("POINT", "WKT", 100.0, 101.0, 0.0, 1.0, land_geometry=None, spatial_clusters=spatial_clusters)
        coords = geom.replace("POINT (", "").replace(")", "").split()
        lon, lat = float(coords[0]), float(coords[1])
        self.assertAlmostEqual(lon, 100.5)
        self.assertAlmostEqual(lat, 0.5)

    def test_create_spatial_clustering_model(self):
        model = fg.create_spatial_clustering_model(100.0, 101.0, 0.0, 1.0, self.test_polygon, cluster_count=3, points_per_cluster=100)
        self.assertIn('model', model)
        self.assertIn('centers', model)
        samples = model['model'].resample(10).T
        self.assertEqual(samples.shape[1], 2)

    def test_generate_correlated_values(self):
        labels = ["Income per Capita", "Education Level", "Poverty Rate"]
        correlation_engine = fg.CorrelationEngine(fg.VARIABLE_CORRELATIONS, fg.VALUE_RANGES)
        values = correlation_engine.generate_values(labels, seed=42)
        self.assertIn("Income per Capita", values)
        self.assertIn("Poverty Rate", values)
        self.assertIn("Education Level", values)

    def test_correlation_directions(self):
        correlation_engine = fg.CorrelationEngine(fg.VARIABLE_CORRELATIONS, fg.VALUE_RANGES)
        income_high, poverty_low = 0, 0
        for i in range(20):
            vals = correlation_engine.generate_values(["Income per Capita", "Poverty Rate", "Education Level"], seed=i)
            income_median = (fg.VALUE_RANGES["Income per Capita"][0] + fg.VALUE_RANGES["Income per Capita"][1]) / 2
            if vals["Income per Capita"] > income_median:
                income_high += 1
                poverty_median = (fg.VALUE_RANGES["Poverty Rate"][0] + fg.VALUE_RANGES["Poverty Rate"][1]) / 2
                if vals["Poverty Rate"] < poverty_median:
                    poverty_low += 1
        self.assertTrue(poverty_low >= 0.6 * income_high)

    @patch('file_generator.generate_random_geom')
    def test_generate_batch_rows(self, mock_generate_random_geom):
        mock_generate_random_geom.side_effect = [f"POINT ({100.5 + i * 0.01} 0.5)" for i in range(5)]
        correlation_engine = fg.CorrelationEngine(fg.VARIABLE_CORRELATIONS, fg.VALUE_RANGES)
        batch_params = (
            0, 5, "POINT", "WKT", 100.0, 101.0, 0.0, 1.0, self.test_polygon,
            ["id", "geom", "date_created", "Population"], False, False, None, correlation_engine
        )
        rows = fg.generate_batch_rows(batch_params)
        self.assertEqual(len(rows), 5)
        for i, row in enumerate(rows):
            self.assertEqual(row["id"], i + 1)
            self.assertIn("date_created", row)
            self.assertIn("Population", row)

    def test_generate_batch_rows_demographic(self):
        correlation_engine = fg.CorrelationEngine(fg.VARIABLE_CORRELATIONS, fg.VALUE_RANGES)
        batch_params = (
            0, 5, "POINT", "WKT", 100.0, 101.0, 0.0, 1.0, None,
            ["id", "geom", "date_created", "Gender", "Occupation", "Education Level"], True, False, None, correlation_engine
        )
        rows = fg.generate_batch_rows(batch_params)
        self.assertEqual(len(rows), 5)
        for row in rows:
            self.assertIn(row["Gender"], fg.GENDER_OPTIONS)
            self.assertIn(row["Occupation"], fg.OCCUPATION_OPTIONS)
            self.assertIn(row["Education Level"], fg.EDUCATION_OPTIONS)

    def test_generate_batch_rows_economic(self):
        correlation_engine = fg.CorrelationEngine(fg.VARIABLE_CORRELATIONS, fg.VALUE_RANGES)
        batch_params = (
            0, 5, "POINT", "WKT", 100.0, 101.0, 0.0, 1.0, None,
            ["id", "geom", "date_created", "Household Income", "Employment Status", "Access to Healthcare"], False, True, None, correlation_engine
        )
        rows = fg.generate_batch_rows(batch_params)
        self.assertEqual(len(rows), 5)
        for row in rows:
            self.assertTrue(0 <= row["Household Income"] <= 1000000)
            self.assertIn(row["Employment Status"], fg.EMPLOYMENT_STATUS_OPTIONS)
            self.assertIn(row["Access to Healthcare"], fg.HEALTHCARE_ACCESS_OPTIONS)

    def test_save_files_chunked_large_dataset(self):
        df = pd.DataFrame({"id": range(150000), "value": range(150000)})
        prefix = "test_large"
        fg.save_files_chunked(df, prefix, chunk_size=50000)
        self.assertTrue(os.path.exists(os.path.join("output", f"{prefix}_part1.csv")))
        self.assertTrue(os.path.exists(os.path.join("output", f"{prefix}_part2.csv")))
        self.assertTrue(os.path.exists(os.path.join("output", f"{prefix}_part3.csv")))
        # Clean up: Remove files first, then directory
        for i in range(1, 4):
            file_path = os.path.join("output", f"{prefix}_part{i}.csv")
            if os.path.exists(file_path):
                os.remove(file_path)
        # Remove the output directory if it exists and is empty
        if os.path.exists("output") and not os.listdir("output"):
            os.rmdir("output")

    def test_analyze_correlations(self):
        df = pd.DataFrame({
            "Income per Capita": [10000, 20000, 30000, 40000, 50000],
            "Poverty Rate": [40, 30, 20, 10, 5],
            "Education Level": [2, 4, 6, 8, 10]
        })
        fg.analyze_correlations(df)  # Should log, passes if no exception

    def test_load_config(self):
        config = fg.load_config(self.config_path)
        self.assertEqual(config['format_choice'], 1)
        self.assertEqual(config['include_demographic'], 'yes')
        self.assertEqual(config['num_rows'], 10)

    @patch('builtins.input')
    def test_validate_input(self, mock_input):
        mock_input.side_effect = ['1', 'yes', '10']
        result = fg.validate_input("Test prompt: ", [1, 2], int)
        self.assertEqual(result, 1)
        result = fg.validate_input("Test prompt: ", ['yes', 'no'])
        self.assertEqual(result, 'yes')
        result = fg.validate_input("Test prompt: ", None, int)
        self.assertEqual(result, 10)

if __name__ == '__main__':
    unittest.main()