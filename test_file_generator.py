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

# Import your generator module
import file_generator as fg

class TestFileGenerator(unittest.TestCase):
    def setUp(self):
        # Simple square polygon for geo tests
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

    def test_load_land_geometry(self):
        geometry = fg.load_land_geometry(self.geojson_path)
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
        # Make a fake KDE with resample method
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
        model = fg.create_spatial_clustering_model(100.0, 101.0, 0.0, 1.0, self.test_polygon)
        self.assertIn('model', model)
        self.assertIn('centers', model)
        # ensure the model can generate samples
        samples = model['model'].resample(10).T
        self.assertEqual(samples.shape[1], 2)

    def test_generate_correlated_values(self):
        labels = ["Income per Capita", "Education Level", "Poverty Rate"]
        values = fg.generate_correlated_values(labels, seed=42)
        self.assertIn("Income per Capita", values)
        self.assertIn("Poverty Rate", values)
        self.assertIn("Education Level", values)

    def test_correlation_directions(self):
        income_high, poverty_low = 0, 0
        for i in range(20):
            vals = fg.generate_correlated_values(["Income per Capita", "Poverty Rate", "Education Level"], seed=i)
            income_median = (fg.VALUE_RANGES["Income per Capita"][0] + fg.VALUE_RANGES["Income per Capita"][1]) / 2
            if vals["Income per Capita"] > income_median:
                income_high += 1
                poverty_median = (fg.VALUE_RANGES["Poverty Rate"][0] + fg.VALUE_RANGES["Poverty Rate"][1]) / 2
                if vals["Poverty Rate"] < poverty_median:
                    poverty_low += 1
        self.assertTrue(poverty_low >= 0.6 * income_high)

    @patch('file_generator.generate_random_geom')
    def test_generate_batch_rows(self, mock_generate_random_geom):
        # Return a different geometry string each call to ensure uniqueness
        mock_generate_random_geom.side_effect = [
            f"POINT ({100.5 + i * 0.01} 0.5)" for i in range(5)
        ]
        batch_params = (
            0, 5, "POINT", "WKT", 100.0, 101.0, 0.0, 1.0, self.test_polygon,
            ["id", "geom", "date_created", "Population"], False, False, None
        )
        rows = fg.generate_batch_rows(batch_params)
        self.assertEqual(len(rows), 5)
        for i, row in enumerate(rows):
            self.assertEqual(row["id"], i + 1)
            self.assertIn("date_created", row)
            self.assertIn("Population", row)

    def test_analyze_correlations(self):
        df = pd.DataFrame({
            "Income per Capita": [10000, 20000, 30000, 40000, 50000],
            "Poverty Rate": [40, 30, 20, 10, 5],
            "Education Level": [2, 4, 6, 8, 10]
        })
        fg.analyze_correlations(df)  # Should print, test passes if no exception is thrown

    @patch('builtins.open')
    @patch('os.path.getsize')
    @patch('pandas.DataFrame.to_csv')
    @patch('pandas.DataFrame.to_excel')
    def test_save_files_chunked(self, mock_to_excel, mock_to_csv, mock_getsize, mock_open):
        mock_getsize.return_value = 1024
        df = pd.DataFrame({"id": range(1, 101), "value": range(100)})
        fg.save_files_chunked(df, "test_output", chunk_size=10)
        self.assertEqual(mock_to_csv.call_count, 10)

if __name__ == '__main__':
    unittest.main()