"""Dataset Generator Package

This package provides functionality for generating synthetic geospatial datasets
with various geometry types including points, polygons, and H3 cells.
"""

__version__ = "1.0.0"
__author__ = "zhafran-bvt"

# Import main functions for easier access
from .file_generator import (
    generate_parallel_dataframe,
    save_files_chunked,
    validate_generated_data,
    BOUNDING_BOXES,
    REALISTIC_LABELS,
    H3_AVAILABLE,
)

__all__ = [
    "generate_parallel_dataframe",
    "save_files_chunked", 
    "validate_generated_data",
    "BOUNDING_BOXES",
    "REALISTIC_LABELS",
    "H3_AVAILABLE",
]