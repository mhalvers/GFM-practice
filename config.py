from pathlib import Path

DATA_DIR = Path.home() / "data_science/geospatial/mpls_land_use/data"

# Minneapolis-St. Paul metro area bounding box
BBOX = [-94.22608230, 44.53677921, -92.51781747, 45.36347663]

# max cloud cover percentage for searches
MAX_CLOUD_COVER = 1  # in percent

# dictionary of bands download keyed by collection
BANDS = {
    "hls2-s30": [
        "B02",        # Blue
        "B03",        # Green
        "B04",        # Red
        "B8A",        # NIR Narrow
        "B11",        # SWIR1
        "B12",        # SWIR2
        "thumbnail"
    ],
    "hls2-l30": [
        "B02",        # Blue
        "B03",        # Green
        "B04",        # Red
        "B05",        # NIR Narrow (called B8A in Sentinel)
        "B06",        # SWIR1
        "B07",        # SWIR2
        "thumbnail"
    ]
}

COLLECTIONS = ["hls2-s30", "hls2-l30"]  # Sentinel-2 and Landsat harmonized collections
PC_CATALOG = "https://planetarycomputer.microsoft.com/api/stac/v1"