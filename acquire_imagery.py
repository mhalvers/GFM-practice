import random

from pystac_client import Client
import planetary_computer as pc

from helpers import download_items
from config import BANDS, BBOX, MAX_CLOUD_COVER, PC_CATALOG, COLLECTIONS

TEST = True  # if True, limit downloads for testing
DOWNLOAD = True

IMAGERY_DIR = "data"

if TEST:
    BBOX = [-93.5, 45, -93.3, 45.2]  # Smaller bbox for testing


def main():
    """Main execution function."""

    print("=" * 70)
    print("HARMONIZED SENTINEL-2/LANDSAT IMAGERY DOWNLOAD")
    print("=" * 70)

    print(f"  Area of Interest (bbox): {BBOX}")
    print(f"  Cloud cover threshold: <= {MAX_CLOUD_COVER}%")
    print(f"  Collections: {COLLECTIONS}")
    print(f"  Requested Landsat Harmonized Bands: {BANDS['hls2-l30']}")
    print(f"  Requested Sentinel-2 Harmonized Bands: {BANDS['hls2-s30']}")
    print(f"  Latitude: {BBOX[1]} to {BBOX[3]}")
    print(f"  Longitude: {BBOX[0]} to {BBOX[2]}\n")

    # =========================================================================
    # 1. CONNECT TO PLANETARY COMPUTER STAC CATALOG
    # =========================================================================

    if DOWNLOAD:
        print("Connecting to Planetary Computer STAC Catalog...")

        try:
            # Create client
            catalog = Client.open(PC_CATALOG,
                                modifier=pc.sign_inplace,)
            print(f"  ✓ Connected to: {PC_CATALOG}\n")
        except Exception as e:
            print(f"  ✗ Failed to connect to Planetary Computer: {e}\n")
            return

        # =========================================================================
        # SEARCH FOR HARMONIZED IMAGERY
        # =========================================================================
        print("Searching for harmonized Sentinel-2/Landsat imagery...\n")

        items = {collection: None for collection in COLLECTIONS}
        for collection in COLLECTIONS:
            print(f"  Querying {collection}...")
            search_kwargs = {
                    "collections": collection,
                    "bbox": BBOX,
                    "query": {"eo:cloud_cover": {"lte": MAX_CLOUD_COVER}}
                }

            search = catalog.search(**search_kwargs)
            items[collection] = list(search.items())


        # =========================================================================
        # RESULTS SUMMARY
        # =========================================================================
        print("Query Results Summary")
        print("=" * 70)
        print("Region: Minneapolis-St. Paul Twin Cities")
        print(f"Bounding Box: {BBOX}")
        print(f"Cloud Cover Threshold: <= {MAX_CLOUD_COVER}%")
        print(f"Collections: {COLLECTIONS}")
        print(f"\nTotal Landsat items available: {len(items['hls2-l30'])}")
        print(f"Total Sentinel-2 items available: {len(items['hls2-s30'])}")
        print("=" * 70)

        # =========================================================================
        # DOWNLOAD IMAGERY
        # =========================================================================
        
        if TEST:
            print("\nTEST MODE: Limiting downloads to 3 items per collection/mission\n")
            random_integer = random.randint(0, min(len(items['hls2-l30']), len(items['hls2-s30']))-3)
            idx = slice(random_integer, random_integer + 3)
            items = {collection: items[collection][idx] for collection in COLLECTIONS}

        if items:
            print("\nDownloading harmonized imagery...")
            stats = download_items(items, IMAGERY_DIR, band_names=BANDS)

            print("\nDownload Summary:")
            print("=" * 70)
            for k, v in stats.items():
                print(f"  {k.replace('_', ' ').title()}: {v}")
            print("=" * 70)
        else:
            print("\nNo items to download")


if __name__ == "__main__":
    main()
