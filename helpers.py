import requests
from pathlib import Path
from typing import List, Dict

from pystac import Item


def download_items(
    items_dict: Dict[str, List[Item]],
    collection_name: str,
    output_dir: None | str = None,
    band_names: Dict[str, List[str]] = None,
) -> Dict[str, int]:
    """
    Download specified bands from STAC items to local directory.

    Args:
        items_dict: Dictionary of STAC items to download, keyed by collection name
        collection_name: Name of collection (e.g., 'sentinel-2-l2a', 'landsat-c2-l2')
        output_dir: Root output directory for downloads
        band_names: List of band names to download (e.g., ['B02', 'B03', 'B04', 'B08', 'B11', 'B12'] for S2)
                   If None, downloads all available bands

    Returns:
        Dictionary with download statistics
    """
    stats = {
        "total_items": sum(len(items) for items in items_dict.values()),
        "downloaded": 0,
        "failed": 0,
        "skipped": 0,
        "files_downloaded": 0,
    }

    # Create output directory structure
    base_path = Path(output_dir) / collection_name if output_dir else Path(collection_name)
    base_path.mkdir(parents=True, exist_ok=True)

    print(f"\nDownloading {collection_name} items to {base_path}")
    print("-" * 70)

    # flatten the nested lists in items
    items = []
    for item_list in items_dict.values():
        items.extend(item_list)

    for idx, item in enumerate(items, 1):
        item_date = item.datetime.strftime("%Y-%m-%d") if item.datetime else "unknown"
        item_path = base_path / item.id
        item_path.mkdir(parents=True, exist_ok=True)

        print(f"[{idx}/{len(items)}] {item.id} ({item_date})")

        try:
            # Determine collection ID from item or collection name
            collection_id = (
                item.collection_id if hasattr(item, "collection_id") else None
            )
            if not collection_id and "s30" in collection_name:
                collection_id = "hls2-s30"
            elif not collection_id and "l30" in collection_name:
                collection_id = "hls2-l30"

            # Get available assets
            available_bands = list(item.assets.keys())

            # Filter to requested bands if specified
            if band_names:
                bands_to_download = band_names[collection_id]
            else:
                bands_to_download = available_bands

            # Check if any requested bands are not available
            unavailable_bands = [band for band in bands_to_download if band not in available_bands]
            if unavailable_bands:
                print(f"      ⊘ Requested bands not available: {unavailable_bands}. Available: {available_bands}")

            if not bands_to_download:
                print(f"      ⊘ No requested bands found. Available: {available_bands}")
                stats["skipped"] += 1
                continue

            # Download each band
            bands_downloaded = []
            for requested_band in bands_to_download:

                asset = item.assets[requested_band]
                url = asset.href

                # Sign the URL with Planetary Computer credentials
                # signed_url = planetary_computer.sign_url(url)

                filename = item_path / f"{requested_band}.tif"

                if filename.exists():
                    print(f"      ✓ {requested_band} (exists)")
                    stats["files_downloaded"] += 1
                    bands_downloaded.append(requested_band)
                else:
                    try:
                        response = requests.get(url, timeout=30)
                        response.raise_for_status()
                        with open(filename, "wb") as f:
                            f.write(response.content)
                        print(
                            f"      ✓ {requested_band} ({response.headers.get('content-length', '?')} bytes)"
                        )
                        stats["files_downloaded"] += 1
                        bands_downloaded.append(requested_band)
                    except Exception as e:
                        print(f"      ✗ {requested_band} failed: {e}")
                        stats["failed"] += 1

            if bands_downloaded:
                stats["downloaded"] += 1

        except Exception as e:
            print(f"      ✗ Item failed: {e}")
            stats["failed"] += 1

    print("-" * 70)
    print(
        f"Download complete: {stats['downloaded']} items, {stats['files_downloaded']} files"
    )

    return stats
