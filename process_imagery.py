from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd
import geopandas as gpd
import xarray as xr
import rioxarray
from rioxarray.merge import merge_datasets
from shapely import box
from tqdm import tqdm

from config import DATA_DIR, BBOX, HLS_CRS

xr.set_options(display_style="text")

N_JOBS = 6


def convert_latlon_bbox_to_hls_crs(bbox):
    geom = box(*bbox)
    gdf = gpd.GeoDataFrame({"geometry": [geom]}, crs="EPSG:4326")
    gdf = gdf.to_crs(HLS_CRS)

    return gdf.total_bounds.tolist()  # minx, miny, maxx, maxy


def bands_to_multiband_tif(image_path: Path, bbox: list[float] = None) -> xr.DataArray:
    """Process a single image directory containing multiple bands.
    Applies masking and scaling to each band based on attributes
    in the TIFF files, then combines them into a single DataArray.

        Args:
           image_path (Path): Path to the image directory.
        Returns:
            xr.DataArray: Processed data array with bands as a dimension.
    """

    bands = list(image_path.glob("B*tif"))

    ds_list = []
    for band in bands:
        _ds = rioxarray.open_rasterio(band, mask_and_scale=True, band_as_variable=True)
        _ds = _ds.rename({"band_1": band.stem})   # note variable name is lost when writing tif
        if bbox is not None:
            _ds = _ds.rio.clip_box(minx=bbox[0], miny=bbox[1], maxx=bbox[2], maxy=bbox[3])
        ds_list.append(_ds)

    ds = xr.merge(ds_list, compat="override")
    ds = ds.clip(min=0, max=1)
    ds.coords["name"] = image_path.stem

    return ds

def process_and_save_image(image: Path, bbox: list[float] = None) -> tuple[Path, bool]:
    """Process a single image and save the result.

    Args:
        image: Path to the image directory

    Returns:
        Tuple of (output_file_path, success)
    """
    
    try:
        ds = bands_to_multiband_tif(image, bbox=bbox)
        file_name = image.parent / f"{image.name}_processed.tif"
        ds.rio.to_raster(file_name)
        print(f"Wrote processed bands to {file_name}")
        return file_name, True
    except Exception as e:
        print(f"Error processing {image}: {e}")
        return image, False


def merge_adjacent_tiles(
    processed_tifs: list[Path], output_path: Path = DATA_DIR, bounds: list[float] = None
) -> Path:
    """Merge adjacent tiles taken on the same day into a single raster."""

    # first, find duplicate dates

    df = pd.DataFrame(
        index=[idir.stem for idir in processed_tifs], columns=["date", "path"]
    )
    for idir in processed_tifs:
        date_as_str = idir.stem.split(".")[3].split("T")[0]
        date = pd.to_datetime(date_as_str, format="%Y%j").date()
        df.loc[idir.stem, "date"] = date
        df.loc[idir.stem, "path"] = idir

    duplicates = df[df["date"].duplicated(keep=False)]
    date = duplicates.iloc[0]["date"]
    duplicates = duplicates["path"].tolist()

    print(f"Merging {[d.name for d in duplicates]} taken on {date}")

    # load each in with rasterio and merge
    ds = merge_datasets(
        [rioxarray.open_rasterio(f, band_as_variable=True) for f in duplicates],
        bounds=bounds,
    )

    # ds = ds.rename({band:ds[band].attrs["long_name"] for band in ds})

    # write date as string in YYYYDDD format
    filename = (
        duplicates[0].name.rsplit(".", 4)[0]
        + "."
        + date.strftime("%Y%jT000000")
        + ".v2.0_merged_processed.tif"
    )
    ds.rio.to_raster(output_path / filename)
    print(f"Wrote merged tile to {output_path / filename}")


if __name__ == "__main__":

    bbox = convert_latlon_bbox_to_hls_crs(BBOX)

    images = sorted(list(DATA_DIR.glob("HLS*")))
    if not images:
        raise ValueError(f"No image folders found in {DATA_DIR}")


    if N_JOBS==1:
        print(f"Processing {len(images)} images sequentially...")
        # Process sequentially (easier for debugging)
        completed = 0
        for img in tqdm(images, desc="Processing images"):
            _, success = process_and_save_image(img)
            if success:
                completed += 1

    else:
        print(f"Processing {len(images)} images in parallel...")
        with ProcessPoolExecutor(max_workers=N_JOBS) as executor:
            # Submit all tasks
            futures = {executor.submit(process_and_save_image, img): img for img in images}

            # Process results as they complete
            completed = 0
            for future in tqdm(as_completed(futures), total=len(images), desc="Processing images"):
                try:
                    output_path, success = future.result()
                    if success:
                        completed += 1
                except Exception as e:
                    img = futures[future]
                    print(f"Error processing {img}: {e}")

    print(
        f"Conversion to multiband TIFFs complete: {completed} of {len(images)} images processed successfully."
    )

    # now merge adjacent tiles taken on same day
    processed_images = sorted(list(DATA_DIR.glob("HLS*_processed.tif")))
    # merge_adjacent_tiles(processed_images, DATA_DIR, bounds=bbox)

