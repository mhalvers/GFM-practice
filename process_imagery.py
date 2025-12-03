from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import xarray as xr
import rioxarray

from config import DATA_DIR


def process_image(image_path: Path) -> xr.DataArray:
    """Process a single image directory containing multiple bands.
    Applies masking and scaling to each band based on attributes
    in the TIFF files, then combines them into a single DataArray.

        Args:
           image_path (Path): Path to the image directory.
        Returns:
            xr.DataArray: Processed data array with bands as a dimension.
    """

    bands = list(image_path.glob("B*tif"))
    print(f"Processing {image_path.stem} with {len(bands)} bands")

    da_list = []
    for band in bands:
        _da = rioxarray.open_rasterio(band, mask_and_scale=True).squeeze()
        _da = _da.clip(min=0, max=1)
        _da = _da.expand_dims(band=[band.stem])
        da_list.append(_da)

    da = xr.concat(da_list, dim="band")
    da = da.sortby("band")
    da.name = image_path.stem

    return da


def process_and_save_image(image: Path) -> tuple[Path, bool]:
    """Process a single image and save the result.
    
    Args:
        image: Path to the image directory
        
    Returns:
        Tuple of (output_file_path, success)
    """
    try:
        da = process_image(image)
        file_name = image.parent / f"{image.name}_processed.tif"
        da.rio.to_raster(file_name)
        print(f"Wrote processed bands to {file_name}")
        return file_name, True
    except Exception as e:
        print(f"Error processing {image}: {e}")
        return image, False


def main():
    """Main processing function.
    Processes all images in the DATA_DIR in parallel and writes processed bands to disk.
    Processed images are saved as multi-band TIFFs in the parent directory.

    Args:
        None
    Returns:
        None
    """

    images = list(DATA_DIR.glob("HLS*"))
    
    if not images:
        print(f"No images found in {DATA_DIR}")
        return
    
    print(f"Processing {len(images)} images in parallel...")
    
    # Use ProcessPoolExecutor for parallel processing
    # max_workers defaults to min(32, os.cpu_count() + 4)
    with ProcessPoolExecutor() as executor:
        # Submit all tasks
        futures = {executor.submit(process_and_save_image, img): img for img in images}
        
        # Process results as they complete
        completed = 0
        for future in as_completed(futures):
            try:
                output_path, success = future.result()
                if success:
                    completed += 1
            except Exception as e:
                img = futures[future]
                print(f"Error processing {img}: {e}")
    
    print(f"Processing complete. Successfully processed {completed}/{len(images)} images.")


if __name__ == "__main__":
    main()
