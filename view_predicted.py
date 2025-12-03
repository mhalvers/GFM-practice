# %%
from pathlib import Path

import xarray as xr
import rioxarray as rio
import matplotlib.pyplot as plt

PREDICTED_DIR = Path("output")

xr.set_options(display_style="text", display_expand_data=False)

# %%
files = sorted(list(PREDICTED_DIR.glob("*.tiff")))

# %%
da = rio.open_rasterio(files[0])
da.name = files[0].stem
da

# %%
_, ax = plt.subplots(2, 3, figsize=(10, 6))
ax = ax.flatten()
for i, band in enumerate(da.band.values):
    da.sel(band=band).plot(ax=ax[i], cmap="viridis", add_colorbar=False)
    ax[i].axis("off")
    ax[i].set_title(f"Band {band}")



# %% outputs
_, ax = plt.subplots(3, 1, figsize=(5, 15))
for i, file in enumerate(files[3:]):
    print(f"File: {file.name}")
    da = rio.open_rasterio(file)
    da.plot(ax=ax[i])


# %%
