# %%
import numpy as np
import xarray as xr
import rioxarray as rxr
import matplotlib.pyplot as plt
from PIL import Image

from config import DATA_DIR

xr.set_options(display_style="text", display_expand_data=False)

# %% view the app.py example chips
# FILES = sorted(list(Path(".").rglob("chip*.tif")))
# rxr.open_rasterio(FILES[0])

# %%thumbnails of downloaded HLS imagery
THUMBNAILS = sorted(DATA_DIR.rglob("thumbnail.tif"))
RED = sorted(DATA_DIR.rglob("B04.tif"))


# %% find the bbox that surrounds all images
xmin, xmax, ymin, ymax = [], [], [], []

for f in RED:
    da = rxr.open_rasterio(f).squeeze()
    xmin.append(da.x.min().item())
    xmax.append(da.x.max().item())
    ymin.append(da.y.min().item())
    ymax.append(da.y.max().item())

xmin = np.min(xmin)
xmax = np.max(xmax)
ymin = np.min(ymin)
ymax = np.max(ymax)

# %% display red band
_, ax = plt.subplots(3, 2, figsize=(10, 10))
ax = ax.ravel()
for i, f in enumerate(RED):
    da = rxr.open_rasterio(f, mask_and_scale=True).squeeze()
    da.plot.imshow(ax=ax[i], vmin=0, vmax=1, add_colorbar=False)
    ax[i].set_title(f.parent.name)
    ax[i].set_xlabel(None)
    ax[i].set_ylabel(None)
    ax[i].set_xticklabels([])
    ax[i].set_yticklabels([])
    ax[i].set_xlim(xmin, xmax)
    ax[i].set_ylim(ymin, ymax)

# %% display thumbnails
_, ax = plt.subplots(3, 2, figsize=(10, 10))    
ax = ax.ravel()
for i, f in enumerate(THUMBNAILS):
    img = Image.open(f)
    ax[i].imshow(img)
    ax[i].set_title(f.parent.name)
    ax[i].axis("off")
plt.tight_layout()








# %%
da_list = []
for f in FILES[0:3]:
    _da = rxr.open_rasterio(f).squeeze()
    _da.name = f.parent.name
    da_list.append(_da)

# %%
da = xr.concat(da_list, dim="band")
da.name = "HLS.L30.T15TVL.2021249T165852.v2.0"
da

# %% display true color composite
da.plot.imshow(vmin=0, vmax=3000)

# %%
band_max = da.quantile(0.98)
band_min = da.quantile(0.02)

da_stretched = (da - band_min) / (band_max - band_min)
da_stretched = da_stretched.clip(0, 1)

# display stretched true color composite
da_stretched.plot.imshow()
plt.title(da.attrs["SENSING_TIME"])
