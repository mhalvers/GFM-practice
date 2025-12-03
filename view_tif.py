# %%
import xarray as xr
import rioxarray as rxr
import matplotlib.pyplot as plt
from PIL import Image

from config import DATA_DIR

# thumbnails
FILES = sorted(DATA_DIR.rglob("thumbnail.tif"))

xr.set_options(display_style="text", display_expand_data=False)

# %%
[f.parent.name + " / " + f.name for f in FILES]

# %% display thumbnails
# open with matplotlib
_, ax = plt.subplots(2, 2, figsize=(10, 10))    
ax = ax.ravel()
for i, f in enumerate(FILES[0:4]):
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
