# %%
from huggingface_hub import snapshot_download

repo_name = "ibm-nasa-geospatial/Prithvi-EO-1.0-100M-multi-temporal-crop-classification"
local_download_path = "./prithvi_local_repo"

# %%
print(f"Starting download of {repo_name} to {local_download_path}...")

# Execute the download
# This function handles authentication, progress bars, and file integrity checks
snapshot_download(
    repo_id=repo_name, 
    local_dir=local_download_path,
    # This is an optional argument to only download the essential model files 
    # and ignore heavy or unnecessary files, if needed.
    # ignore_patterns=["*.safetensors", "*.msgpack"] 
)

print("Download complete. Files are available at:", local_download_path)