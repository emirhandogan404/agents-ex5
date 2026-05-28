import requests
import rootpath
rootpath.append(pattern="pyproject.toml")
import project_paths as pp

ds_name =pp.CONFIG["dataset"]["name"]
out_path = pp.DATA_DIR / ds_name

print(f"Downloading {ds_name}...")
response = requests.get(pp.CONFIG["dataset"]["url"])
response.raise_for_status()

with open(out_path, "wb") as f:
    f.write(response.content)

print(f"Downloaded {ds_name} to:", out_path)