from astroquery.alma import Alma
from astroquery.simbad import Simbad
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

# -----------------------------
# User input
# -----------------------------
ra = float(input("Enter RA (deg): "))
dec = float(input("Enter Dec (deg): "))
radius = input("Search radius (arcsec, default=10): ").strip()
radius = float(radius) if radius else 10.0

coord = SkyCoord(ra=ra*u.deg, dec=dec*u.deg, frame="icrs")

print("\nQuerying archive...\n")

# -----------------------------
# Query observations
# -----------------------------
#obs = Alma.query_region(coord, radius=radius*u.arcsec)
obs = Simbad.query_region(coord, radius=radius*u.arcsec)

if len(obs) == 0:
    print("No data found.")
    sys.exit(0)

print(f"Found {len(obs)} observations")

# -----------------------------
# Pick ONE observation (fast!)
# -----------------------------
obs_id = obs["obs_id"][0]
print(f"\nUsing obs_id: {obs_id}")

# -----------------------------
# Retrieve ONLY science images
# -----------------------------
download_dir = "alma_fast"
os.makedirs(download_dir, exist_ok=True)

print("\nRetrieving science FITS images...\n")

files = Simbad.retrieve_data(
    obs_id,
    product_type="image",
    fitsonly=True,
    download_dir=download_dir
)

if not files:
    print("No FITS images retrieved.")
    sys.exit(0)

# -----------------------------
# Display first valid image
# -----------------------------
for f in files:
    if not f.lower().endswith(".fits"):
        continue

    with fits.open(f) as hdul:
        for hdu in hdul:
            if hdu.data is None:
                continue

            data = hdu.data
            while data.ndim > 2:
                data = data[0]

            if not np.isfinite(data).any():
                continue

            vmin, vmax = np.nanpercentile(data, [5, 99])

            plt.figure(figsize=(6, 6))
            plt.imshow(data, origin="lower", cmap="inferno",
                       vmin=vmin, vmax=vmax)
            plt.colorbar(label="Intensity")
            plt.title(os.path.basename(f))
            plt.tight_layout()
            plt.show()

            sys.exit(0)

print("No plottable image found.")

