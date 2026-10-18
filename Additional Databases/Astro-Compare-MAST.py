from astroquery.mast import Observations
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

print("\nQuerying MAST archive...\n")

# -----------------------------
# Query observations
# -----------------------------
obs = Observations.query_region(coord, radius=radius*u.arcsec)

if obs is None or len(obs) == 0:
    print("No data found.")
    sys.exit(0)

print(f"Found {len(obs)} observations")

# -----------------------------
# Retrieve ONLY science images
# -----------------------------
download_dir = "mast_data"
os.makedirs(download_dir, exist_ok=True)

products = Observations.get_product_list(obs)

# Keep only science FITS images
products = Observations.filter_products(
    products,
    productType="SCIENCE",
    extension="fits"
)

if len(products) == 0:
    print("No science FITS products found.")
    sys.exit(0)

# Take only the first product to avoid downloading everything
first_product = products[0:1]

files = Observations.download_products(
    first_product,
    download_dir=download_dir
)

if not files:
    print("No FITS images retrieved.")
    sys.exit(0)

# -----------------------------
# Display first valid image
# -----------------------------
f = list(files.values())[0][0]  # get path of the first downloaded FITS file

with fits.open(f) as hdul:
    # loop through HDUs to find first with data
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
        break  # only plot the first valid HDU