from astroquery.cadc import Cadc
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.io import fits
from astropy.wcs import WCS
import numpy as np
import tempfile
import os

# ---------------------------------------
# 1. Ask for search mode
# ---------------------------------------
query_type = input("Search by (1) coordinates [RA,Dec in deg] or (2) pixel indices (local file)? Enter 1 or 2: ").strip()

if query_type == "1":
    ra = float(input("Enter Right Ascension (deg): "))
    dec = float(input("Enter Declination (deg): "))
    coord = SkyCoord(ra*u.deg, dec*u.deg, frame='icrs')
else:
    print("Pixel-based lookup requires a local FITS file — switching to coordinate mode for remote CANFAR search.")
    ra = float(input("Enter Right Ascension (deg): "))
    dec = float(input("Enter Declination (deg): "))
    coord = SkyCoord(ra*u.deg, dec*u.deg, frame='icrs')

# ---------------------------------------
# 2. Connect to CANFAR (CADC)
# ---------------------------------------
cadc = Cadc()
print("\n🔍 Searching the CANFAR archive for KEYSTONE data...")
results = cadc.query_region(coord, radius=0.5*u.deg, collection='KEYSTONE')

if len(results) == 0:
    print("❌ No KEYS­TONE data found near that position.")
    exit()

print(f"✅ Found {len(results)} candidate file(s).")
for i, row in enumerate(results):
    print(f"[{i}] {row['productID']} — {row['dataProductType']} — {row['instrument_name']}")

choice = int(input("\nSelect file index to download: "))
selected_row = results[choice]

# ---------------------------------------
# 3. Download the file from VOSpace
# ---------------------------------------
file_uri = selected_row['uri']
print(f"\n📥 Downloading {file_uri} ...")

local_path = tempfile.mktemp(suffix=".fits")
cadc.get_data(file_uri, destination=local_path)

print(f"Saved locally as: {local_path}")

# ---------------------------------------
# 4. Open and inspect FITS file
# ---------------------------------------
hdu = fits.open(local_path)[0]
data = np.squeeze(hdu.data)
wcs = WCS(hdu.header)

print(f"\nOpened {os.path.basename(local_path)}")
print(f"Data shape: {data.shape}")
print(f"Units: {hdu.header.get('BUNIT', 'unknown')}")
print(f"RA/DEC ref: {hdu.header.get('CRVAL1', '')}, {hdu.header.get('CRVAL2', '')}")

# ---------------------------------------
# 5. Ask what to extract
# ---------------------------------------
print("\nWhat do you want to extract?")
print("1. Intensity / Brightness")
print("2. Temperature (NH3 1,1 and 2,2 pair if available)")
option = input("Enter choice (1 or 2): ").strip()

# get pixel from coordinate
xpix, ypix = wcs.world_to_pixel(coord)
xpix, ypix = int(xpix), int(ypix)
value = data[ypix, xpix]
print(f"\nPixel value at ({xpix},{ypix}) = {value:.3f} {hdu.header.get('BUNIT','')}")

"""""
# ---------------------------------------
# 6. Optional: derive temperature
# ---------------------------------------
if option == "2":
    # Look for NH3 1,1 and 2,2 pair near same region
    print("\nSearching for matching NH3 (1,1) and (2,2) files for temperature computation...")
    f11, f22 = None, None
    for row in results:
        pid = row['productID']
        if "11" in pid and not f11:
            f11 = row
        elif "22" in pid and not f22:
            f22 = row
    if f11 and f22:
        f11_local = tempfile.mktemp(suffix=".fits")
        f22_local = tempfile.mktemp(suffix=".fits")
        cadc.get_data(f11['uri'], destination=f11_local)
        cadc.get_data(f22['uri'], destination=f22_local)
        d11 = np.squeeze(fits.getdata(f11_local))
        d22 = np.squeeze(fits.getdata(f22_local))
        wcs11 = WCS(fits.getheader(f11_local))
        wcs22 = WCS(fits.getheader(f22_local))
        x11, y11 = wcs11.world_to_pixel(coord)
        x22, y22 = wcs22.world_to_pixel(coord)
        T11 = d11[int(y11), int(x11)]
        T22 = d22[int(y22), int(x22)]
        if T11 > 0 and 0 < T22/T11 < 1:
            Trot = -41.5 / np.log(-0.282 / (T22/T11))
            print(f"Derived rotational temperature: {Trot:.2f} K")
        else:
            print("Could not compute temperature: invalid brightness ratio.")
    else:
        print("No NH3 (1,1)/(2,2) file pair found.")

"""""
