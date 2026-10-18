import requests
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u
import io

# target coordinate (example: Orion)
target = SkyCoord(83.63*u.deg, -5.39*u.deg)

base_url = "https://dataverse.harvard.edu/api/search"

page = 0
per_page = 25

while True:

    params = {
        "q": "fits",
        "type": "file",
        "subtree": "complete",
        "per_page": per_page,
        "start": page * per_page
    }

    r = requests.get(base_url, params=params)
    data = r.json()["data"]

    items = data["items"]

    if not items:
        break

    for item in items:

        file_id = item["file_id"]
        name = item["name"]

        download_url = f"https://dataverse.harvard.edu/api/access/datafile/{file_id}"

        print("Checking:", name)

        try:

            # download only first FITS block (header)
            header_bytes = requests.get(download_url, headers={"Range": "bytes=0-2880"}).content

            hdul = fits.open(io.BytesIO(header_bytes), ignore_missing_end=True)

            wcs = WCS(hdul[0].header)

            x, y = wcs.world_to_pixel(target)

            nx = hdul[0].header.get("NAXIS1")
            ny = hdul[0].header.get("NAXIS2")

            if nx and ny and 0 < x < nx and 0 < y < ny:
                print("MATCH FOUND")
                print(name)
                print(download_url)
                print()

        except Exception as e:
            print("Skipped:", name)

    page += 1