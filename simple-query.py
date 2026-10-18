from astropy.io import fits
import requests
from io import BytesIO
import warnings
from astropy.io.fits.verify import VerifyWarning
import time

#warnings.simplefilter("ignore", category=VerifyWarning)

#fits_url = "https://www.canfar.net/storage/vault/file/AstroDataCitationDOI/CISTI.CANFAR/19.0074/data/CygX_N_NH3_11_all_rebase_multi.fits"
#fits_url = "https://www.canfar.net/storage/vault/file/AstroDataCitationDOI/CISTI.CANFAR/19.0074/data/CygX_N_NH3_11_all_rebase_multi_mom0_QA.fits"
#fits_url= "https://www.canfar.net/storage/vault/file/AstroDataCitationDOI/CISTI.CANFAR/19.0074/data/W3_west_NH3_11_all_rebase_multi.fits"
fits_url="https://dataverse.harvard.edu/api/access/datafile/:persistentId?persistentId=doi:10.7910/DVN/PXDKBC/QULHRK"
#fits_url="https://dataverse.harvard.edu/api/access/datafile/11587873"
start_t = time.perf_counter()


# Request only first ~6 MB of file
headers = {"Range": "bytes=0-6291456"}
response = requests.get(fits_url, headers=headers, stream=True, timeout=120)
response.raise_for_status()

partial_data = BytesIO(response.content)

with fits.open(partial_data, ignore_missing_end=True) as hdul:
    print("Successfully read FITS header (without full download).")
    print("\nHeader of first HDU:\n")
    print(repr(hdul[0].header))
    #print(repr(hdul[0].data))


end_t = time.perf_counter()

print(f"\nTime taken to download and read header: {end_t - start_t:.2f} seconds")

