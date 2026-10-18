import numpy as np
import warnings
import requests
from io import BytesIO
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.visualization import simple_norm
from astropy.io.fits.verify import VerifyWarning
from astropy.visualization.wcsaxes import WCSAxes
from astropy.coordinates import Angle
from astroquery.simbad import Simbad
import sys

warnings.simplefilter("ignore", category=VerifyWarning)

Fits_Files = "https://dataverse.harvard.edu/api/search"
HEADER_BYTES = 6 * 1024 * 1024  # header-only read

# =========================
# FAST HEADER READ
# =========================
def read_header_fast(url):
    headers = {"Range": f"bytes=0-{HEADER_BYTES}"}
    r = requests.get(url, headers=headers, timeout=60)
    r.raise_for_status()
    with fits.open(BytesIO(r.content), ignore_missing_end=True) as hdul:
        return hdul[0].header

# =========================
# TARGET INPUT
# =========================
def resolve_target():
    print("\nHow would you like to specify the target?")
    print("1) Object name")
    print("2) Sky coordinates (RA, Dec in degrees)")
    choice = input("Enter 1 or 2: ").strip()

    try:
        if choice == "1":
            object = input("Enter object name: ").strip()
            result = Simbad.query_object(object)
            ra = result["ra"][0]
            dec = result["dec"][0]
            coord = SkyCoord(ra*u.deg, dec*u.deg)
            print(f"Resolved coordinates: RA={ra:.6f}, Dec={dec:.6f}")
        elif choice == "2":
            ra = float(input("Enter RA (deg): "))
            dec = float(input("Enter Dec (deg): "))
            #ra = input("Enter RA: ")
            #dec = input("Enter Dec: ")
            coord = SkyCoord(ra*u.deg, dec*u.deg)
            #coord = SkyCoord(ra, dec, unit=(u.hourangle, u.deg))
        else:
            raise ValueError
        return coord

    except Exception:
        exit()

# =========================
# FIND MATCHING FILE
# =========================
def find_matching_keystone(coord, urls):
    print("\nSearching FITS files...")
    for url in urls:
        try:
            hdr = read_header_fast(url)
            w = WCS(hdr, naxis=2)
            x, y = w.world_to_pixel(coord)
            if 0 <= x < hdr["NAXIS1"] and 0 <= y < hdr["NAXIS2"]:
                return url, hdr, (int(x), int(y))
        except Exception:
            continue
    return None, None, None

# =========================
# AXIS REPORT (HEADER ONLY)
# =========================
def report_axes(hdr):
    print("\nFILE AXIS SUMMARY (header-only)")
    naxis = hdr.get("NAXIS", 0)
    print(f"NAXIS = {naxis}")
    print("Spatial axes: RA, Dec")

    if naxis <= 2:
        print("2D image — no planes")
        return

    for i in range(3, naxis + 1):
        print(
            f"Axis {i}: "
            f"{hdr.get(f'CTYPE{i}', 'UNKNOWN')} "
            f"({hdr.get(f'NAXIS{i}', '?')} elements)"
        )

# =========================
# LOAD MOM0 QA IMAGE
# =========================
def load_mom0_qa(url):
    mom0_url = url.replace(".fits", "_mom0_QA.fits")
    #print(f"\nLoading QA image:\n{mom0_url}")
    r = requests.get(mom0_url, timeout=120)
    r.raise_for_status()
    with fits.open(BytesIO(r.content)) as hdul:
        return hdul[0].data, hdul[0].header

# =========================
# SAFE CUTOUT
# =========================
def safe_cutout(image, x, y, r):
    ny, nx = image.shape
    x0 = max(0, x - r)
    x1 = min(nx, x + r)
    y0 = max(0, y - r)
    y1 = min(ny, y + r)
    region = image[y0:y1, x0:x1]
    return None if region.size == 0 else region

# =========================
# Radial profile function
# =========================
def radial_profile(image, center):
    y, x = np.indices(image.shape)
    r = np.sqrt((x-center[0])**2 + (y-center[1])**2).astype(int)
    tbin = np.bincount(r.ravel(), image.ravel())
    nr = np.bincount(r.ravel())
    return tbin / np.maximum(nr, 1)

# =========================
# Mean and Median over Region
# =========================
def mean_median_region(image, mask=None, label="Pixel Value", unit="", load_cube_func=None, plane_index=None):

    import numpy as np
    import matplotlib.pyplot as plt

    # --- Optional cube loading ---
    if load_cube_func is not None and plane_index is not None:
        print(f"Loading full cube for plane {plane_index}...")
        cube_data = load_cube_func()
        image = cube_data[plane_index]

    # --- Select region ---
    if mask is not None:
        region_pixels = image[mask]
    else:
        region_pixels = image.ravel()

    # --- Clean data ---
    region_pixels = region_pixels[np.isfinite(region_pixels)]

    if region_pixels.size == 0:
        print("No valid pixels in region.")
        return np.nan, np.nan

    # --- Statistics ---
    mean_val = np.mean(region_pixels)
    median_val = np.median(region_pixels)

    # --- Plot histogram ---
    plt.figure(figsize=(8, 5))

    plt.hist(region_pixels, bins=50, color='skyblue', edgecolor='black')

    plt.axvline(mean_val, color='red', linestyle='--',
                label=f"Mean = {mean_val:.2f} {unit}")

    plt.axvline(median_val, color='green', linestyle='--',
                label=f"Median = {median_val:.2f} {unit}")

    plt.xlabel(f"{label} {unit}")
    plt.ylabel("Number of Pixels")

    plt.title(f"{label} Distribution in Selected Region")

    plt.legend()
    plt.grid(alpha=0.3)

    plt.show()

    return mean_val, median_val


# =========================
# COMPUTE MOMENT 1 (VELOCITY)
# =========================
def compute_moment1(url):
    import numpy as np
    import requests
    from astropy.io import fits
    from io import BytesIO

    # --- Download FITS safely ---
    r = requests.get(url, stream=True, timeout=120)
    r.raise_for_status()

    data = BytesIO()
    for chunk in r.iter_content(chunk_size=1024 * 1024):
        if chunk:
            data.write(chunk)
    data.seek(0)

    # --- Load cube ---
    with fits.open(data, ignore_missing_end=True) as hdul:
        cube = hdul[0].data.astype(float)
        hdr = hdul[0].header

    # --- Build frequency axis (Hz) ---
    nchan = cube.shape[0]
    crval = hdr["CRVAL3"]
    cdelt = hdr["CDELT3"]
    crpix = hdr["CRPIX3"]
    rest_freq = hdr["RESTFRQ"]

    freqs = (np.arange(nchan) - (crpix - 1)) * cdelt + crval

    # --- Convert to velocity (km/s) ---
    c = 299792.458  # km/s
    velocities = -c * (freqs - rest_freq) / rest_freq
    velocities = velocities[:, np.newaxis, np.newaxis]

    # --- Estimate noise ---
    rms = np.nanstd(cube)

    # --- Estimate noise ---
    rms = np.nanstd(cube)

    # --- Integrated signal mask (FIX) ---
    moment0_temp = np.nansum(cube, axis=0)
    mask2d = moment0_temp > (0.07 * rms * np.sqrt(nchan))

    cube_masked = np.where(mask2d[np.newaxis, :, :], cube, 0)

    # --- Moment 0 ---
    moment0 = np.sum(cube_masked, axis=0)

    # --- Moment 1 ---
    moment1 = np.sum(cube_masked * velocities, axis=0) / np.where(moment0 > 0, moment0, np.nan)

    # --- Clean output ---
    moment1 = np.where(np.isfinite(moment1), moment1, np.nan)

    return moment1, hdr
# =========================
# COMPUTE MOMENT 2 (DISPERSION)
# =========================
def compute_moment2(url):
    import numpy as np
    import requests
    from astropy.io import fits
    from io import BytesIO

    # --- Download FITS safely ---
    r = requests.get(url, stream=True, timeout=120)
    r.raise_for_status()

    data = BytesIO()
    for chunk in r.iter_content(chunk_size=1024 * 1024):
        if chunk:
            data.write(chunk)
    data.seek(0)

    # --- Load cube ---
    with fits.open(data, ignore_missing_end=True) as hdul:
        cube = hdul[0].data.astype(float)
        hdr = hdul[0].header

    # --- Spectral axis ---
    nchan = cube.shape[0]
    crval = hdr["CRVAL3"]
    cdelt = hdr["CDELT3"]
    crpix = hdr["CRPIX3"]
    rest_freq = hdr["RESTFRQ"]

    freqs = (np.arange(nchan) - (crpix - 1)) * cdelt + crval

    # --- Velocity axis (km/s) ---
    c = 299792.458
    velocities = -c * (freqs - rest_freq) / rest_freq
    velocities = velocities[:, np.newaxis, np.newaxis]

    # --- Noise estimate ---
    rms = np.nanstd(cube)

    # --- Mask (same as moment 1!) ---
    mask = cube > (0.095 * rms)
    cube_masked = np.where(mask, cube, 0)

    # --- Moment 0 ---
    moment0 = np.sum(cube_masked, axis=0)

    # --- Moment 1 (needed for moment 2) ---
    moment1 = np.sum(cube_masked * velocities, axis=0) / np.where(moment0 > 0, moment0, np.nan)

    # reshape moment1 for broadcasting
    moment1_3d = moment1[np.newaxis, :, :]

    # --- Moment 2 (velocity dispersion) ---
    variance = np.sum(cube_masked * (velocities - moment1_3d)**2, axis=0) / np.where(moment0 > 0, moment0, np.nan)

    moment2 = np.sqrt(variance)

    # --- Clean output ---
    moment2 = np.where(np.isfinite(moment2), moment2, np.nan)

    return moment2, hdr
# =========================
# Brightness Temperature Maps
# =========================
def compute_peak_map(url):
    print("Computing peak NH3 (1,1) brightness map...")

    r = requests.get(url, timeout=120)
    r.raise_for_status()

    with fits.open(BytesIO(r.content)) as hdul:
        cube = hdul[0].data.astype(float)
        hdr = hdul[0].header

    if cube.ndim == 4:
        cube = cube[0]

    cube[cube <= 0] = np.nan

    T_peak = np.nanmax(cube, axis=0)

    return T_peak, hdr

def compute_average_spectrum(url, threshold_sigma=3):
    import numpy as np
    import requests
    from astropy.io import fits
    from io import BytesIO

    # Download cube
    r = requests.get(url, timeout=120)
    r.raise_for_status()

    with fits.open(BytesIO(r.content)) as hdul:
        cube = hdul[0].data.astype(float)
        hdr = hdul[0].header

    # Handle 4D cubes (e.g., polarization)
    if cube.ndim == 4:
        cube = cube[0]

    # Compute peak emission per pixel
    peak_map = np.nanmax(cube, axis=0)

    # Threshold based on noise
    noise = np.nanstd(peak_map)
    threshold = threshold_sigma * noise

    # Create mask of "emission pixels"
    mask = peak_map > threshold

    # Avoid empty mask
    if not np.any(mask):
        raise ValueError("No emission pixels found above threshold.")

    # Compute average spectrum over selected pixels
    masked_cube = cube[:, mask]
    spectrum = np.nanmean(masked_cube, axis=1)

    return spectrum, hdr


import matplotlib.patches as patches

def draw_region(ax, px, py, half_size, region_choice):
    """
    Draws the selected region on a WCS plot.
    """
    if region_choice == "1":  # Circle
        circle = patches.Circle(
            (px, py),
            half_size,
            edgecolor='cyan',
            facecolor='none',
            linewidth=2
        )
        ax.add_patch(circle)

    elif region_choice == "2":  # Rectangle
        rect = patches.Rectangle(
            (px - half_size, py - half_size),
            2 * half_size,
            2 * half_size,
            edgecolor='cyan',
            facecolor='none',
            linewidth=2
        )
        ax.add_patch(rect)

def get_region_cutout(image, header, px, py, half_size, region_choice):
    from astropy.nddata import Cutout2D
    from astropy.wcs import WCS

    # Build WCS from the FITS header (this is critical)
    wcs = WCS(header).celestial

    if region_choice == "1":  # Circle → approximate as square
        size = 2 * half_size

    elif region_choice == "2":
        size = 2 * half_size

    else:
        return image, None

    cutout = Cutout2D(
        image,
        position=(px, py),
        size=(size, size),
        wcs=wcs,              
        mode="partial",
        fill_value=np.nan
    )

    return cutout.data, cutout
# =========================
# MAIN
# =========================


print("\nQuerying Dataverse...\n")

coord = resolve_target()

params = {
    "q": "astronomy",  # <-- change this
    "type": "dataset"
}

r = requests.get(Fits_Files, params=params)
r.raise_for_status()

data = r.json()
items = data.get("data", {}).get("items", [])

if len(items) == 0:
    print("No datasets found.")
    sys.exit(0)

print(f"Found {len(items)} datasets")

# --- collect FITS URLs (this replaces your "files") ---
urls = []

for item in items:
    try:
        persistent_id = item["global_id"]

        dataset_url = f"https://dataverse.harvard.edu/api/datasets/:persistentId/?persistentId={persistent_id}"
        r2 = requests.get(dataset_url)
        r2.raise_for_status()

        dataset_info = r2.json()

        files = dataset_info["data"]["latestVersion"]["files"]
        fits_files = [
            f["url"] for f in files
            if f.get("url", "").lower().endswith(".fits")
        ]
        urls = fits_files
        for f in files:
            filename = f["dataFile"]["filename"]
            file_id = f["dataFile"]["id"]

            if filename.endswith(".fits"):
                urls.append(
                    f"https://dataverse.harvard.edu/api/access/datafile/{file_id}"
                )

    except Exception:
        continue

url, hdr, pixel = find_matching_keystone(coord, urls)

if url is None:
    print("No matching file found for this coordinate.")
    sys.exit(0)

px, py = pixel

if len(urls) == 0:
    print("No FITS files found.")
    sys.exit(0)

print(f"Found {len(urls)} FITS files")

print("\nA matching file has been located!")
print("URL:", url)
print(f"Pixel location: ({px}, {py})")

report_axes(hdr)

# -------------------------
# OPTIONAL PLANE SELECTION
# -------------------------
use_plane = False
plane_index = None
cube_data = None
if hdr.get("NAXIS", 0) >= 3 and hdr.get("NAXIS3", 0) > 1:
    choice = input("\nWould you like to select a specific spectral plane? (y/n): ").strip().lower()
    if choice == "y":
        use_plane = True
        naxis3 = hdr["NAXIS3"]
        print(f"Available planes: 0 to {naxis3 - 1}")
        plane_index = int(input("Enter plane number: "))
        plane_index = max(0, min(plane_index, naxis3 - 1))
        cube_data, _ = load_mom0_qa(url)

# =========================
# LOAD MOM0 QA IMAGE
# =========================
mom0, mom0_hdr = load_mom0_qa(url)
pixscale = abs(mom0_hdr["CDELT2"]) * 3600  # arcsec/pixel
if use_plane and cube_data is not None:
    mom0 = cube_data[plane_index]
image_shape = mom0.shape

def circular_mask(image_shape, x0, y0, radius):
    """
    Returns a boolean mask of shape image_shape with True inside the circle.
    """
    y, x = np.indices(image_shape)
    return (x - x0)**2 + (y - y0)**2 <= radius**2
# -------------------------
# IF PLANE SELECTED, LOAD THAT PLANE
# -------------------------
if use_plane:
    print(f"\nLoading full cube and selecting plane {plane_index}...")
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    with fits.open(BytesIO(r.content)) as hdul:
        cube = hdul[0].data
    mom0 = cube[plane_index]

# =========================
# REGION SELECTION
# =========================
print("\nChoose region shape:")
print("1) Circle")
print("2) Rectangle")

region_choice = input("Enter 1 or 2: ").strip()


if region_choice == "1":
    radius_arcsec = float(input("Circle radius (arcsec): "))
    half_size = max(1, int(radius_arcsec / pixscale))
elif region_choice == "2":
    width_arcsec = float(input("Rectangle width (arcsec): "))
    height_arcsec = float(input("Rectangle height (arcsec): "))
    half_size = max(int(width_arcsec / pixscale), int(height_arcsec / pixscale)) // 2
else:
    print("Improper region selection.")
    exit()

# =========================
# INTERACTIVE SNAPSHOT MENU
# =========================
while True:
    print("\nChoose quick-look image:")
    print("1) Moment 0 Map (Integrated intensity)")
    print("2) Moment 1 Map (Velocity)")
    print("3) Moment 2 Map (Velocity Dispersion)")
    print("4) Mean over area")
    print("5) Median over area")
    print("6) Brightness Temperature Map")
    print("7) Average Spectrum Map")
    print("q) Quit")

    choice = input("Enter 1–7 or q: ").strip()

    if choice.lower() == "q":
        print("Exiting menu...")
        break

    plt.close('all')  # closes old plots so things don't stack

    # =========================
    # OPTIONS
    # =========================
    if choice == "1":
        print("Plotting Moment 0...")

        region_mom0, cutout = get_region_cutout(
            mom0,
            mom0_hdr,  
            px,
            py,
            half_size,
            region_choice
        )

        fig = plt.figure()
        ax = plt.subplot(projection=cutout.wcs)  # now valid

        im = ax.imshow(
            region_mom0,
            origin="lower",
            cmap="inferno",
            norm=simple_norm(region_mom0, stretch="sqrt", percent=99))

        ax.set_title("Moment 0 Map")
        ax.set_xlabel("RA (J2000)")
        ax.set_ylabel("Dec (J2000)")

        plt.colorbar(im, ax=ax, label="Integrated Intensity")

        plt.show()
   

    elif choice == "2":
        print("Computing moment 1...")

        m1, hdr = compute_moment1(url)

        region_m1, cutout = get_region_cutout(
            m1, hdr, px, py, half_size, region_choice
        )
        wcs = WCS(hdr).celestial
        fig = plt.figure()
        ax = plt.subplot(projection=cutout.wcs)

        im = ax.imshow(region_m1, origin="lower", cmap="plasma", vmin=-20, vmax=20)

        plt.colorbar(im, ax=ax, label="Velocity (km/s)")

        ax.set_title("Moment 1 Map")
        ax.set_xlabel("RA (J2000)")
        ax.set_ylabel("Dec (J2000)")
        plt.show()

    elif choice == "3":
        print("Computing moment 2...")

        m2, hdr = compute_moment2(url)

        region_m2, cutout = get_region_cutout(
            m2, hdr, px, py, half_size, region_choice
        )

        fig = plt.figure()
        ax = plt.subplot(projection=cutout.wcs)

        #im = ax.imshow(region_m2, origin="lower", cmap="viridis", vmin=0, vmax=10)
        im = ax.imshow(region_m2, origin="lower", cmap="viridis")

        plt.colorbar(im, ax=ax, label="Velocity Dispersion (km/s)")
        ax.set_xlabel("RA (J2000)")
        ax.set_ylabel("Dec (J2000)")
        ax.set_title("Moment 2 Map")

        plt.show()

    elif choice == "4":  # Mean
        if region_choice == "1":
            mask = circular_mask(mom0.shape, px, py, half_size)
            region_values = mom0[mask]
            display_region = np.where(mask, mom0, np.nan)
        else:
            display_region = safe_cutout(mom0, px, py, half_size)
            region_values = display_region.ravel()

        region_values = region_values[~np.isnan(region_values)]

        if region_values.size == 0:
            display_region = safe_cutout(mom0, px, py, 10)
            region_values = display_region.ravel()
            region_values = region_values[~np.isnan(region_values)]

        mean_val = np.nanmean(region_values)
        print(f"Mean intensity in selected region: {mean_val:.3f}")

        plt.figure()
        plt.imshow(display_region, origin="lower", cmap="inferno",
                   norm=simple_norm(display_region, stretch="sqrt", percent=99))
        plt.colorbar(label="Intensity")
        plt.title(f"Region Image (Mean = {mean_val:.3f})")
        plt.show()

        plt.figure()
        plt.hist(region_values, bins=30)
        plt.axvline(mean_val, linestyle="--", label=f"Mean = {mean_val:.3f}")
        plt.xlabel("Pixel Intensity")
        plt.ylabel("Number of Pixels")
        plt.title("Intensity Distribution in Region")
        plt.legend()
        plt.show()

    elif choice == "5":  # Median
        if region_choice == "1":
            mask = circular_mask(mom0.shape, px, py, half_size)
            region_values = mom0[mask]
            display_region = np.where(mask, mom0, np.nan)
        else:
            display_region = safe_cutout(mom0, px, py, half_size)
            region_values = display_region.ravel()

        region_values = region_values[~np.isnan(region_values)]

        if region_values.size == 0:
            display_region = safe_cutout(mom0, px, py, 10)
            region_values = display_region.ravel()
            region_values = region_values[~np.isnan(region_values)]

        median_val = np.nanmedian(region_values)
        print(f"Median intensity in selected region: {median_val:.3f}")

        plt.figure()
        plt.imshow(display_region, origin="lower", cmap="inferno",
                   norm=simple_norm(display_region, stretch="sqrt", percent=99))
        plt.colorbar(label="Intensity")
        plt.title(f"Region Image (Median = {median_val:.3f})")
        plt.show()

        plt.figure()
        plt.hist(region_values, bins=30)
        plt.axvline(median_val, linestyle="--", label=f"Median = {median_val:.3f}")
        plt.xlabel("Pixel Intensity")
        plt.ylabel("Number of Pixels")
        plt.title("Intensity Distribution in Region")
        plt.legend()
        plt.show()

    elif choice == "6":
        print("Generating map...")

        result, hdr = compute_peak_map(url)

        region_temp, cutout = get_region_cutout(
            result, hdr, px, py, half_size, region_choice
        )

        fig = plt.figure()
        ax = plt.subplot(projection=cutout.wcs)

        im = ax.imshow(region_temp, origin="lower", cmap="inferno", vmin=0, vmax=5)

        plt.colorbar(im, ax=ax, label="Temperature (K)")
        ax.set_xlabel("RA (J2000)")
        ax.set_ylabel("Dec (J2000)")
        ax.set_title("Intensity Temperature Map")

        plt.show()

    elif choice == "7":
        print("Generating average spectrum...")

        spectrum, hdr = compute_average_spectrum(url)

        # Build frequency axis
        crval3 = hdr['CRVAL3']
        cdelt3 = hdr['CDELT3']
        crpix3 = hdr['CRPIX3']

        nchan = len(spectrum)
        freq = crval3 + (np.arange(nchan) - (crpix3 - 1)) * cdelt3

        plt.figure()
        plt.plot(freq, spectrum)
        plt.title("Average NH3 (1,1) Spectrum")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Brightness Temperature (K)")
        plt.show()
