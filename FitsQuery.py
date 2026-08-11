import numpy as np
import warnings
import requests
import os
from io import BytesIO
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
from astropy.nddata import Cutout2D
from astropy.coordinates import SkyCoord
import astropy.units as u
import matplotlib.patches as patches
from astropy.visualization import simple_norm
from astropy.io.fits.verify import VerifyWarning
from astropy.visualization.wcsaxes import WCSAxes
from astropy.coordinates import Angle
from astroquery.simbad import Simbad

warnings.simplefilter("ignore", category=VerifyWarning)

Fits_Files = "keystone_url.txt"
HEADER_BYTES = 6 * 1024 * 1024  # header-only read

# HEADER-ONLY READING FUNCTION (FAST)
def read_header_fast(url):
    headers = {"Range": f"bytes=0-{HEADER_BYTES}"}
    r = requests.get(url, headers=headers, timeout=60)
    r.raise_for_status()
    with fits.open(BytesIO(r.content), ignore_missing_end=True) as hdul:
        return hdul[0].header

#First prompt-Enter TARGET Coordinates
def resolve_target():
    print("\nHow would you like to specify the target?")
    print("1) Object name")
    print("2) Sky coordinates (RA, Dec)")
    choice = input("Enter 1 or 2: ").strip()

    try:
        if choice == "1":
            object = input("Enter object name: ").strip()
            result = Simbad.query_object(object)
            ra = result["ra"][0]
            dec = result["dec"][0]
            coord = SkyCoord(ra*u.deg, dec*u.deg)
            #print(f"Resolved coordinates: RA={ra:.6f}, Dec={dec:.6f}")
        elif choice == "2":
            #ra = float(input("Enter RA (deg): "))
            #dec = float(input("Enter Dec (deg): "))
            ra = input("Enter RA: ")
            dec = input("Enter Dec: ")
            #coord = SkyCoord(ra*u.deg, dec*u.deg)
            coord = SkyCoord(ra, dec, unit=(u.hourangle, u.deg))
        else:
            raise ValueError
        return coord

    except Exception:
        exit()

#Searching Function to find the FITS file that contains the target coordinates
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

#FITS Header summary function
def report_axes(hdr):
    print("\nSummary of FITS header:")
    naxis = hdr.get("NAXIS", 0)
    print(f"NAXIS = {naxis}")
    print("Spatial axes: RA, Dec")

    if naxis <= 2:
        print("2D image")
        return

    for i in range(3, naxis + 1):
        print(
            f"Axis {i}: "
            f"{hdr.get(f'CTYPE{i}', 'UNKNOWN')} "
            f"({hdr.get(f'NAXIS{i}', '?')} elements)"
        )

#Load MOM0 QA image if available, otherwise compute from cube (with optional cube download)
def load_or_compute_mom0(url, compute_if_missing=True):

    mom0_url = url.replace(".fits", "_mom0_QA.fits")

    #First try dowloading existing mom0 QA image (fast if available, avoids cube download)
    try:

        r = requests.get(mom0_url, timeout=10)

        if r.status_code == 200:
            with fits.open(BytesIO(r.content)) as hdul:
                return hdul[0].data, hdul[0].header

        else:
            print("Moment 0 file not found online.")

    except Exception:
        print("Could not access Moment 0 file.")

    #Next option is to download cube and compute moment 0 if missing (slow drawback)
    if compute_if_missing:
        #print("Computing Moment 0 from cube")
        r = requests.get(url, timeout=120)
        r.raise_for_status()

        with fits.open(BytesIO(r.content)) as hdul:
            cube = hdul[0].data
            hdr = hdul[0].header
        mom0 = compute_moment0(cube, hdr)
        return mom0, hdr

    #raise FileNotFoundError("Moment 0 not available and compute_if_missing=False")


#Function that cuts out a defined region
def safe_cutout(image, x, y, r):
    ny, nx = image.shape
    x0 = max(0, x - r)
    x1 = min(nx, x + r)
    y0 = max(0, y - r)
    y1 = min(ny, y + r)
    region = image[y0:y1, x0:x1]
    return None if region.size == 0 else region

#Function to compute mean and median in a region, with optional cube loading for plane selection
def mean_median_region(image, mask=None, label="Pixel Value", unit="", load_cube_func=None, plane_index=None):

    # Optional cube loading if user selected a specific plane (only loads that plane to save time)
    if load_cube_func is not None and plane_index is not None:
        print(f"Loading full cube for plane {plane_index}...")
        cube_data = load_cube_func()
        image = cube_data[plane_index]

    #Select region pixels using mask if provided, otherwise use entire image
    if mask is not None:
        region_pixels = image[mask]
    else:
        region_pixels = image.ravel()

    # Remove NaNs and infs
    region_pixels = region_pixels[np.isfinite(region_pixels)]

    if region_pixels.size == 0:
        print("No valid pixels in region.")
        return np.nan, np.nan


    mean_val = np.mean(region_pixels)
    median_val = np.median(region_pixels)
    return mean_val, median_val

#Compute moment 0 (integrated intensity)
def compute_moment0(cube_data, header):
    wcs = WCS(header)

    # Identify spectral axis
    spec_axis = wcs.wcs.spec
    if spec_axis is None:#Can assume last axis is spectral if not explicitly defined
        spec_axis = -1

    pix_scale = wcs.wcs.cdelt[spec_axis]#Use WCS to get pixel scale in spectral axis (either frequency or velocity)

    # Convert to physical units (if possible) from WCS (in CUNIT)
    if "CUNIT{}".format(spec_axis + 1) in header:
        unit = u.Unit(header["CUNIT{}".format(spec_axis + 1)])
        delta_v = (pix_scale * unit).to(u.km/u.s)
    else:
        delta_v = pix_scale

    # Sum over the spectral axis
    moment0 = np.nansum(cube_data, axis=spec_axis) * delta_v

    return moment0

def get_moment0(moment0_path, cube_url, compute_if_missing=True):

    #If moment 0 file exists, load it (fast)
    if moment0_path is not None and os.path.exists(moment0_path):
        print("Using existing Moment 0 file (no cube download needed)...")

        with fits.open(moment0_path) as hdul:
            return hdul[0].data, hdul[0].header

    #Compute moment 0 from cube if no existing file and user allows computation (will require cube download, slow)
    if compute_if_missing:
        print("Moment 0 not found → downloading cube to compute...")

        r = requests.get(cube_url, timeout=120)
        r.raise_for_status()

        with fits.open(BytesIO(r.content)) as hdul:
            cube = hdul[0].data.astype(float)
            hdr = hdul[0].header

        if cube.ndim == 4:
            cube = cube[0]
        elif cube.ndim == 3:
            pass
        else:
            raise ValueError(f"Unexpected cube shape: {cube.shape}")

        mom0 = compute_moment0(cube, hdr)

        return mom0, hdr

    raise FileNotFoundError("No Moment 0 file and computation disabled.")

#Compute moment 1 (velocity)
def compute_moment1(url):

    r = requests.get(url, stream=True, timeout=120)
    r.raise_for_status()

    data = BytesIO()
    for chunk in r.iter_content(chunk_size=1024 * 1024):
        if chunk:
            data.write(chunk)
    data.seek(0)

    #Load whole cube
    with fits.open(data, ignore_missing_end=True) as hdul:
        cube = hdul[0].data.astype(float)
        hdr = hdul[0].header


    if cube.ndim == 4:
        cube = cube[0]
    if cube.ndim == 3:
        cube = cube  
    else:
        raise ValueError(f"Unexpected cube shape: {cube.shape}")

    #For the spectral axis fall back to header keywords (CRVAL3, CDELT3, CRPIX3)
    nchan = cube.shape[0]
    crval = hdr["CRVAL3"]
    cdelt = hdr["CDELT3"]
    crpix = hdr["CRPIX3"]
    rest_freq = hdr["RESTFRQ"]

    freqs = (np.arange(nchan) - (crpix - 1)) * cdelt + crval

    #Convert to velocity (km/s)
    c = 299792.458 #Value of speed of light in km/s
    velocities = -c * (freqs - rest_freq) / rest_freq
    velocities = velocities[:, np.newaxis, np.newaxis]

    #Noise estimate from cube
    rms = np.nanstd(cube)

    #Mask noise
    moment0_temp = np.nansum(cube, axis=0)
    #mask2d = moment0_temp > (0.07 * rms * np.sqrt(nchan))
    mask2d = moment0_temp > (0.0005 * rms * np.sqrt(nchan))

    cube_masked = np.where(mask2d[np.newaxis, :, :], cube, 0)

    #Compute moment 0
    moment0 = np.sum(cube_masked, axis=0)

    #Compute moment 1 (velocity) using masked cube and velocity axis
    moment1 = np.sum(cube_masked * velocities, axis=0) / np.where(moment0 > 0, moment0, np.nan)

    #Clean output by replacing infs with NaNs
    moment1 = np.where(np.isfinite(moment1), moment1, np.nan)

    return moment1, hdr

#Compute moment 2 (velocity dispersion)
def compute_moment2(url):

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

  
    c = 299792.458
    velocities = -c * (freqs - rest_freq) / rest_freq
    velocities = velocities[:, np.newaxis, np.newaxis]

    #Reduce noise
    rms = np.nanstd(cube)

    #Mask
    mask = cube > (0.095 * rms)
    cube_masked = np.where(mask, cube, 0)

    #Calculate moment 0 using masked cube
    moment0 = np.sum(cube_masked, axis=0)

    #Calculate moment 1 (velocity) using masked cube and velocity axis
    moment1 = np.sum(cube_masked * velocities, axis=0) / np.where(moment0 > 0, moment0, np.nan)

    #Reshape moment 1
    moment1_3d = moment1[np.newaxis, :, :]

    #Calculate Moment 2
    variance = np.sum(cube_masked * (velocities - moment1_3d)**2, axis=0) / np.where(moment0 > 0, moment0, np.nan)

    moment2 = np.sqrt(variance)

    moment2 = np.where(np.isfinite(moment2), moment2, np.nan)

    return moment2, hdr

#Function for computing peak temperature map (T_peak) from the cube
def compute_peak_map(url):
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

#Function to compute average spectrum over pixels with significant emission (for quick-look spectrum)
def compute_average_spectrum(url, threshold_sigma=3):
    # Download the whole cube
    r = requests.get(url, timeout=120)
    r.raise_for_status()

    with fits.open(BytesIO(r.content)) as hdul:
        cube = hdul[0].data.astype(float)
        hdr = hdul[0].header

    #Check if it is a 4D cubes (e.g., polarization)
    if cube.ndim == 4:
        cube = cube[0]
    cunit = hdr.get("BUNIT", "").strip()
    #Compute peak emission per pixel by collapsing the cube along the spectral axis
    peak_map = np.nanmax(cube, axis=0)

    # Estimate noise from the peak map (assuming most pixels are noise-dominated)
    noise = np.nanstd(peak_map)
    threshold = threshold_sigma * noise

    #Mask pixels that do not have significant emission (below threshold) to avoid skewing the average spectrum with noise
    mask = peak_map > threshold

    # Avoid empty mask which would cause issues in averaging
    if not np.any(mask):
        raise ValueError("No emission pixels found above threshold.")

    #Now average the spectrum only over the masked pixels with significant emission
    masked_cube = cube[:, mask]
    spectrum = np.nanmean(masked_cube, axis=1)

    return spectrum, hdr

#Function to draw the selected region (circle or rectangle) on the quick-look image
def draw_region(ax, px, py, half_size, region_choice):
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

#Function to safely cut out the selected region from the image, with fallback if cutout fails (e.g., due to WCS issues)
def get_region_cutout(image, header, px, py, half_size, region_choice):

    #Make sure it's a 2D image (cutout only works on 2D)
    if image.ndim == 4:
        image = image[0, 0]
    elif image.ndim == 3:
        image = image[0]

    #Celestial coordinates only for cutout (ignore spectral axes if present)
    wcs = WCS(header).celestial

    size = 2 * half_size
    
    try:
        cutout = Cutout2D(
            image,
            position=(px, py),
            size=(size, size),
            wcs=wcs,
            mode="partial",
            fill_value=np.nan
        )
        return cutout.data, cutout

    except Exception:
        ny, nx = image.shape
        x0 = int(max(0, px - half_size))
        x1 = int(min(nx, px + half_size))
        y0 = int(max(0, py - half_size))
        y1 = int(min(ny, py + half_size))

        return image[y0:y1, x0:x1], None
    

#MAIN CODE STARTS HERE

with open(Fits_Files) as f:
    urls = [l.strip() for l in f if l.strip()]

coord = resolve_target()
if coord is None:
    print("Exiting program.")
    exit()

result = find_matching_keystone(coord, urls)

if result is None:
    print("No file contains these coordinates.")
    exit()

url, hdr, pixel = result

if url is None:
    print("No file contains these coordinates.")
    exit()

px, py = pixel

print("\nA matching file has been located!")
print("URL:", url)
print(f"Pixel location: ({px}, {py})")

report_axes(hdr)

#Options for plane selection if 3D cube
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
        cube_data, _ = load_or_compute_mom0(url)


#Load moment 0 for quick-look and to get pixel scale
mom0, mom0_hdr = load_or_compute_mom0(url)
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
#Load the selected plane if user chose to select a specific plane (only loads that plane to save time, otherwise uses mom0 for quick-look)
if use_plane:
    print(f"\nLoading full cube and selecting plane {plane_index}...")
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    with fits.open(BytesIO(r.content)) as hdul:
        cube = hdul[0].data
    mom0 = cube[plane_index]

#Region Selection Prompt
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

# Main menu loop for quick-look images and statistics
while True:
    print("\nChoose quick-look image:")
    print("1) Moment 0 Map (Integrated intensity)")
    print("2) Moment 1 Map (Velocity)")
    print("3) Moment 2 Map (Velocity Dispersion)")
    print("4) Mean over area")
    print("5) Median over area")
    print("6) Temperature Map")
    print("7) Average Spectrum")
    print("q) Quit")

    choice = input("Enter 1–7 or q: ").strip()
    if choice in ["2", "3"]:  # Moment 1 and 2
        print("\n--- Velocity Selection ---")
        vmin = float(input("Enter lower velocity bounds: "))
        vmax = float(input("Enter upper velocity bounds: "))
    else:
        vmin = vmax = None
    
    if choice.lower() == "q":
            print("Exiting menu...")
            break

    plt.close('all')


    #Options:

    if choice == "1":
        print("Plotting Moment 0...")

        region_mom0, cutout = get_region_cutout(
            mom0, mom0_hdr, px, py, half_size, region_choice
        )

        #Cut out region and apply circular mask if selected
        if region_choice == "1":
            cx = region_mom0.shape[1] // 2
            cy = region_mom0.shape[0] // 2

            mask = circular_mask(region_mom0.shape, cx, cy, half_size)
            region_mom0 = np.where(mask, region_mom0, np.nan)

        fig = plt.figure()
        ax = plt.subplot(projection=cutout.wcs)

        im = ax.imshow(
            region_mom0,
            origin="lower",
            cmap="inferno",
            norm=simple_norm(region_mom0, stretch="sqrt", percent=99)
        )
        bunit = hdr.get("BUNIT", "").strip()
        if not bunit:
            bunit = "K km/s"
        ax.set_title("Moment 0 Map", fontsize=25)
        ax.set_xlabel("RA (J2000)", fontsize=20)
        ax.set_ylabel("Dec (J2000)", fontsize=20)
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label(f"Integrated Intensity ({bunit})", fontsize=20)  
        #plt.colorbar(im, ax=ax, label=f"Integrated Intensity ({bunit})", fontsize=16)
        plt.show()
   

    elif choice == "2":
        print("Computing moment 1...")

        m1, hdr = compute_moment1(url)

        region_m1, cutout = get_region_cutout(
            m1, hdr, px, py, half_size, region_choice
        )
        #Apply circular cutout mask if selected
        if region_choice == "1":
            cx = region_m1.shape[1] // 2
            cy = region_m1.shape[0] // 2

            mask = circular_mask(region_m1.shape, cx, cy, half_size)
            region_m1 = np.where(mask, region_m1, np.nan)
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
        #Apply circular cutout mask if selected
        if region_choice == "1":
            cx = region_m2.shape[1] // 2
            cy = region_m2.shape[0] // 2

            mask = circular_mask(region_m2.shape, cx, cy, half_size)
            region_m2 = np.where(mask, region_m2, np.nan)
        fig = plt.figure()
        ax = plt.subplot(projection=cutout.wcs)

        #im = ax.imshow(region_m2, origin="lower", cmap="viridis", vmin=0, vmax=10)
        im = ax.imshow(region_m2, origin="lower", cmap="viridis")

        #plt.colorbar(im, ax=ax, label="Velocity Dispersion (km/s)",fontsize=16)
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label(f"Velocity Dispersion (km/s)",fontsize=20)
        ax.set_xlabel("RA (J2000)",fontsize=20)
        ax.set_ylabel("Dec (J2000)", fontsize=20)
        ax.set_title("Moment 2 Map", fontsize=25)

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


    elif choice == "6":
        print("Generating temperature map...")

        temp_map, hdr = compute_peak_map(url)

        region_temp, cutout = get_region_cutout(
            temp_map, hdr, px, py, half_size, region_choice
        )

        #Cut out region and apply circular mask if selected
        if region_choice == "1":
            cx = region_temp.shape[1] // 2
            cy = region_temp.shape[0] // 2

            mask = circular_mask(region_temp.shape, cx, cy, half_size)
            region_temp = np.where(mask, region_temp, np.nan)

        fig = plt.figure()
        ax = plt.subplot(projection=cutout.wcs)

        vmin = np.nanpercentile(region_temp, 5)
        vmax = np.nanpercentile(region_temp, 99)

        im = ax.imshow(region_temp, origin="lower", cmap="inferno", vmin=vmin, vmax=vmax)

        plt.colorbar(im, ax=ax, label="Temperature (K)")
        ax.set_title("Brightness Temperature Map")
        ax.set_xlabel("RA (J2000)")
        ax.set_ylabel("Dec (J2000)")

        plt.show()

    elif choice == "7":
        print("Generating average spectrum...")

        spectrum, hdr = compute_average_spectrum(url)

        import numpy as np
        import matplotlib.pyplot as plt

        # --- Build frequency axis ---
        crval3 = hdr['CRVAL3']
        cdelt3 = hdr['CDELT3']
        crpix3 = hdr['CRPIX3']

        nchan = len(spectrum)
        freq = crval3 + (np.arange(nchan) - (crpix3 - 1)) * cdelt3

        # Convert to GHz (better for astronomy plots)
        freq = freq / 1e9

        #Check units and convert to brightness temperature if needed
        bunit = hdr.get("BUNIT", "").strip().lower()

        if "jy/beam" in bunit:
            print("Converting Jy/beam to Brightness Temperature (K)")

            # Beam parameters (deg → arcsec)
            bmaj = hdr['BMAJ'] * 3600
            bmin = hdr['BMIN'] * 3600

            # Convert to brightness temperature
            Tb = (1.222e6 * spectrum) / (freq**2 * bmaj * bmin)

            y = Tb
            ylabel = "Brightness Temperature (K)"

        elif "k" in bunit:
            print("Data already in brightness temperature")

            y = spectrum
            ylabel = "Brightness Temperature (K)"

        else:
            print(f"Unknown unit: {bunit} — leaving as-is")

            y = spectrum
            ylabel = f"Intensity ({hdr.get('BUNIT', 'Unknown')})"

        # --- Plot ---
        plt.figure(figsize=(8, 5))
        plt.plot(freq, y, color='black')

        plt.title("Average Spectrum", fontsize=25)
        plt.xlabel("Frequency (GHz)", fontsize=20)
        plt.ylabel(ylabel, fontsize=20)
        plt.show()
