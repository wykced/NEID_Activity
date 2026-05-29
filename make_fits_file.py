#%%
#%%
import pandas as pd
import numpy as np
import warnings
from astropy.io import fits
from astropy import modeling
from matplotlib import pyplot as plt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
from astropy.constants import c
from astropy.modeling import models
from astropy import units as u
from specutils.spectra import Spectrum1D, SpectralRegion
from specutils.fitting import fit_lines, fit_generic_continuum
from specutils.analysis import equivalent_width
from specutils.fitting import find_lines_derivative
from astropy.stats import sigma_clip
from astropy.table import Table
import os

path = '/Users/brooklyngustaf/Desktop/research/Brooklyn_NEID_spectra_project/l2fits/'

#%%
def get_spectral_order(folder_path, echelle_order):
    all_fluxes = []
    all_wavelengths = []

    for filename in os.listdir(folder_path):
        if filename.endswith(".fits"):
            file_path = os.path.join(folder_path, filename)
            with fits.open(file_path) as hdul:
                # Minimal robust header extraction (safer than using dict on wildcards)
                hdr0 = hdul[0].header
                hdr12 = hdul[12].header
                rv_correction = hdr0.get(f'SSBRV{173 - echelle_order:03d}', 0)
                rv = hdr12.get('CCFRVMOD', 0) - rv_correction
                c_kms = c.to('km/s').value
                wave = (hdul[7].data[echelle_order] * u.AA) / (1 + (rv / c_kms))
                flux = hdul[1].data[echelle_order]
                all_wavelengths.append(wave.value)
                all_fluxes.append(flux)
                header = {
                    'OBJECT': hdr0.get('OBJECT', 'UNKNOWN'),
                    'DATE-OBS': hdr0.get('DATE-OBS', 'UNKNOWN'),
                    'ORDER': echelle_order,
                    'RV_CORR': rv_correction,
                    'SSBRV': rv_correction,
                    'RV_FINAL': rv
                }

    # Convert to arrays
    all_fluxes = np.array(all_fluxes)
    all_wavelengths = np.array(all_wavelengths)

    # choose reference grid (e.g., first spectrum)
    wavelengths = all_wavelengths[0]

    resampled_fluxes = []

    for wave, flux in zip(all_wavelengths, all_fluxes):
        interp_flux = np.interp(wavelengths, wave, flux)
        resampled_fluxes.append(interp_flux)

    flux = np.mean(resampled_fluxes, axis=0)

    return wavelengths, flux, header

#%%
from astropy.modeling import models, fitting
from astropy.stats import sigma_clip
import numpy as np
import astropy.units as u


#%%
# Set folder and output
TIC = 'TIC_105485569'
folder_path = path + TIC
output_file = '/Users/brooklyngustaf/Desktop/research/Brooklyn_NEID_spectra_project/l2fits/Stacked/' + TIC + '_stacked.fits'

all_waves, all_fluxes = [], []
meta_headers = []

for order in range(122):  # NEID echelle orders are typically 0–121
    try:
        wave, flux, hdr = get_spectral_order(folder_path, order)
        all_waves.append(wave)
        all_fluxes.append(flux)
        meta_headers.append(hdr)
        print(f"Processed order {order}")
    except Exception as e:
        print(f"Skipping order {order}: {e}")

# Stack into arrays
all_waves = np.array(all_waves)
all_fluxes = np.array(all_fluxes)

#%%
# Primary HDU with general metadata
primary_hdu = fits.PrimaryHDU()
primary_hdu.header['NORDERS'] = len(all_fluxes)
primary_hdu.header['OBJECT'] = meta_headers[0]['OBJECT']
primary_hdu.header['DATE-OBS'] = meta_headers[0]['DATE-OBS']
primary_hdu.header['COMMENT'] = "Median-combined NEID orders, Doppler-corrected"

# Add each array as ImageHDUs
flux_hdu = fits.ImageHDU(all_fluxes, name='FLUX')
wave_hdu = fits.ImageHDU(all_waves, name='WAVELENGTH')

# Optionally store per-order RV info as a table
order_info = Table(meta_headers)
order_hdu = fits.BinTableHDU(order_info, name='ORDER_INFO')

# Write all to a single file
hdul = fits.HDUList([primary_hdu, wave_hdu, flux_hdu, order_hdu])
hdul.writeto(output_file, overwrite=True)

print(f"✅ Saved stacked FITS file: {output_file}")

#%%
# Path to the stacked file you made
stacked_file = '/Users/brooklyngustaf/Desktop/research/Brooklyn_NEID_spectra_project/l2fits/Stacked/' + TIC + '_stacked.fits'

# Open file
hdul = fits.open(stacked_file)

# Print contents to confirm HDU names
hdul.info()

# Extract arrays
waves = hdul['WAVELENGTH'].data   # shape (122, Npix)
fluxes = hdul['FLUX'].data        # shape (122, Npix)
info = hdul['ORDER_INFO'].data

# Optional: check number of orders and pixel count
print(f"Shape of WAVELENGTH array: {waves.shape}")
print(f"Shape of FLUX array: {fluxes.shape}")
order_info_table = Table(info)
print(order_info_table[-10:])
print(order_info_table[:53])
print({info.shape})

# Orders: H-alpha (80), H-beta (47), Ca II H&K (18 and 17), Ca II Triplet (101 and 102)
#         Na I Doublet (69)
# Vacuum Wavelengths: H-alpha (6564.614), H-beta (4862.721), Ca II H&K (3969.29, 3934.77), Ca II Triplet (8500.35, 8544.44, and 8664.52)
#                     Na I Doublet (5891.58, 5897.56)

order = 80
vacuum = 6564.614

wave = waves[order]
flux = fluxes[order]

# --- Plot the order ---
plt.figure(figsize=(12,4))
plt.plot(wave, flux, linewidth=0.6, color='royalblue')

plt.title(f"Echelle Order {order}")
plt.xlabel("Wavelength (Å)")
plt.ylabel("Flux")
plt.xlim(vacuum - 10, vacuum +10)
plt.axvline(x=vacuum, color='green', linestyle='--')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# Close file when done
hdul.close()

# %%
