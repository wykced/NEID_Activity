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
def fit_spectrum(wavelengths, flux):   
    
    # Filter out outliers
    flux_filtered = sigma_clip(flux, sigma=10) 
   
    # Remove NaNs/infs
    valid = np.isfinite(flux_filtered) & np.isfinite(wavelengths)
    lambda_valid = wavelengths[valid] * u.AA
    flux_valid = flux_filtered[valid] + 10
    
# Continuum 
    # Initial guess at Gaussian parameters and offsets
    an_amplitude2 = flux_valid.max()
    an_mean2 = lambda_valid[flux_valid.argmax()]
    an_stddev2 = np.sqrt(np.sum((lambda_valid - an_mean2)**2) / (len(lambda_valid) - 1))
    # Create continuum gaussian
    gaussian_continuum = models.Gaussian1D(an_amplitude2,
                            an_mean2,
                            an_stddev2)

    fit_g = modeling.fitting.LevMarLSQFitter()

    continuum_fit = fit_g(gaussian_continuum, lambda_valid, flux_valid)

    # Normalize
    flux_corrected = flux_valid / continuum_fit(lambda_valid)
    continuum_level = np.median(flux_corrected)
    flux_norm = (flux_corrected / continuum_level)

    return continuum_fit, lambda_valid, flux_valid, flux_norm

#%%
# Set folder and output
folder_path = path + 'TIC_63939535'
output_file = folder_path + 'stacked_deblazed.fits'

all_waves, all_fluxes = [], []
meta_headers = []

for order in range(122):  # NEID echelle orders are typically 0–121
    try:
        wave, flux, hdr = get_spectral_order(folder_path, order)
        continuum_fit, lambda_valid, flux_valid, flux_norm = fit_spectrum(wave, flux)
        all_waves.append(lambda_valid)
        all_fluxes.append(flux_norm)
        meta_headers.append(hdr)
        print(f"Processed order {order}")
    except Exception as e:
        print(f"Skipping order {order}: {e}")

# Determine max pixel count across orders
max_len = max(len(arr) for arr in all_waves)

# Pad each order with NaN to reach same length
waves_padded = np.full((len(all_waves), max_len), np.nan)
fluxes_padded = np.full((len(all_fluxes), max_len), np.nan)

for i, (w, f) in enumerate(zip(all_waves, all_fluxes)):
    n = len(w)
    waves_padded[i, :n] = w
    fluxes_padded[i, :n] = f

# Now safely convert to numpy arrays
all_waves = waves_padded
all_fluxes = fluxes_padded


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

# %%
from astropy import conf
conf.max_lines = 1000      # Show more rows if needed
conf.max_width = 2000      # Show full table width (characters)
conf.max_columns = 1000    # Show all columns
#%%
# Path 
stacked_file = output_file

hdul = fits.open(stacked_file)
hdul.info()

# Extract arrays
waves = hdul['WAVELENGTH'].data   # shape (122, Npix)
fluxes = hdul['FLUX'].data        # shape (122, Npix)
info = hdul['ORDER_INFO'].data

# Info
print(f"Shape of WAVELENGTH array: {waves.shape}")
print(f"Shape of FLUX array: {fluxes.shape}")
order_info_table = Table(info)
print(order_info_table[-10:])
print(order_info_table[:53])
print({info.shape})

# Orders: H-alpha (80), H-beta (47), Ca II H&K (18 and 17), Ca II Triplet (101 and 102)
#         Na I Doublet (69), H-Gamma (32)
# Vacuum Wavelengths: H-alpha (6564.614), H-beta (4862.721), Ca II H&K (3969.29, 3934.77), Ca II Triplet (8500.35, 8544.44, and 8664.52)
#                     Na I Doublet (5891.58, 5897.56), H-gamma (4340.46)
order = 80
vacuum = 6564.614

wave = waves[order]
flux = fluxes[order]

# Plot
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
