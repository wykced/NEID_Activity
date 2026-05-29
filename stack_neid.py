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

plt.rcParams.update({
    "text.usetex": False,
    "font.serif": ["Computer Modern"]
})


path = '/Users/brooklyngustaf/Desktop/research/Brooklyn_NEID_spectra_project/l2fits/'

stars=['2M07515777+1807352', '2M16294740+3941054', '2M07463186+6548350', '2M16010481+2639362']
tic = ['TIC 63082902', 'TIC 255928963', 'TIC 393275621', 'TIC 462612605']

#%%
def get_spectral_order(fits_file, echelle_order):
    all_fluxes = []
    all_wavelengths = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".fits"):
            file_path = os.path.join(folder_path, filename)
            with fits.open(file_path) as hdul:
                header = dict(hdul[12].header['*DVR\D+|*RV\D+|*CCFJDSUM*|ccfrvmod|'].items())
                header.update(dict(hdul[0].header['OBJECT*|*DATE-OBS|*SCI-OBJ|*QRA|*QDEC|OBSJD|*QSPT'].items()))
                header.update({'rv_correction':hdul[0].header[f'SSBRV{173 - echelle_order:03d}']})
                c_kms = c.to('km/s').value  # Speed of light in km/s
                rv = header['CCFRVMOD'] - header['rv_correction'] #correcting from heliocentric rv
                wave = (hdul[7].data[echelle_order] * u.AA) / (1 + (rv / c_kms)) #doppler shift correction
                flux = hdul[1].data[echelle_order] 
                all_wavelengths.append(wave) 
                all_fluxes.append(flux)
                print(rv)

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

# def fit_spectrum(wavelengths, flux):   
    
#     # Filter out outliers
#     flux_filtered = sigma_clip(flux, sigma=10) 
   
#     # Remove NaNs/infs
#     valid = np.isfinite(flux_filtered) & np.isfinite(wavelengths)
#     lambda_valid = wavelengths[valid] * u.AA
#     flux_valid = flux_filtered[valid] + 10
    
# # Continuum 
#     # Initial guess at Gaussian parameters and offsets
#     an_amplitude2 = flux_valid.max()
#     an_mean2 = lambda_valid[flux_valid.argmax()]
#     an_stddev2 = np.sqrt(np.sum((lambda_valid - an_mean2)**2) / (len(lambda_valid) - 1))
#     # Create continuum gaussian
#     gaussian_continuum = models.Gaussian1D(an_amplitude2,
#                             an_mean2,
#                             an_stddev2)

#     fit_g = modeling.fitting.LevMarLSQFitter()

#     continuum_fit = fit_g(gaussian_continuum, lambda_valid, flux_valid)

#     # Normalize
#     flux_corrected = flux_valid / continuum_fit(lambda_valid)
#     continuum_level = np.median(flux_corrected)
#     flux_norm = (flux_corrected / continuum_level)

#     return continuum_fit, lambda_valid, flux_valid, flux_norm

# def fit_spectrum(wavelengths, flux):   
    
#     # Filter out outliers
#     flux_filtered = sigma_clip(flux, sigma=10) 
   
#     # Remove NaNs/infs
#     valid = np.isfinite(flux_filtered) & np.isfinite(wavelengths)
#     lambda_valid = wavelengths[valid] * u.AA
#     flux_valid = flux_filtered[valid] + 10
    
# # Continuum 
#     # Initial guess at Gaussian parameters and offsets
#     an_amplitude2 = flux_valid.max()
#     an_mean2 = lambda_valid[flux_valid.argmax()]
#     an_stddev2 = np.sqrt(np.sum((lambda_valid - an_mean2)**2) / (len(lambda_valid) - 1))
#     # Create continuum gaussian
#     gaussian_continuum = models.Gaussian1D(an_amplitude2,
#                             an_mean2,
#                             an_stddev2)

#     fit_g = modeling.fitting.LevMarLSQFitter()
#     p_init = models.Polynomial1D(degree=5)
#     continuum_fit = fit_g(p_init, lambda_valid, flux_valid)

#     # Normalize
#     flux_corrected = flux_valid / continuum_fit(lambda_valid)
#     continuum_level = np.median(flux_corrected)
#     flux_norm = (flux_corrected / continuum_level)

#     return continuum_fit, lambda_valid, flux_valid, flux_norm

def fit_spectrum(wavelengths, flux, poly_degree=10, n_iter=6, sigma=2.0):

    # --- Remove NaNs / Infs ---
    valid = np.isfinite(wavelengths) & np.isfinite(flux)
    lambda_valid = wavelengths[valid]
    flux_valid = flux[valid]

    # Optional: small offset if needed
    flux_valid = flux_valid + 10

    # --- Initial mask (start with everything) ---
    mask = np.ones(len(lambda_valid), dtype=bool)

    fit_g = modeling.fitting.LevMarLSQFitter()

    for _ in range(n_iter):

        # Fit polynomial to currently accepted pixels
        p_init = models.Polynomial1D(degree=poly_degree)
        continuum_fit = fit_g(p_init,
                              lambda_valid[mask],
                              flux_valid[mask])

        # Compute residuals
        model_flux = continuum_fit(lambda_valid)
        residuals = flux_valid - model_flux

        # Estimate scatter robustly
        std = np.std(residuals[mask])

        # Reject only absorption features (negative residuals)
        new_mask = residuals > -sigma * std

        # Stop if mask stops changing
        if np.all(new_mask == mask):
            break

        mask = new_mask

    # --- Final continuum ---
    continuum = continuum_fit(lambda_valid)

    # --- Normalize ---
    flux_norm = flux_valid / continuum

    return continuum_fit, lambda_valid, flux_valid, flux_norm

def fit_feature(region_min, region_max, vacuum_wavelength):    
    mask = (lambda_valid > region_min * u.AA) & (lambda_valid < region_max * u.AA)

    lambda_mask = lambda_valid[mask]
    flux_mask = flux_norm[mask]

    an_amplitude = np.min(flux_mask)
    an_mean = vacuum_wavelength * u.AA
    an_stddev = np.std((lambda_mask - an_mean).to(u.AA).value)
    an_disp = np.max(flux_mask)

    gaussian_feature = (models.Const1D(an_disp) +
                        models.Gaussian1D(amplitude=(an_amplitude - an_disp),
                                          mean=an_mean.value,
                                          stddev=an_stddev))

    fit_g = modeling.fitting.LevMarLSQFitter()
    feature_fit = fit_g(gaussian_feature, lambda_mask, flux_mask)

    return feature_fit, lambda_mask, flux_mask

ews = []
def get_eqs(order, vacuum):
    spectrum = Spectrum1D(spectral_axis=lambda_mask, flux=feature_fit(lambda_mask) * u.dimensionless_unscaled)

    ew = equivalent_width(spectrum, continuum = np.max(feature_fit(lambda_mask)), regions = None, mask_interpolation = None)
    ews.append(ew)

    return ews

# %%
# Orders: H-alpha (80), H-beta (47), Ca II H&K (18 and 17), Ca II Triplet (101 and 102)
#         Na I Doublet (69), H-Gamma (32)
# Vacuum Wavelengths: H-alpha (6564.614), H-beta (4862.721), Ca II H&K (3969.29, 3934.77), Ca II Triplet (8500.35, 8544.44, and 8664.52)
#                     Na I Doublet (5891.58, 5897.56), H-gamma (4340.46)

# Files: 'TIC_462612605_all_fits', 'TIC_393275621_all_fits', 'TIC_236017803_all_fits', 'TIC_255928963_all_fits'
# Files: 'TIC_38376846_all_fits', 'TIC_116415110_all_fits', 'TIC_63082902_all_fits'
folder_path = path + 'TIC_22903436'


# echelle_order = [80]
# vacuum = [(5891.58+5897.56)/2]
# vacuum = [6564.614]

# Mg I Triplet
# echelle_order = [55, 55, 55]
# vacuum = [5168.76, 5174.12, 5185.05]

echelle_order = [17, 18, 32, 47, 68, 80, 101, 101, 102]
vacuum = [3934.77, 3969.29, 4341.68, 4862.721, 5855.32, 6564.614, 8500.35, 8544.44, 8664.52]
labels = ['CaII H', 'CaII K', 'H Gamma', 'H Beta', 'Ba', 'H Alpha', 'CaII IRT 1', 'CaII IRT 2', 'CaII IRT 3']

# for echelle_order, vacuum, labels in zip(echelle_order, vacuum, labels):
for echelle_order, vacuum in zip(echelle_order, vacuum):
    wavelengths, flux, header = get_spectral_order(folder_path, echelle_order)
    continuum_fit, lambda_valid, flux_valid, flux_norm = fit_spectrum(wavelengths, flux)
    # feature_fit, lambda_mask, flux_mask = fit_feature(vacuum - 5, vacuum + 5, vacuum)
    # ews = get_eqs(echelle_order, vacuum)
            
    fig, ax = plt.subplots(1, figsize =(12,4))
    ax.xaxis.set_minor_locator(MultipleLocator(0.5))
    # ax.grid(which='major')
    ax.plot(lambda_valid, flux_norm, linewidth=.6) 
    # ax.plot(lambda_mask, feature_fit(lambda_mask), color='black')
    # ax.plot(lambda_valid, continuum_fit(lambda_valid), color='black')              
    ax.set_title(f"Star: {header['OBJECT']}")
    ax.axvline(x=vacuum, color='green', linestyle='--', linewidth = 0.6)
    # ax.axvline(x=8544.44, color='green', linestyle='--')
    ax.set_xlim(vacuum - 10, vacuum + 10)
    # ax.set_xlim(8490, 8555)
    # ax.axhline(y=1, color='darkgray', linestyle='--', linewidth=.6)
    # x_coords = [vacuum-(0.94641008/2), vacuum-(0.94641008/2), vacuum+(0.94641008/2), vacuum+(0.94641008/2)]  # x-coordinates of the corners
    # y_coords = [1, 0, 0, 1]  # y-coordinates of the corners

    # plt.fill(x_coords, y_coords, color='lightgray', alpha=0.7)
    ax.set_ylim(0, 1.5)
    ax.legend()
    ax.set_xlabel("Wavelength (Å)")
    ax.set_ylabel("Normalized Flux")
#%%
ews
# %%
