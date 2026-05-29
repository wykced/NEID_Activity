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
from glob import glob
import os

#%%

def get_ccf(fits_file, reweight=True, orders = (0,122)):
    with fits.open(fits_file) as hdul:
        if reweight:
            ccf = np.sum([wt*hdul[12].data[i] for i, wt in enumerate(hdul[12].header['CCFW*'].values()) if type(wt)==float],axis=0)
        else:
            ccf = np.sum(hdul[12].data,axis=0)
        data_length = ccf.shape[0] 
        rvs = np.linspace(hdul[12].header['CCFSTART'],
                    hdul[12].header['CCFSTART']+(data_length-1)*hdul[12].header['CCFSTEP'],
                    data_length)
        header = dict(hdul[12].header['*DVR\D+|*RV\D+'].items())
        header.update(dict(hdul[0].header['OBJECT*|*DATE-OBS|*SCI-OBJ|*QRA|*QDEC|OBSJD'].items()))
    return rvs, ccf, header

def get_spectral_order(fits_file, echelle_order):
    with fits.open(fits_file) as hdul:
        wavelengths = hdul[7].data[echelle_order]
        flux = hdul[1].data[echelle_order]
        header = dict(hdul[12].header['*DVR\D+|*RV\D+|*CCFJDSUM*|ccfrvmod|'].items())
        header.update(dict(hdul[0].header['OBJECT*|*DATE-OBS|*SCI-OBJ|*QRA|*QDEC|OBSJD|*QSPT'].items()))
        header.update({'rv_correction':hdul[0].header[f'SSBRV{173 - echelle_order:03d}']}) #specfic echelle order RV correction
        activity = hdul[13].data

        c_kms = c.to('km/s').value  # Speed of light in km/s
        lambda_obs = wavelengths * u.AA

        # CCFRVMOD rvs accounts for the Earth moving around the Sun and its velocity relative to the direction of the star at the time of observation
        # this means the rv is from the perspective of the sun, not the Earth. We want to subtract this correction off, but the correction is dependent on the echelle order
        rv = header['CCFRVMOD'] - header['rv_correction'] # km/s
        # CCFRVMOD is the most accurate RV for the star itself, but rv_correction is more accurate for the specific order
        # Doppler shift correction
        wavelengths = lambda_obs / (1 + (rv / c_kms))

        return wavelengths, flux, header, activity, rv

# %%
def fit_spectrum(wavelengths, flux):   
    
    # Filter out outliers
    flux_filtered = sigma_clip(flux, sigma=10) 
   
    # Remove NaNs/infs
    valid = np.isfinite(flux_filtered) & np.isfinite(wavelengths)
    lambda_valid = wavelengths[valid]
    flux_valid = flux_filtered[valid]
    warnings.filterwarnings("ignore")
    
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

def fit_feature(region_min, region_max, vacuum_wavelength):   
# Spectral Feature 
    # Mask H-alpha region
    mask = (lambda_valid > [region_min]*u.AA) & (lambda_valid < [region_max]*u.AA)

    lambda_mask = lambda_valid[mask]
    flux_mask = flux_norm[mask]

    # Initial guess at Gaussian parameters and offsets
    an_amplitude = np.min(flux_mask) 
    an_mean = [vacuum_wavelength] * u.AA
    an_stddev = np.sqrt(np.sum((lambda_mask - an_mean)**2) / (len(lambda_mask) - 1))
    an_disp = np.max(flux_mask)

    # Fit feature Gaussian
    gaussian_feature =  (models.Const1D(an_disp) +
                         models.Gaussian1D(an_amplitude - an_disp,
                            an_mean,
                            an_stddev))

    fit_g = modeling.fitting.LevMarLSQFitter()
    feature_fit = fit_g(gaussian_feature, lambda_mask, flux_mask)

    # g_fit = fit_g((continuum_fit + feature_fit), lambda_valid, flux_valid)

    return feature_fit, lambda_mask, flux_mask

ews = []
feature_type = []
def get_eqs(order, vacuum):
    spectrum = Spectrum1D(spectral_axis=lambda_mask, flux=feature_fit(lambda_mask) * u.dimensionless_unscaled)

    ew = equivalent_width(spectrum, continuum = np.max(feature_fit(lambda_mask)), regions = None, mask_interpolation = None)
    ews.append(ew)

    return ews

#%%
def plot_spectrum(order, vacuum):
    fig, ax = plt.subplots(1,figsize =(12,4))
    ax.set_title(f"Star: {header['OBJECT']}   Date: {header['OBSJD']}   RVS: {header['CCFRVMOD']}")
    ax.xaxis.set_minor_locator(MultipleLocator(0.5))
    ax.grid(which='major')
    ax.plot(lambda_valid, flux_norm, linewidth=.6, label='Normalized Spectrum')
    ax.axvline(x=vacuum, color='green', linestyle='--')
    ax.set_xlim(vacuum - 10, vacuum + 10)
    ax.set_ylim(-1, 4)
    # ax.plot(lambda_mask, feature_fit(lambda_mask), c = 'red', label='Feature Fit')
    # ax.plot(lambda_valid, continuum_fit(lambda_valid))
    # ax.plot(lambda_valid, g_fit(lambda_valid), c = 'red')
    # ax.legend()

#%%

path = '/Users/brooklyngustaf/Desktop/research/Brooklyn_NEID_spectra_project/l2fits/'

star = 'TIC_136954576'

folder = path + star + '/'
files = sorted(glob(os.path.join(folder, '*.fits')))

# Orders: H-alpha (80), H-beta (47), Ca II H&K (18 and 17), Ca II Triplet (101 and 102)
# Vacuum Wavelengths: H-alpha (6564.614), H-beta (4862.721), Ca II H&K (3969.29, 3934.77), Ca II Triplet (8500.35, 8544.44, and 8664.52)

order = 80
vacuum = 6564.614

import math

num_files = len(files)
ncols = 2
nrows = math.ceil(num_files / ncols)

fig, axes = plt.subplots(
    nrows,
    ncols,
    figsize=(24, 4 * nrows),
    sharex=True,
    sharey=True
)

axes = axes.flatten()  # makes axes[i] work cleanly


# # Prepare subplots: one row per file
# num_files = len(files)
# fig, axes = plt.subplots(num_files, 1, figsize=(12, 4 * num_files), sharex=True)

if num_files == 1:
    axes = [axes]  # ensure axes is iterable even if only one plot

for i, file in enumerate(files):
    rvs, ccf, header = get_ccf(file)
    wavelengths, flux, header, activity, rv = get_spectral_order(file, order)
    continuum_fit, lambda_valid, flux_valid, flux_norm = fit_spectrum(wavelengths, flux)
    # feature_fit, lambda_mask, flux_mask = fit_feature(vacuum - 5, vacuum + 5, vacuum)
    # ews = get_eqs(order, vacuum)


    ax = axes[i]
    ax.set_title(f"Star: {header['OBJECT']}   Date: {header['OBSJD']}   RVS: {rv}\nfile = {files[i]}", fontsize=14)
    ax.xaxis.set_minor_locator(MultipleLocator(0.5))
    ax.grid(which='major')
    ax.plot(lambda_valid, flux_norm, linewidth=.6, label='Normalized Spectrum')
    ax.axvline(x=vacuum, color='green', linestyle='--')
    ax.set_xlim(vacuum - 10, vacuum + 10)
    # ax.set_xlim(6562.5, 6567)
    ax.set_ylim(-1, 5)
    if i == len(files) - 1:
        ax.set_xlabel("Wavelength (Å)")
    ax.set_ylabel("Normalized Flux")


plt.tight_layout()
plt.show()

# %%
h_alpha = {
    "Star": tic,
    "EW": ews,
}
df = pd.DataFrame(h_alpha)

styled_df = df.style.set_properties(**{'text-align': 'left'}) \
                    .set_table_styles([dict(selector='th', props=[('text-align', 'left')])])

styled_df
# %%
air = 5183.6042
s = 10**4/air
n = 1 + 0.00008336624212083 + 0.02408926869968 / (130.1065924522 - s**2) + 0.0001599740894897 / (38.92568793293 - s**2)
vac = air * n
print(vac)
# %%

# %%