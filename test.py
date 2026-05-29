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

import spectra_helper
from spectra_helper import get_ccf, get_spectral_order, fit_continuum, fit_spectral_feature

#%%
path = '/Users/brooklyngustaf/Desktop/research/Brooklyn_NEID_spectra_project/l2fits/'

files=['neidL2_20241129T092548.fits', 
       'neidL2_20250207T114935.fits', 
       'neidL2_20250510T041611.fits', 
       'neidL2_20250522T112122.fits']

stars=['2M16294740+3941054', '2M16010481+2639362', '2M07463186+6548350', '2M07515777+1807352']
tic = ['TIC 63082902', 'TIC 255928963', 'TIC 393275621', 'TIC 462612605']
#%%
for file, star in zip([path + file for file in files], tic):
    rvs, ccf, header = get_ccf(file)
    wavelengths, spectrum, header = get_spectral_order(file,80)
    y_continuum_fit, continuum_level, lambda_fit = fit_continuum(wavelengths, spectrum)
    y_feature_fit, continuum_level, lambda_mask = fit_spectral_feature(wavelengths, spectrum, 6562, 6567, 6564.614)

    fig, ax = plt.subplots(1,figsize =(12,4))

    ax.set_title(f"Star: {header['OBJECT']}   Date: {header['OBSJD']}   RVS: {header['CCFRVMOD']}")
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.grid(which='both')
    ax.plot(wavelengths, spectrum,linewidth=.6)
    ax.plot(lambda_mask, y_feature_fit * continuum_level)
    ax.axvline(x=6564.614, color='red', linestyle='--')
    ax.set_xlim(6555, 6575)

    plt.tight_layout()


#%%
##### Full order plot
for file, star in zip([path + file for file in files], tic):
    rvs, ccf, header = get_ccf(file)
    wavelengths, spectrum, header = get_spectral_order(file,80)

    # Remove NaNs/infs
    valid = np.isfinite(spectrum) & np.isfinite(wavelengths)
    lambda_fit = wavelengths[valid]
    flux_fit = spectrum[valid]

    # Normalize flux to continuum
    continuum_level = np.median(flux_fit)
    flux_norm = flux_fit / continuum_level

    # Construct Spectrum1D
    spectrum = Spectrum1D(flux=flux_norm*u.Jy, spectral_axis=lambda_fit*u.dimensionless_unscaled)

    #initial guess at Gaussian parameters and offsets
    an_amplitude = flux_norm.max()
    an_mean = lambda_fit[flux_norm.argmax()]
    an_stddev = np.sqrt(np.sum((lambda_fit - an_mean)**2) / (len(lambda_fit) - 1))

    with warnings.catch_warnings():  # Ignore warnings
        warnings.simplefilter('ignore')
        g1_fit = models.Gaussian1D(an_amplitude,
                            an_mean,
                            an_stddev)

    g_fit = fit_lines(spectrum, g1_fit)


    y_continuum_fitted = g_fit(lambda_fit).value

    fig, ax = plt.subplots(1,figsize =(12,4))

    ax.set_title(f"Star: {header['OBJECT']}   Date: {header['OBSJD']}   RVS: {header['CCFRVMOD']}")
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.grid(which='both')
    ax.plot(lambda_fit, flux_fit,linewidth=.6)
    ax.axvline(x=6564.614, color='red', linestyle='--')
    ax.plot(lambda_fit, y_continuum_fitted * continuum_level)

    plt.tight_layout()
#%%
##### Zoomed in plot w RV correction
for file, star in zip([path + file for file in files], tic):
    rvs, ccf, header = get_ccf(file)
    wavelengths, spectrum, header = get_spectral_order(file,80)

    fig, ax = plt.subplots(1,figsize =(12,4))

    ax.set_title(f"Star: {header['OBJECT']}   Date: {header['OBSJD']}   RVS: {header['CCFRVMOD']}")
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.grid(True)

    c_kms = c.to('km/s').value  # Speed of light in km/s
    lambda_obs = wavelengths * u.AA
    flux = spectrum

    #Accounts for the Earth moving around the Sun and its velocity relative to the direction of the star at the time of observation
    rv = header['CCFRVMOD'] - header['rv_correction'] # km/s
        #CCFRVMOD is the most accurate RV for the star itself, but rv_correction (see spectra_helper) is more accurate for the specific order

    # Doppler shift correction
    lambda_rest = lambda_obs / (1 + (rv / c_kms))

    # Mask H-alpha region
    mask = (lambda_rest > 6562*u.AA) & (lambda_rest < 6567*u.AA)
    lambda_fit = lambda_rest[mask]
    flux_fit = flux[mask]

    # Remove NaNs/infs
    valid = np.isfinite(flux_fit) & np.isfinite(lambda_fit)
    lambda_fit = lambda_fit[valid]
    flux_fit = flux_fit[valid]

    # Normalize flux to continuum
    continuum_level = np.median(flux_fit)
    flux_norm = flux_fit / continuum_level

    # Invert absorption to fit as emission
    flux_inverted = 1 - flux_norm

    # Construct Spectrum1D for inverted profile
    spectrum = Spectrum1D(spectral_axis=lambda_fit, flux=flux_inverted * u.dimensionless_unscaled)

    # Fit inverted Gaussian
    g_init = models.Gaussian1D(amplitude=np.max(flux_inverted),
                            mean=6564.614 * u.AA,
                            stddev=0.3 * u.AA)

    g_fit = fit_lines(spectrum, g_init)

    # Evaluate and un-invert
    y_fit = 1 - g_fit(lambda_fit).value

    # Plot
    ax.plot(lambda_rest, flux, label='Rest-frame spectrum', linewidth=0.6)
    ax.plot(lambda_fit, y_fit * continuum_level, 'g-', label='Improved Gaussian fit')  # Scale back to original flux

    ax.axvline(x=6564.614, color='red', linestyle='--')
    ax.set_xlim(6555, 6575)  # Rest frame H-alpha region
    ax.set_xlabel('Wavelength (Å, Rest Frame)')
    ax.set_ylabel('Flux')

    plt.tight_layout()
    # plt.savefig(f'/Users/brooklyngustaf/Desktop/research/plots/{star}_narrow_order_{echelle_order}.png')

# %%
