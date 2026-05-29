"""Below are functions that help work with NEID fits files, including spectra and ccfs

Example to import:

from spectra_helper import get_ccf, get_spectral_order
file = 'l2fits/neidL2_20240310T102910.fits'
rvs, ccf, header = get_ccf(file)
wavelengths, spectrum, header = get_spectral_order(file,80)
"""
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

def get_ccf(fits_file, reweight=True, orders = (0,120)):
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
        spectrum = hdul[1].data[echelle_order]
        header = dict(hdul[12].header['*DVR\D+|*RV\D+|*CCFJDSUM*|ccfrvmod|'].items())
        header.update(dict(hdul[0].header['OBJECT*|*DATE-OBS|*SCI-OBJ|*QRA|*QDEC|OBSJD'].items()))
        header.update({'rv_correction':hdul[0].header[f'SSBRV{echelle_order:03d}']}) #specfic echelle order RV correction
        
        c_kms = c.to('km/s').value  # Speed of light in km/s
        lambda_obs = wavelengths * u.AA

        #Accounts for the Earth moving around the Sun and its velocity relative to the direction of the star at the time of observation
        rv = header['CCFRVMOD'] - header['rv_correction'] # km/s
        #CCFRVMOD is the most accurate RV for the star itself, but rv_correction (see spectra_helper) is more accurate for the specific order

        # Doppler shift correction
        wavelengths = lambda_obs / (1 + (rv / c_kms))
        # rv_correction = np.mean(list(hdul[0].header['SSBRV*'].values()))
        return wavelengths, spectrum, header
    
def fit_continuum(wavelengths, spectrum):
    # Remove NaNs/infs
    valid = np.isfinite(spectrum) & np.isfinite(wavelengths)
    lambda_fit = wavelengths[valid]
    flux_fit = spectrum[valid]

    # Normalize flux to continuum
    continuum_level = np.median(flux_fit)
    flux_norm = flux_fit / continuum_level

    # Construct Spectrum1D
    spectrum = Spectrum1D(flux=flux_norm*u.Jy, spectral_axis=lambda_fit*u.dimensionless_unscaled)

    #Initial guess at Gaussian parameters and offsets
    an_amplitude = flux_norm.max()
    an_mean = lambda_fit[flux_norm.argmax()]
    an_stddev = np.sqrt(np.sum((lambda_fit - an_mean)**2) / (len(lambda_fit) - 1))

    #Create model
    with warnings.catch_warnings():  # Ignore warnings
        warnings.simplefilter('ignore')
        g1_fit = models.Gaussian1D(an_amplitude,
                            an_mean,
                            an_stddev)
    g_fit = fit_lines(spectrum, g1_fit)
    y_continuum_fit = g_fit(lambda_fit).value

    return y_continuum_fit, continuum_level, lambda_fit

def fit_spectral_feature(wavelengths, spectrum, region_min, region_max, vacuum_wavelength):
        # Remove NaNs/infs
    valid = np.isfinite(spectrum) & np.isfinite(wavelengths)
    lambda_fit = wavelengths[valid]
    flux_fit = spectrum[valid]
    
    # Mask H-alpha region
    mask = (lambda_fit > [region_min]*u.AA) & (lambda_fit < [region_max]*u.AA)
    print(mask)
    lambda_mask = lambda_fit[mask]
    flux_mask = flux_fit[mask]

    # Normalize flux to continuum
    continuum_level = np.median(flux_mask)
    flux_norm = flux_mask / continuum_level

    # Invert absorption to fit as emission
    flux_inverted = 1 - flux_norm

    # Construct Spectrum1D for inverted profile
    spectrum = Spectrum1D(spectral_axis=lambda_mask, flux=flux_inverted * u.dimensionless_unscaled)

    # Fit inverted Gaussian
    g_init = models.Gaussian1D(amplitude=np.max(flux_inverted),
                            mean=[vacuum_wavelength] * u.AA,
                            stddev=0.3 * u.AA)

    g_fit = fit_lines(spectrum, g_init)

    # Evaluate and un-invert
    y_feature_fit = 1 - g_fit(lambda_mask).value

    return y_feature_fit, continuum_level, lambda_mask

# if __name__ == '__main__':
#     file = 'l2fits/neidL2_20240310T102910.fits'
#     get_spectral_order(file,80)

# %%
