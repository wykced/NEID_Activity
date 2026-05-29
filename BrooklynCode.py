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

path = '/Users/brooklyngustaf/Desktop/research/Brooklyn_NEID_spectra_project/l2fits/'

files=['neidL2_20241129T092548.fits', 
       'neidL2_20250207T114935.fits', 
       'neidL2_20250510T041611.fits', 
       'neidL2_20250522T112122.fits']

stars=['2M16294740+3941054', '2M16010481+2639362', '2M07463186+6548350', '2M07515777+1807352']
tic = ['TIC 63082902', 'TIC 255928963', 'TIC 393275621', 'TIC 462612605']

#%%

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
        flux = hdul[1].data[echelle_order]
        header = dict(hdul[12].header['*DVR\D+|*RV\D+|*CCFJDSUM*|ccfrvmod|'].items())
        header.update(dict(hdul[0].header['OBJECT*|*DATE-OBS|*SCI-OBJ|*QRA|*QDEC|OBSJD'].items()))
        header.update({'rv_correction':hdul[0].header[f'SSBRV{echelle_order:03d}']}) #specfic echelle order RV correction
        
        c_kms = c.to('km/s').value  # Speed of light in km/s
        lambda_obs = wavelengths * u.AA

        #CCFRVMOD rvs accounts for the Earth moving around the Sun and its velocity relative to the direction of the star at the time of observation
        #this means the rv is from the perspective of the sun, not the Earth. We want to subtract this correction off, but the correction is dependent on the echelle order
        rv = header['CCFRVMOD'] - header['rv_correction'] # km/s
        #CCFRVMOD is the most accurate RV for the star itself, but rv_correction is more accurate for the specific order

        # Doppler shift correction
        wavelengths = lambda_obs / (1 + (rv / c_kms))

        return wavelengths, flux, header

# %%
def fit_spectrum(wavelengths, spectrum, region_min, region_max, vacuum_wavelength):
#Spectral Feature    
    # Remove NaNs/infs
    valid = np.isfinite(spectrum) & np.isfinite(wavelengths)
    lambda_valid = wavelengths[valid]
    flux_valid = spectrum[valid]
    
    # Mask H-alpha region
    mask = (lambda_valid > [region_min]*u.AA) & (lambda_valid < [region_max]*u.AA)

    lambda_mask = lambda_valid[mask]
    flux_mask = flux_valid[mask]

    # Construct Spectrum1D for inverted profile
    spectrum = Spectrum1D(spectral_axis=lambda_mask, flux=flux_mask * u.dimensionless_unscaled)

    #Initial guess at Gaussian parameters and offsets
    an_amplitude = np.min(flux_mask)
    an_mean = [vacuum_wavelength] * u.AA
    an_stddev = np.sqrt(np.sum((lambda_mask - an_mean)**2) / (len(lambda_mask) - 1))
    an_disp = np.max(flux_mask)

    # Fit feature Gaussian
    gaussian_feature =  (models.Const1D(an_disp) +
                         models.Gaussian1D(an_amplitude - an_disp,
                            an_mean,
                            an_stddev))
    
#Continuum
    # Normalize flux to continuum
    continuum_level = np.median(flux_valid)
    flux_norm = flux_valid / continuum_level

    spectrum2 = Spectrum1D(flux=flux_norm*u.Jy, spectral_axis=lambda_valid*u.dimensionless_unscaled)
    
    #Initial guess at Gaussian parameters and offsets
    an_amplitude2 = flux_norm.max()
    an_mean2 = lambda_valid[flux_norm.argmax()]
    an_stddev2 = np.sqrt(np.sum((lambda_valid - an_mean)**2) / (len(lambda_valid) - 1))

    #Create continuum gaussian
    gaussian_continuum = models.Gaussian1D(an_amplitude2,
                            an_mean2,
                            an_stddev2)

    fit_g = modeling.fitting.LevMarLSQFitter()

    continuum_fit = fit_g(gaussian_continuum, lambda_valid, flux_valid)
    feature_fit = fit_g(gaussian_feature, lambda_mask, flux_mask)

    g_fit = fit_g((continuum_fit + feature_fit), lambda_valid, flux_valid)
    # g_fit = (continuum_fit + feature_fit)

    return feature_fit, continuum_fit, g_fit, lambda_mask, flux_mask, lambda_valid, flux_valid
# %%
for file, star in zip([path + file for file in files], tic):
    rvs, ccf, header = get_ccf(file)
    wavelengths, flux, header = get_spectral_order(file, 50)
    feature_fit, continuum_fit, g_fit, lambda_mask, flux_mask, lambda_valid, flux_valid = fit_spectrum(wavelengths, flux, 6562, 6567, 6564.614)

    fig, ax = plt.subplots(1,figsize =(12,4))
    ax.set_title(f"Star: {header['OBJECT']}   Date: {header['OBSJD']}   RVS: {header['CCFRVMOD']}")
    ax.xaxis.set_minor_locator(MultipleLocator(0.5))
    ax.grid(which='major')
    ax.plot(wavelengths, flux, linewidth=.6)
    ax.axvline(x=6564.614, color='green', linestyle='--')
    # ax.plot(lambda_mask, feature_fit(lambda_mask), label="Fit result")
    # ax.set_xlim(6555, 6575)
    # ax.plot(lambda_valid, continuum_fit(lambda_valid))
    # ax.plot(lambda_valid, g_fit(lambda_valid), c = 'red')


# %%
for file, star in zip([path + file for file in files], tic):
    rvs, ccf, header = get_ccf(file)
    wavelengths, flux, header = get_spectral_order(file, 80)
    feature_fit, continuum_fit, g_fit, lambda_mask, flux_mask, lambda_valid, flux_valid, spectrum = fit_spectrum(wavelengths, flux, 6562, 6567, 6564.614)    

    flux_corrected = flux_valid - continuum_fit(lambda_valid)
    continuum_level = np.max(flux_corrected)
    flux_norm = flux_corrected / continuum_level

    fig, ax = plt.subplots(1,figsize =(12,4))
    ax.plot(lambda_valid, flux_norm, linewidth=0.6)
    # ax.plot(lambda_mask, feature_fit(lambda_mask))
    ax.set_xlim(6555, 6575)
    # print(continuum_level)

    eq = equivalent_width(spectrum, continuum=0, regions=None, mask_interpolation=None)
    print(eq)
# %%
