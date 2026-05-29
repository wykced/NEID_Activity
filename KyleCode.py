
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
        spectrum = hdul[1].data[echelle_order]
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

        return wavelengths, spectrum, header
    
        # rv_correction = np.mean(list(hdul[0].header['SSBRV*'].values()))
    
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
        gaussian_continuum = models.Gaussian1D(an_amplitude,
                            an_mean,
                            an_stddev)

        # g1_fit = fit_generic_continuum(spectrum)

    g_fit = fit_lines(spectrum, gaussian_continuum)
    y_continuum_fit = g_fit(lambda_fit).value

    return y_continuum_fit, continuum_level, lambda_fit, flux_fit, gaussian_continuum

def fit_spectral_feature(wavelengths, spectrum, region_min, region_max, vacuum_wavelength):
        # Remove NaNs/infs
    valid = np.isfinite(spectrum) & np.isfinite(wavelengths)
    lambda_fit = wavelengths[valid]
    flux_fit = spectrum[valid]
    
    # Mask H-alpha region
    mask = (lambda_fit > [region_min]*u.AA) & (lambda_fit < [region_max]*u.AA)

    lambda_mask = lambda_fit[mask]
    flux_mask = flux_fit[mask]

    # Normalize flux to continuum
    continuum_level_feature = np.median(flux_mask)
    flux_norm = flux_mask / continuum_level_feature

    # Invert absorption to fit as emission
    flux_inverted = 1 - flux_norm

    # Construct Spectrum1D for inverted profile
    spectrum = Spectrum1D(spectral_axis=lambda_mask, flux=flux_inverted * u.dimensionless_unscaled)

    #Initial guess at Gaussian parameters and offsets
    an_amplitude = np.max(flux_inverted)
    an_mean = [vacuum_wavelength] * u.AA
    an_stddev = np.sqrt(np.sum((lambda_mask - an_mean)**2) / (len(lambda_mask) - 1))
    # an_stddev = .3
    print(an_stddev)

    # Fit inverted Gaussian
    gaussian_feature = models.Gaussian1D(an_amplitude,
                            an_mean,
                            an_stddev)

    g_fit = fit_lines(spectrum, gaussian_feature)

    # Evaluate and un-invert
    y_feature_fit = 1 - g_fit(lambda_mask).value

    return y_feature_fit, continuum_level_feature, lambda_mask, gaussian_feature, mask

def fit_spectrum(lambda_fit, flux_fit):
    g_init = (gaussian_continuum + gaussian_feature) 
    # g_init.bounding_box=(-10,10) #force fit to be within certain bounds

    #fit model
    fit_g = modeling.fitting.LevMarLSQFitter()
    return fit_g(g_init, lambda_fit, flux_fit)


#%%

for file, star in zip([path + file for file in files], tic):
    rvs, ccf, header = get_ccf(file)
    wavelengths, spectrum, header = get_spectral_order(file, 80)
    y_continuum_fit, continuum_level, lambda_fit, flux_fit, gaussian_continuum = fit_continuum(wavelengths, spectrum)
    y_feature_fit, continuum_level_feature, lambda_mask, gaussian_feature,mask = fit_spectral_feature(wavelengths, spectrum, 6562, 6567, 6564.614)
    
    fig, ax = plt.subplots()
    ax.scatter(lambda_mask,flux_fit[mask],s=1)
    ax.plot(lambda_mask,y_feature_fit)
    
    g = fit_spectrum(lambda_fit, flux_fit)

    fig, ax = plt.subplots(1,figsize =(12,4))

    ax.set_title(f"Star: {header['OBJECT']}   Date: {header['OBSJD']}   RVS: {header['CCFRVMOD']}")
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.grid(which='both')
    ax.plot(wavelengths, spectrum,linewidth=.6)
    # ax.plot(lambda_mask, y_feature_fit * continuum_level_feature)
    # ax.plot(lambda_fit, y_continuum_fit * continuum_level)
    ax.plot(lambda_fit, g(lambda_fit))
    ax.axvline(x=6564.614, color='red', linestyle='--')
    # ax.set_xlim(6555, 6575)

    plt.tight_layout()
# %%