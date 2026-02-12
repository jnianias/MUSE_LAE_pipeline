#!/usr/bin/env python3
# coding: utf-8

"""
Script to analyse MUSE LAE spectra (overhauled from previous version).
Usage: 
    S01_MUSE_LAE_spectral_fitting.py <CLUSTER_NAME> <SPEC_SOURCE> <SPEC_TYPE>

Arguments:
    <CLUSTER_NAME> : Name of the cluster (e.g., 'A2744', 'MACS0416NE', etc.)
    <SPEC_SOURCE> : Source of the spectrum data (R21 versus aperture extraction)
    <SPEC_TYPE> : Type of spectrum to use (e.g., 'noweight_skysub', '2fwhm', etc.)

Options:
    --aper-size <int> : Aperture size in FWHM (required for APER)
    --overwrite-spectra : Flag to overwrite existing extracted spectra
    --optimise-apertures : Which aperture optimisation to use ('auto', 'none', 'lya')
    --optimise-apertures-kwargs : Additional keyword arguments for lya peak detection (e.g., nstruct, niter)
    -h, --help : Show this help message and exit
    --plot-images : whether to plot aperture placement diagnostic images

Description:
    This script refits stacked spectra of Lyman-alpha emitters (LAEs) in MUSE clusters using initial guesses from existing catalogues.
    It supports both R21 spectra and aperture-extracted spectra.
    The script generates fit results for Lyman-alpha and other spectral lines, saving the results to disk.
"""

import argparse
from collections import namedtuple
import numpy as np
import astro_utils.io as auio
import astro_utils.constants as auconst
import astro_utils.image_processing as auip
import astro_utils.ifs as auifs
import astro_utils.catalogue_operations as aucat
import astro_utils.lya_fitting as lyafit
import astro_utils.fitting as aufit
from pathlib import Path
import ast
import astropy.table as aptb

wavedict = auconst.wavedict
doublet_dict = auconst.doublets

def get_R21_paths(source_cat, cluster_name, r21_type='noweight_skysub'):
    """
    Retrieves the file paths for R21 spectra for sources in the source catalog.

    Parameters
    ----------
    source_cat : list of dict
        List of source dictionaries containing source information.
    cluster_name : str
        Name of the cluster (e.g., 'A2744').
    r21_type : str, optional
        Type of R21 spectra to use. Default is 'noweight_skysub'.

    Returns
    -------
    dict
        Dictionary mapping source identifiers to their R21 spectrum file paths.
    """
    spec_paths = {}
    for source in source_cat:
        iden = source['iden']
        idfrom = source['idfrom']
        spec_path = auio.get_r21_spectra_dir(cluster_name)
        full_iden = f"{idfrom[0].replace('E', 'X')}{iden}"
        spec_file = Path(spec_path) / f"spec_{full_iden}_{r21_type}.fits"
        spec_paths[full_iden] = spec_file
    return spec_paths

def check_spectra_availability(spec_paths):
    """
    Checks if the spectra files exist for all sources.

    Parameters
    ----------
    spec_paths : dict
        Dictionary mapping source identifiers to their spectrum file paths.

    Raises
    ------
    FileNotFoundError
        If any spectrum file is missing.
    """
    missing_spectra = [iden for iden, path in spec_paths.items() if not path.exists()]
    if missing_spectra:
        raise FileNotFoundError(f"Spectra missing for sources: {', '.join(missing_spectra)}")
    
def insert_lya_results(lya_results, full_iden, fit_results, mu, mu_err, ra_new, dec_new):
    """
    Inserts Lyman alpha fit results into the lya_results table.

    Parameters
    ----------
    lya_results : astropy.table.Table
        Lyman alpha results table.
    full_iden : str
        Full source identifier.
    fit_results : dict
        Dictionary containing fit results.

    Returns
    -------
    None
    """
    new_row = {}
    new_row['iden'] = full_iden
    new_row['LINE'] = 'LYALPHA'
    new_row['LBDA_REST'] = auconst.wavedict['LYALPHA']
    new_row['MU'] = mu
    new_row['RA'] = ra_new
    new_row['DEC'] = dec_new
    new_row['MU_ERR'] = mu_err
    new_row['Z'] = fit_results['param_dict'].get('LPEAKR', np.nan) / auconst.wavedict['LYALPHA'] - 1
    new_row['Z_ERR'] = (fit_results['error_dict'].get('LPEAKR', np.nan) / auconst.wavedict['LYALPHA'])
    new_row['SNRR'] = fit_results['param_dict'].get('FLUXR', np.nan) / fit_results['error_dict'].get('FLUXR', np.nan)
    new_row['SNRB'] = fit_results['param_dict'].get('FLUXB', np.nan) / fit_results['error_dict'].get('FLUXB', np.nan)
    # Approximate upper bound flux for blue component using the uncertainty on red component flux
    new_row['FLUXB_UB'] = fit_results['error_dict'].get('FLUXR', np.nan) * 3.0
    new_row['RCHSQ'] = fit_results.get('reduced_chisq', np.nan)
    new_row['FLAG'] = '' # no flagging has been done yet but will be in subsequent steps
    for key, value in fit_results['param_dict'].items():
        new_row[key] = value
        new_row[f"{key}_ERR"] = fit_results['error_dict'].get(key, np.nan)
    lya_results.add_row(new_row)

def insert_line_results(line_results, full_iden, line_name, fit_results, mu, mu_err):
    """
    Inserts line fit results into the line_results table. Handles both single lines and doublets.
    Secondary lines of doublets are added as separate rows, identical in all respects except for line name, flux
    and wavelengths

    Parameters
    ----------
    line_results : astropy.table.Table
        Line results table.
    full_iden : str
        Full source identifier.
    line_name : str
        Name of the spectral line.
    fit_results : dict
        Dictionary containing fit results.

    Returns
    -------
    None
    """
    rest_wave = auconst.wavedict[line_name]
    doublet = doublet_dict.get(line_name, (None, None))
    rest_wave_2 = wavedict.get(doublet[1])

    new_row_1 = {}
    new_row_2 = {} # For any secondary components (if not present, simply isn't added to the table)
    new_row_1['iden'] = full_iden
    new_row_2['iden'] = full_iden
    new_row_1['LINE'] = line_name
    new_row_2['LINE'] = doublet[1]  # gets secondary line name if it's a doublet, else 'None'
    new_row_1['LBDA_REST'] = rest_wave
    new_row_2['LBDA_REST'] = rest_wave_2
    new_row_1['FLAG'] = fit_results.get('multipeak_flag', '')
    new_row_2['FLAG'] = fit_results.get('multipeak_flag', '')
    new_row_1['RCHSQ'] = fit_results.get('reduced_chisq', np.nan)
    new_row_2['RCHSQ'] = fit_results.get('reduced_chisq', np.nan)
    new_row_1['SNR'] = fit_results['param_dict'].get('FLUX', np.nan) / fit_results['error_dict'].get('FLUX', np.nan)
    new_row_2['SNR'] = fit_results['param_dict'].get('FLUX2', np.nan) / fit_results['error_dict'].get('FLUX2', np.nan)
    new_row_1['MU'] = mu
    new_row_2['MU'] = mu
    new_row_1['MU_ERR'] = mu_err
    new_row_2['MU_ERR'] = mu_err
    new_row_1['Z'] = fit_results['param_dict'].get('LPEAK', np.nan) / rest_wave - 1
    new_row_2['Z'] = fit_results['param_dict'].get('LPEAK', np.nan) / rest_wave - 1 # simultaneously fitted, so same z
    new_row_1['Z_ERR'] = (fit_results['error_dict'].get('LPEAK', np.nan) / rest_wave)
    new_row_2['Z_ERR'] = (fit_results['error_dict'].get('LPEAK', np.nan) / rest_wave) # simultaneously fitted, so same error

    # Insert parameters into row_1
    for key, value in fit_results['param_dict'].items():
        if key in line_results.colnames:
            # Add the value to the new row
            new_row_1[key] = value
            # Add the corresponding error
            err_key = f"{key}_ERR"
            new_row_1[err_key] = fit_results['error_dict'].get(key, np.nan)
    
    # Insert parameter into row 2
    if doublet[1] is not None:
        for key, value in fit_results['param_dict'].items():
            if key == 'FLUX2':
                base_key = 'FLUX'
                new_row_2[base_key] = value
                err_key = f"{base_key}_ERR"
                new_row_2[err_key] = fit_results['error_dict'].get(key, np.nan)
            elif key == 'LPEAK':
                new_row_2[key] = fit_results['param_dict'].get('LPEAK', np.nan) * (rest_wave_2 / rest_wave)
                err_key = f"{key}_ERR"
                new_row_2[err_key] = fit_results['error_dict'].get('LPEAK', np.nan) * (rest_wave_2 / rest_wave)
            else:
                if key in line_results.colnames:
                    new_row_2[key] = value
                    err_key = f"{key}_ERR"
                    new_row_2[err_key] = fit_results['error_dict'].get(key, np.nan)
    
    # Any missing columns in the table need to be filled with NaN as astropy.Table doesn't do this automatically
    for col in line_results.colnames:
        if col not in new_row_1:
            new_row_1[col] = np.nan
        if col not in new_row_2 and new_row_2['LINE'] is not None:
            new_row_2[col] = np.nan
    
    line_results.add_row(new_row_1)
    if new_row_2['LINE'] is not None:
        line_results.add_row(new_row_2)
    
def fit_spectrum(row, line_results, lya_results, CLUSTER_NAME, SPEC_SOURCE, SPEC_TYPE):
    """
    Refit a stacked spectrum using initial guesses from the corresponding rows of the provided catalogs.
    Modifies line_results and lya_results tables in place, inserting new fit results.

    Parameters
    ----------
    source_cat : astropy.table.Table
        Table containing source information.
    line_cat : astropy.table.Table
        Table containing line information.
    spec_paths : dict
        Dictionary mapping source identifiers to their spectrum file paths.
    line_results : astropy.table.Table
        Results table into which new fit results will be inserted.
    lya_results : astropy.table.Table
        Lyman alpha results table into which new fit results will be inserted.
    CLUSTER_NAME : str
        Name of the cluster (e.g., 'A2744').
    SPEC_SOURCE : str
        Source of the spectrum data (R21 or APER).

    Returns
    -------
    None
    """
    # Get source identifier and cluster
    iden = row['iden']
    full_iden = f"{row['idfrom'][0].replace('E', 'X')}{iden}"
    idfrom = row['idfrom']
    z = row['z']
    mu = row['MU']
    mu_err = row['MU_ERR']
    ra, dec = row['RA'], row['DEC']
    clus = CLUSTER_NAME

    # Get table of spectral lines for this source from the r21 line catalog
    line_table = aucat.get_line_table(full_iden, clus, exclude_lya=False)

    if len(line_table) == 0:
        print(f"\nNo lines to fit for source {full_iden}. Skipping...")
        return # Skip sources with no lines to fit

    print(f"\nFitting spectrum for {clus} {full_iden}...")

    # Load the spectrum
    spec_tab = auio.load_spec(clus, full_iden, idfrom, spec_source=SPEC_SOURCE, spec_type=SPEC_TYPE)
    
    # Get wavelength, spectrum, and error arrays
    wave = spec_tab['wave']
    spec = spec_tab['spec']
    error = spec_tab['spec_err']

    # Generate initial guesses
    lya_p0 = aufit.get_initial_guesses_from_catalog(line_table, 'LYALPHA', type='em')

    # Fit the Lyman alpha line first
    plot_dir = auio.get_plot_dir(clus, full_iden)
    lya_fit_results = lyafit.fit_lya_line(wave, spec, error, lya_p0, full_iden, clus,
                                           plot_result=True, width=50,
                                           save_plots=True, plot_dir=plot_dir,
                                           spec_type=SPEC_TYPE) # default bounds will be used here
    
    # If the Lyman alpha fit failed, raise a warning and skip this source
    if not lya_fit_results:
        print(f"WARNING: Lyman alpha fit failed for source {full_iden}. Skipping further line fitting.")
        return

    # Insert Lyman alpha fit results into the lya_results table
    insert_lya_results(lya_results, full_iden, lya_fit_results, mu, mu_err, ra, dec)

    # Now fit any other lines from the R21 catalogues
    other_fit_results = {}

    for i, line_row in enumerate(line_table):
        if line_row['LINE'] == 'LYALPHA':
            continue  # Already fitted Lyman alpha
        # If it's the secondary component of a doublet, check that the primary is in the table,
        # and if it is, then skip, otherwise, raise a warning and continue
        if np.any(auconst.slines == line_row['LINE']):
            idx = np.where(auconst.slines == line_row['LINE'])[0][0]
            primary = auconst.flines[idx]
            if np.any(line_table['LINE'] == primary):
                print(f"Skipping secondary line {line_row['LINE']} as primary {primary} is also present.")
                continue
            else:
                print(f"Warning: Secondary line {line_row['LINE']} present without primary {primary}. Proceeding to fit.")
        
        line_name = line_row['LINE']
        print(f"Fitting {line_name} for {clus} {full_iden}...")
        # Get initial guesses and fit the line
        initial_guesses = aufit.get_initial_guesses_from_catalog(line_table, line_name)
        # Prepare inputs for fitting -- making sure bounds are appropriate
        initial_guesses, bounds = aufit.prep_inputs(initial_guesses, line_name, z)
        
        line_fit = aufit.fit_line(wave, spec, error, line_name, initial_guesses, bounds, plot_result=True,
                                  save_plots=True, plot_dir=plot_dir, spec_type=SPEC_TYPE, cluster=clus, 
                                  full_iden=full_iden)
        other_fit_results[line_name] = line_fit

        # Insert line fit results into the line_results table
        if line_fit: # You'll get an empty dict if the fit failed
            insert_line_results(line_results, full_iden, line_name, line_fit, mu, mu_err)

    
    # Check to see whether any important lines have not been fitted due to not appearing in the catalog
    important_lines = ['CIV1548', 'SiII1260', 'OIII1660', 'SiIV1394', 'CIII1907', 'HeII1640']
    important_lines = set(important_lines) - set(other_fit_results.keys()) # Only check those not already fitted

    # Force fitting of important lines if they fall within the wavelength range of the spectrum
    for line in important_lines:
        rest_wavelength = auconst.wavedict[line]
        observed_wavelength = rest_wavelength * (1 + z)
        if line not in other_fit_results and observed_wavelength >= wave.min() and observed_wavelength <= wave.max():
            print(f"Important line {line} not fitted for source {full_iden}. Refitting with manual initial guesses.")
            # Define some rough initial guesses for these lines based on the source redshift
            manual_p0 = {
                'LPEAK': observed_wavelength,
                'FLUX': lya_fit_results['param_dict']['AMPR'] * 0.1 * 1.25,  # Assume 10% of Lyman alpha flux
                'FWHM': 3.0,  # Assume a FWHM of 3 Angstroms
                'CONT': np.nanmedian(spec),  # Median of the spectrum as continuum
                'SLOPE': 0.0  # Flat continuum
            }
            if line in auconst.flines: # If it's a doublet, add a second component
                manual_p0['FLUX2'] = manual_p0['FLUX']  # Assume secondary component is half the flux of primary
            manual_fit = aufit.fit_line(wave, spec, error, line, manual_p0, plot_result=True,
                                        save_plots=True, plot_dir=plot_dir, spec_type=SPEC_TYPE, cluster=clus, 
                                        full_iden=full_iden)
            if manual_fit:
                insert_line_results(line_results, full_iden, line, manual_fit, mu, mu_err)
        

def main():

    parser = argparse.ArgumentParser(description="Script to analyse MUSE LAE spectra.")
    parser.add_argument("CLUSTER_NAME", type=str, help="Name of the cluster (e.g., 'A2744')")
    parser.add_argument("SPEC_SOURCE", type=str, help="Source of the spectrum data (R21 or APER)")
    parser.add_argument("SPEC_TYPE", type=str, help="Type of spectrum to use (e.g., 'noweight_skysub', '2fwhm')")
    parser.add_argument("--aper-size", type=int, default=None, help="Aperture size in FWHM (required for APER)")
    parser.add_argument("--overwrite-spectra", action="store_true", help="Overwrite existing spectra")
    parser.add_argument("--optimise-apertures", type=str, default='auto', 
                        help="Which aperture optimisation to use ('auto', 'none', 'lya')")
    parser.add_argument("--optimise-apertures-kwargs", type=str, default='{"nstruct":2, "niter":1}', 
                        help="Additional keyword arguments for aperture optimisation methods (nstruct, niter)")
    parser.add_argument("--plot-images", action="store_true", help="Whether to plot aperture placement diagnostic images")
    args = parser.parse_args()

    CLUSTER_NAME = args.CLUSTER_NAME
    SPEC_SOURCE = args.SPEC_SOURCE
    APER_SIZE = args.aper_size
    OVERWRITE = args.overwrite_spectra
    OPTIMISE_APERTURES = args.optimise_apertures if args.optimise_apertures.lower() != 'none' else None
    OPTIMISE_APERTURES_KWARGS = ast.literal_eval(args.optimise_apertures_kwargs)
    SPEC_TYPE = args.SPEC_TYPE + ('_opt' if OPTIMISE_APERTURES else '')
    PLOT_IMAGES = args.plot_images

    # Load source catalog for the specified cluster
    source_cat = auio.load_r21_catalogue(CLUSTER_NAME, type='source')
    # Load lines catalog for the specified cluster
    line_cat = auio.load_r21_catalogue(CLUSTER_NAME, type='line')

    # Filter to look only at LAEs
    source_cat = source_cat[(source_cat['z'] > 2.9) & (source_cat['z'] < 6.7)] # LAEs only for MUSE wavelength coverage
    line_cat   = line_cat[(line_cat['Z'] > 2.9) & (line_cat['Z'] < 6.7)]     # LAEs only for MUSE wavelength coverage

    # Add a full identifier column to the source catalog for easier matching with spectra
    full_idens = [f"{row['idfrom'][0].replace('E', 'X')}{row['iden']}" for row in source_cat]
    source_cat['full_iden'] = full_idens

    # Extract and save spectra for all sources in the catalog if needed
    if SPEC_SOURCE == 'APER':
        if APER_SIZE is None:
            raise ValueError("Aperture size must be specified for APER spectra.")
        spec_paths, positions = auifs.extract_spectra(source_cat, APER_SIZE, CLUSTER_NAME, overwrite=OVERWRITE,
                                     optimise_apertures=OPTIMISE_APERTURES, 
                                     optimise_apertures_kwargs=OPTIMISE_APERTURES_KWARGS,
                                     plot_images=PLOT_IMAGES,
                                     save_plots=True)
    elif SPEC_SOURCE == 'R21':
        spec_paths = get_R21_paths(source_cat, CLUSTER_NAME, SPEC_TYPE)
        positions  = {full_iden: (row['RA'], row['DEC']) for row, full_iden in zip(source_cat, full_idens)}
    else:
        raise ValueError("SPEC_SOURCE must be either 'APER' or 'R21'.")
    
    # Update source catalogue positions
    source_cat['RA'] = np.array([positions[full_iden][0] for full_iden in source_cat['full_iden']])
    source_cat['DEC'] = np.array([positions[full_iden][1] for full_iden in source_cat['full_iden']])
        
    # Check to make sure the spectra are available
    check_spectra_availability(spec_paths)

    # Initialize the results tables
    line_results = aptb.Table(
        names=('iden', 'LINE', 'LBDA_REST', 'FLUX', 'FLUX_ERR',
               'LPEAK', 'LPEAK_ERR', 'FWHM', 'FWHM_ERR', 'SNR',
               'CONT', 'CONT_ERR', 'SLOPE', 'SLOPE_ERR', 'MU',
               'MU_ERR', 'Z', 'Z_ERR', 'RCHSQ', 'FLAG'),
         dtype=('U', 'U', 'f8', 'f8', 'f8',
             'f8', 'f8', 'f8', 'f8', 'f8',
             'f8', 'f8', 'f8', 'f8', 'f8',
             'f8', 'f8', 'f8', 'f8', 'U')
    )
    lya_results = aptb.Table(
        names=('iden', 'LINE', 'LBDA_REST', 
               'RA', 'DEC',
               'FLUXB', 'FLUXB_ERR', 
               'FLUXR', 'FLUXR_ERR', 
               'AMPR', 'AMPR_ERR', 
               'AMPB', 'AMPB_ERR', 
               'LPEAKR', 'LPEAKR_ERR', 
               'LPEAKB', 'LPEAKB_ERR',
               'FWHMR', 'FWHMR_ERR', 
               'FWHMB', 'FWHMB_ERR', 
               'DISPR', 'DISPR_ERR', 
               'DISPB', 'DISPB_ERR', 
               'SNRR', 'SNRB', 
               'ASYMR', 'ASYMR_ERR', 
               'ASYMB', 'ASYMB_ERR', 
               'CONT', 'CONT_ERR', 
               'SLOPE', 'SLOPE_ERR', 
               'TAU', 'TAU_ERR', 
               'FWHM_ABS', 'FWHM_ABS_ERR', 
               'LPEAK_ABS', 'LPEAK_ABS_ERR',
               'MU', 'MU_ERR',
               'Z', 'Z_ERR', 
               'FLUXB_UB', 'RCHSQ', 'FLAG'),
         dtype=(
             'U', 'U', 'f8',
             'f8', 'f8',
             'f8', 'f8',
             'f8', 'f8', 
             'f8', 'f8',
             'f8', 'f8',
             'f8', 'f8',
             'f8', 'f8',
             'f8', 'f8',
             'f8', 'f8', 
             'f8', 'f8', 
             'f8', 'f8', 
             'f8', 'f8', 
             'f8', 'f8',
             'f8', 'f8', 
             'f8', 'f8', 
             'f8', 'f8', 
             'f8', 'f8', 
             'f8', 'f8',
             'f8', 'f8',
             'f8', 'f8',
             'f8', 'f8',
             'f8'
            )
    )

    # Fit lines for each source and populate the results tables
    for row in source_cat:
        # Check to make sure it's in spec_paths
        full_iden = f"{row['idfrom'][0].replace('E', 'X')}{row['iden']}"
        if full_iden not in spec_paths:
            print(f"WARNING! Spectrum for source {full_iden} not found in spec_paths. Skipping...")
            continue
        print(f"Processing source {full_iden}...")
        fit_spectrum(row, line_results, lya_results, CLUSTER_NAME, SPEC_SOURCE, SPEC_TYPE)

    # Force any zero entries in results tables to NaN for consistency
    for col in line_results.colnames:
        if line_results[col].dtype.kind in 'fi':  # Only for float or integer columns
            line_results[col] = np.where(line_results[col] == 0, np.nan, line_results[col])
    for col in lya_results.colnames:
        if lya_results[col].dtype.kind in 'fi':  # Only for float or integer columns
            lya_results[col] = np.where(lya_results[col] == 0, np.nan, lya_results[col])

    # Save the results tables to disk
    results_dir = auio.get_fit_catalog_dir(CLUSTER_NAME)
    line_results_file = Path(results_dir) / f"{CLUSTER_NAME}_{SPEC_TYPE}_lines.fits"
    lya_results_file = Path(results_dir) / f"{CLUSTER_NAME}_{SPEC_TYPE}_lya.fits"

    line_results.write(line_results_file, overwrite=True)
    lya_results.write(lya_results_file, overwrite=True)

if __name__ == "__main__":
    main()
