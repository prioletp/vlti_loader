"""
Utility functions for handling OIFITS data from different instruments.
"""

import numpy as np
from astropy.io import fits


def read_header(fits_file_path, hdu_index=0, keywords=None):
    """
    Efficiently read FITS file header with optional keyword filtering.
    
    Parameters:
    -----------
    fits_file_path : str
        Path to the FITS file
    hdu_index : int, optional
        HDU index to read (default: 0 for primary header)
    keywords : list of str, optional
        Specific keywords to extract. If None, returns all header cards.
        
    Returns:
    --------
    dict or astropy.io.fits.Header
        Header information as dictionary (if keywords specified) or full Header object
    """
    try:
        from astropy.io import fits
        import os
        
        # Check if file exists
        if not os.path.exists(fits_file_path):
            raise FileNotFoundError(f"FITS file not found: {fits_file_path}")
        
        # Efficiently read header without loading data
        with fits.open(fits_file_path, mode='readonly', memmap=True) as hdul:
            # Validate HDU index
            if hdu_index >= len(hdul):
                raise IndexError(f"HDU index {hdu_index} out of range. File has {len(hdul)} HDUs.")
            
            header = hdul[0].header
            
            # Return specific keywords if requested
            if keywords is not None:
                if isinstance(keywords, str):
                    keywords = [keywords]
                
                result = {}
                for keyword in keywords:
                    try:
                        result[keyword] = header[keyword]
                    except KeyError:
                        result[keyword] = None  # Missing keyword
                
                return result
            
            # Return complete header as dictionary for better serialization
            return dict(header)
            
    except ImportError:
        raise ImportError("astropy is required for FITS file handling. Install with: pip install astropy")
    except Exception as e:
        raise RuntimeError(f"Error reading FITS header from {fits_file_path}: {str(e)}")

def create_data_dic_PIONIER(hdul):
        """
        Creates a dictionary containing the oifits data with optimized processing.
        
        Parameters
        ----------
        hdul : HDUList
            HDU list following the oifits format
            
        Returns
        -------
        data_dic : dict
            Dictionary containing the interferometric data with keys:
            - 'VIS2', 'VIS2_err': Squared visibility and errors
            - 'VIS2_waves': Wavelengths for visibility data
            - 'Bu', 'Bv': UV coordinates
            - 'B', 'freqs': Baseline lengths and spatial frequencies
            - 'Vis2_sta_idx_0/1', 'Vis2_tel_name_0/1': Station info
            - 'TEL_type', 'Telescopes': Telescope information
            - T3 data (if available): 'T3_PHI', 'T3_PHI_err', UV coords, etc.
        """
        import numpy as np
        
        # Get HDU content efficiently
        hdul_content = [hdu.name for hdu in hdul]
        data_dic = {}
        # Extract basic data
        header = hdul[0].header
        wavelengths = hdul['OI_WAVELENGTH'].data['EFF_WAVE']
        n_waves = len(wavelengths)
        
        # === VISIBILITY DATA PROCESSING ===
        vis2_data = hdul['OI_VIS2'].data
        V2 = vis2_data['VIS2DATA']
        V2_err = vis2_data['VIS2ERR'] 
        B_u = vis2_data['UCOORD']
        B_v = vis2_data['VCOORD']
        V2_sta_idx = vis2_data['STA_INDEX']
        n_baselines = len(B_u)


        # Vectorized data expansion (much faster than loops)
        data_dic['VIS2'] = V2.ravel()
        data_dic['VIS2_err'] = V2_err.ravel()
        data_dic['VIS2_waves'] = np.tile(wavelengths, n_baselines)
        data_dic['INS_VIS2'] = np.repeat('PIONIER', data_dic['VIS2'].shape[0])

        # Expand baseline coordinates
        data_dic['Bu'] = np.repeat(B_u, n_waves)
        data_dic['Bv'] = np.repeat(B_v, n_waves)
        data_dic['B'] = np.sqrt(data_dic['Bu']**2 + data_dic['Bv']**2)
        data_dic['freqs'] = data_dic['B'] / data_dic['VIS2_waves']
        data_dic['u'] = data_dic['Bu'] / data_dic['VIS2_waves']
        data_dic['v'] = data_dic['Bv'] / data_dic['VIS2_waves']
        
        # Station indices
        data_dic['VIS2_sta_idx_0'] = np.repeat(V2_sta_idx[:, 0], n_waves)
        data_dic['VIS2_sta_idx_1'] = np.repeat(V2_sta_idx[:, 1], n_waves)
        
        # Telescope name mapping
        tel_names = hdul['OI_ARRAY'].data['TEL_NAME']
        station_indices = hdul['OI_ARRAY'].data['STA_INDEX']
        station_to_name = dict(zip(station_indices, tel_names))
        
        # Vectorized telescope name assignment
        data_dic['VIS2_tel_name_0'] = np.array([station_to_name[idx] for idx in data_dic['VIS2_sta_idx_0']])
        data_dic['VIS2_tel_name_1'] = np.array([station_to_name[idx] for idx in data_dic['VIS2_sta_idx_1']])
        
        # Telescope information
        data_dic['TEL_type'] = 'UT' if 'U' in header.get('TELESCOP', '') else 'AT'
        data_dic['Telescopes'] = np.unique(V2_sta_idx)
        
        # === CLOSURE PHASE DATA PROCESSING ===
        if 'OI_T3' in hdul_content:
            print('Loading closure phase data...')
            t3_data = hdul['OI_T3'].data
            T3_phi = t3_data['T3PHI']
            T3_err = t3_data['T3PHIERR']

            # UV coordinates
            U1, V1 = t3_data['U1COORD'], t3_data['V1COORD']
            U2, V2 = t3_data['U2COORD'], t3_data['V2COORD']
            U3, V3 = -(U1 + U2), -(V1 + V2)
            
            n_triangles = len(U1)
            
            # Vectorized T3 data processing
            data_dic['T3_PHI'] = T3_phi.ravel()
            data_dic['T3_PHI_err'] = T3_err.ravel()
            data_dic['T3_waves'] = np.tile(wavelengths, n_triangles)
            data_dic['INS_T3'] = np.repeat('PIONIER', data_dic['T3_PHI'].shape[0])

            # UV coordinates for T3
            data_dic['U1'] = np.repeat(U1, n_waves)
            data_dic['V1'] = np.repeat(V1, n_waves)
            data_dic['U2'] = np.repeat(U2, n_waves)
            data_dic['V2'] = np.repeat(V2, n_waves)
            data_dic['U3'] = np.repeat(U3, n_waves)
            data_dic['V3'] = np.repeat(V3, n_waves)
            
            # Baseline calculations for triangles
            baselines = np.array([
                np.sqrt(U1**2 + V1**2),
                np.sqrt(U2**2 + V2**2), 
                np.sqrt(U3**2 + V3**2)
            ]).T  # Shape: (n_triangles, 3)
            
            data_dic['avg_base'] = np.repeat(np.mean(baselines, axis=1), n_waves)
            data_dic['max_base'] = np.repeat(np.max(baselines, axis=1), n_waves)

            # T3 station indices and telescope names
            T3_sta_idx = t3_data['STA_INDEX']  # shape (n_triangles, 3)
            data_dic['T3_sta_idx_0'] = np.repeat(T3_sta_idx[:, 0], n_waves)
            data_dic['T3_sta_idx_1'] = np.repeat(T3_sta_idx[:, 1], n_waves)
            data_dic['T3_sta_idx_2'] = np.repeat(T3_sta_idx[:, 2], n_waves)
            data_dic['T3_tel_name_0'] = np.array([station_to_name[idx] for idx in data_dic['T3_sta_idx_0']])
            data_dic['T3_tel_name_1'] = np.array([station_to_name[idx] for idx in data_dic['T3_sta_idx_1']])
            data_dic['T3_tel_name_2'] = np.array([station_to_name[idx] for idx in data_dic['T3_sta_idx_2']])
        
        return data_dic
    
def create_data_dic_MATISSE(hdul, N_band=False, flux_bool=False):
        """
        Creates a dictionary containing the oifits data with optimized processing.
        
        Parameters
        ----------
        hdul : HDUList
            HDU list following the oifits format
        flux_bool : bool, optional
            Whether to extract flux data (default: False)
            
        Returns
        -------
        data_dic : dict
            Dictionary containing the interferometric data with keys:
            - 'VIS2', 'VIS2_err': Squared visibility and errors
            - 'VIS2_waves': Wavelengths for visibility data
            - 'Bu', 'Bv': UV coordinates
            - 'B', 'freqs': Baseline lengths and spatial frequencies
            - 'Vis2_sta_idx_0/1', 'Vis2_tel_name_0/1': Station info
            - 'TEL_type', 'Telescopes': Telescope information
            - T3 data (if available): 'T3_PHI', 'T3_PHI_err', UV coords, etc.
            - Flux data (if requested and available)
        """
        import numpy as np
        
        # Get HDU content efficiently
        hdul_content = [hdu.name for hdu in hdul]
        data_dic = {}
        # Extract basic data
        header = hdul[0].header
        wavelengths = hdul['OI_WAVELENGTH'].data['EFF_WAVE']
        n_waves = len(wavelengths)
        
        # === VISIBILITY DATA PROCESSING ===
        vis2_data = hdul['OI_VIS2'].data
        V2 = vis2_data['VIS2DATA']
        V2_err = vis2_data['VIS2ERR'] 
        B_u = vis2_data['UCOORD']
        B_v = vis2_data['VCOORD']
        V2_sta_idx = vis2_data['STA_INDEX']
        n_baselines = len(B_u)
        if N_band:
            print('Loading MATISSE N-band data (AQUARIUS detector)')
            corrflux_data = hdul['OI_VIS'].data
            corrflux = corrflux_data['VISAMP']
            corrflux_err = corrflux_data['VISAMPERR']
            data_dic['CorrFlux'] = corrflux.ravel()
            data_dic['CorrFlux_err'] = corrflux_err.ravel()

        # Vectorized data expansion (much faster than loops)
        data_dic['VIS2'] = V2.ravel()
        data_dic['VIS2_err'] = V2_err.ravel()
        data_dic['VIS2_waves'] = np.tile(wavelengths, n_baselines)
        data_dic['INS_VIS2'] = np.repeat('MATISSE', data_dic['VIS2'].shape[0])

        # Expand baseline coordinates
        data_dic['Bu'] = np.repeat(B_u, n_waves)
        data_dic['Bv'] = np.repeat(B_v, n_waves)
        data_dic['B'] = np.sqrt(data_dic['Bu']**2 + data_dic['Bv']**2)
        data_dic['freqs'] = data_dic['B'] / data_dic['VIS2_waves']
        data_dic['u'] = data_dic['Bu'] / data_dic['VIS2_waves']
        data_dic['v'] = data_dic['Bv'] / data_dic['VIS2_waves']
        
        # Station indices
        data_dic['VIS2_sta_idx_0'] = np.repeat(V2_sta_idx[:, 0], n_waves)
        data_dic['VIS2_sta_idx_1'] = np.repeat(V2_sta_idx[:, 1], n_waves)
        
        # Telescope name mapping
        tel_names = hdul['OI_ARRAY'].data['TEL_NAME']
        station_indices = hdul['OI_ARRAY'].data['STA_INDEX']
        station_to_name = dict(zip(station_indices, tel_names))
        
        # Vectorized telescope name assignment
        data_dic['VIS2_tel_name_0'] = np.array([station_to_name[idx] for idx in data_dic['VIS2_sta_idx_0']])
        data_dic['VIS2_tel_name_1'] = np.array([station_to_name[idx] for idx in data_dic['VIS2_sta_idx_1']])
        
        # Telescope information
        data_dic['TEL_type'] = 'UT' if 'U' in header.get('TELESCOP', '') else 'AT'
        data_dic['Telescopes'] = np.unique(V2_sta_idx)
        
        # === CLOSURE PHASE DATA PROCESSING ===
        if 'OI_T3' in hdul_content:
            print('Loading closure phase data...')
            t3_data = hdul['OI_T3'].data
            T3_phi = t3_data['T3PHI']
            T3_err = t3_data['T3PHIERR']

            # UV coordinates
            U1, V1 = t3_data['U1COORD'], t3_data['V1COORD']
            U2, V2 = t3_data['U2COORD'], t3_data['V2COORD']
            U3, V3 = -(U1 + U2), -(V1 + V2)
            
            n_triangles = len(U1)
            
            # Vectorized T3 data processing
            data_dic['T3_PHI'] = T3_phi.ravel()
            data_dic['T3_PHI_err'] = T3_err.ravel()
            data_dic['T3_waves'] = np.tile(wavelengths, n_triangles)
            data_dic['INS_T3'] = np.repeat('MATISSE', data_dic['T3_PHI'].shape[0])

            # UV coordinates for T3
            data_dic['U1'] = np.repeat(U1, n_waves)
            data_dic['V1'] = np.repeat(V1, n_waves)
            data_dic['U2'] = np.repeat(U2, n_waves)
            data_dic['V2'] = np.repeat(V2, n_waves)
            data_dic['U3'] = np.repeat(U3, n_waves)
            data_dic['V3'] = np.repeat(V3, n_waves)
            
            # Baseline calculations for triangles
            baselines = np.array([
                np.sqrt(U1**2 + V1**2),
                np.sqrt(U2**2 + V2**2), 
                np.sqrt(U3**2 + V3**2)
            ]).T  # Shape: (n_triangles, 3)
            
            data_dic['avg_base'] = np.repeat(np.mean(baselines, axis=1), n_waves)
            data_dic['max_base'] = np.repeat(np.max(baselines, axis=1), n_waves)

            # T3 station indices and telescope names
            T3_sta_idx = t3_data['STA_INDEX']  # shape (n_triangles, 3)
            data_dic['T3_sta_idx_0'] = np.repeat(T3_sta_idx[:, 0], n_waves)
            data_dic['T3_sta_idx_1'] = np.repeat(T3_sta_idx[:, 1], n_waves)
            data_dic['T3_sta_idx_2'] = np.repeat(T3_sta_idx[:, 2], n_waves)
            data_dic['T3_tel_name_0'] = np.array([station_to_name[idx] for idx in data_dic['T3_sta_idx_0']])
            data_dic['T3_tel_name_1'] = np.array([station_to_name[idx] for idx in data_dic['T3_sta_idx_1']])
            data_dic['T3_tel_name_2'] = np.array([station_to_name[idx] for idx in data_dic['T3_sta_idx_2']])
        
        # === FLUX DATA PROCESSING ===
        if flux_bool and 'OI_FLUX' in hdul_content:
            flux_data = hdul['OI_FLUX'].data
            data_dic['FLUX'] = flux_data['FLUXDATA']
            data_dic['FLUX_err'] = flux_data['FLUXERR']
            data_dic['FLUX_sta_idx'] = flux_data['STA_INDEX']
        
        return data_dic


def create_data_dic_GRAVITY(hdul, polarization='combined', fringe_tracker=False, flux_bool=False):
    """
    Creates a dictionary containing GRAVITY OIFITS data with optimized processing.
    
    Parameters
    ----------
    hdul : HDUList
        HDU list following the OIFITS format from GRAVITY
    polarization : str, optional
        Polarization mode: 'P1', 'P2', or 'combined' (default: 'combined')
    fringe_tracker : bool, optional
        Whether to use fringe tracker (FT) or not (default: True)
    flux_bool : bool, optional
        Whether to extract flux data (default: False)
        
    Returns
    -------
    data_dic : dict
        Dictionary containing the interferometric data with keys:
        - 'VIS2', 'VIS2_err': Squared visibility and errors
        - 'VIS2_waves': Wavelengths for visibility data
        - 'Bu', 'Bv': UV coordinates
        - 'B', 'freqs': Baseline lengths and spatial frequencies
        - 'FLUX' (if requested and available)
    """
    
    # Define HDU indices based on instrument configuration
    if fringe_tracker:
        hdu_indices = {
            'P1': {'wavelength': 5, 'vis2': 8, 't3': 9},
            'P2': {'wavelength': 6, 'vis2': 12, 't3': 13},
            'combined': {'wavelength': 4, 'vis2': 6, 't3': 7}
        }
        # t3_index = 7
    else:
        hdu_indices = {
            'P1': {'wavelength': 3, 'vis2': 16, 't3': 17},
            'P2': {'wavelength': 4, 'vis2': 20, 't3': 21},
            'combined': {'wavelength': 3, 'vis2': 10, 't3': 11}
        }
        # t3_index = 11
    hdul_content = [hdu.name for hdu in hdul]

    # Select appropriate indices
    if polarization == 'combined':
        wave_idx = hdu_indices['combined']['wavelength']
        vis2_idx = hdu_indices['combined']['vis2']
        t3_index = hdu_indices['combined']['t3']
    else:
        wave_idx = hdu_indices[polarization]['wavelength']
        vis2_idx = hdu_indices[polarization]['vis2']
        t3_index = hdu_indices[polarization]['t3']
    
    # Extract basic data
    wavelengths = hdul[wave_idx].data['EFF_WAVE']
    n_waves = len(wavelengths)
    header = hdul[0].header

    # Extract visibility data
    vis2_data = hdul[vis2_idx].data
    V2 = vis2_data['VIS2DATA']
    V2_err = vis2_data['VIS2ERR']
    B_u = vis2_data['UCOORD']
    B_v = vis2_data['VCOORD']
    n_baselines = len(B_u)
    V2_sta_idx = vis2_data['STA_INDEX']

    # Vectorized data expansion (optimized approach)
    data_dic = {}
    data_dic['VIS2'] = V2.ravel()
    data_dic['VIS2_err'] = np.abs(V2_err.ravel())
    data_dic['VIS2_waves'] = np.tile(wavelengths, n_baselines)
    data_dic['INS_VIS2'] = np.repeat('GRAVITY', data_dic['VIS2'].shape[0])

    # Expand baseline coordinates efficiently
    data_dic['Bu'] = np.repeat(B_u, n_waves)
    data_dic['Bv'] = np.repeat(B_v, n_waves)
    data_dic['B'] = np.sqrt(data_dic['Bu']**2 + data_dic['Bv']**2)
    data_dic['freqs'] = data_dic['B'] / data_dic['VIS2_waves']
    data_dic['u'] = data_dic['Bu'] / data_dic['VIS2_waves']
    data_dic['v'] = data_dic['Bv'] / data_dic['VIS2_waves']
    
        # Station indices
    data_dic['VIS2_sta_idx_0'] = np.repeat(V2_sta_idx[:, 0], n_waves)
    data_dic['VIS2_sta_idx_1'] = np.repeat(V2_sta_idx[:, 1], n_waves)
    
    # Telescope name mapping
    tel_names = hdul['OI_ARRAY'].data['TEL_NAME']
    station_indices = hdul['OI_ARRAY'].data['STA_INDEX']
    station_to_name = dict(zip(station_indices, tel_names))
    
    # Vectorized telescope name assignment
    data_dic['VIS2_tel_name_0'] = np.array([station_to_name[idx] for idx in data_dic['VIS2_sta_idx_0']])
    data_dic['VIS2_tel_name_1'] = np.array([station_to_name[idx] for idx in data_dic['VIS2_sta_idx_1']])
    
    # Telescope information
    data_dic['TEL_type'] = 'UT' if 'U' in header.get('TELESCOP', '') else 'AT'
    data_dic['Telescopes'] = np.unique(V2_sta_idx)
        
    if 'OI_T3' in hdul_content:
        print('Loading closure phase data...')
        t3_data = hdul[t3_index].data
        T3_phi = t3_data['T3PHI']
        T3_err = t3_data['T3PHIERR']
        
        # UV coordinates
        U1, V1 = t3_data['U1COORD'], t3_data['V1COORD']
        U2, V2 = t3_data['U2COORD'], t3_data['V2COORD']
        U3, V3 = -(U1 + U2), -(V1 + V2)
        
        n_triangles = len(U1)
        
        # Vectorized T3 data processing
        data_dic['T3_PHI'] = T3_phi.ravel()
        data_dic['T3_PHI_err'] = T3_err.ravel()
        data_dic['T3_waves'] = np.tile(wavelengths, n_triangles)
        
        # UV coordinates for T3
        data_dic['U1'] = np.repeat(U1, n_waves)
        data_dic['V1'] = np.repeat(V1, n_waves)
        data_dic['U2'] = np.repeat(U2, n_waves)
        data_dic['V2'] = np.repeat(V2, n_waves)
        data_dic['U3'] = np.repeat(U3, n_waves)
        data_dic['V3'] = np.repeat(V3, n_waves)
        data_dic['INS_T3'] = np.repeat('GRAVITY', data_dic['T3_PHI'].shape[0])
        
        # Baseline calculations for triangles
        baselines = np.array([
            np.sqrt(U1**2 + V1**2),
            np.sqrt(U2**2 + V2**2), 
            np.sqrt(U3**2 + V3**2)
        ]).T  # Shape: (n_triangles, 3)
        
        data_dic['avg_base'] = np.repeat(np.mean(baselines, axis=1), n_waves)
        data_dic['max_base'] = np.repeat(np.max(baselines, axis=1), n_waves)

        # T3 station indices and telescope names
        T3_sta_idx = t3_data['STA_INDEX']  # shape (n_triangles, 3)
        data_dic['T3_sta_idx_0'] = np.repeat(T3_sta_idx[:, 0], n_waves)
        data_dic['T3_sta_idx_1'] = np.repeat(T3_sta_idx[:, 1], n_waves)
        data_dic['T3_sta_idx_2'] = np.repeat(T3_sta_idx[:, 2], n_waves)
        data_dic['T3_tel_name_0'] = np.array([station_to_name[idx] for idx in data_dic['T3_sta_idx_0']])
        data_dic['T3_tel_name_1'] = np.array([station_to_name[idx] for idx in data_dic['T3_sta_idx_1']])
        data_dic['T3_tel_name_2'] = np.array([station_to_name[idx] for idx in data_dic['T3_sta_idx_2']])
    # Extract flux data if requested
    if flux_bool:
        try:
            if 'OI_FLUX' in [hdu.name for hdu in hdul]:
                flux_data = hdul['OI_FLUX'].data
                data_dic['FLUX'] = flux_data['FLUXDATA']
                data_dic['FLUX_err'] = flux_data.get('FLUXERR', None)
                data_dic['FLUX_sta_idx'] = flux_data.get('STA_INDEX', None)
        except (KeyError, IndexError):
            print("Warning: Flux data requested but not found or accessible")
    
    return data_dic

def create_data_dic_GRAVITY_split(hdul, fringe_tracker=False, flux_bool=False):
    """
    Creates separate dictionaries for P1 and P2 polarizations from GRAVITY data.
    
    Parameters
    ----------
    hdul : HDUList
        HDU list following the OIFITS format from GRAVITY
    fringe_tracker : bool, optional
        Whether to use fringe tracker (FT) or not (default: True)
    flux_bool : bool, optional
        Whether to extract flux data (default: False)
        
    Returns
    -------
    tuple
        (data_dic_P1, data_dic_P2, data_dic_combined) containing the data for each polarization
    """
    
    # Extract data for each polarization
    data_dic_P1 = create_data_dic_GRAVITY(hdul, polarization='P1', 
                                         fringe_tracker=fringe_tracker, flux_bool=flux_bool)
    data_dic_P2 = create_data_dic_GRAVITY(hdul, polarization='P2', 
                                         fringe_tracker=fringe_tracker, flux_bool=flux_bool)
    
    # Merge P1 and P2 data
    data_dic_combined = merge_polarization_data([data_dic_P1, data_dic_P2])
    
    return data_dic_P1, data_dic_P2, data_dic_combined

def merge_polarization_data(data_dics):
    """
    Merge multiple polarization data dictionaries.
    
    Parameters
    ----------
    data_dics : list
        List of data dictionaries to merge
        
    Returns
    -------
    dict
        Merged data dictionary
    """
    if not data_dics:
        return {}
    
    merged_dic = {}
    
    # Get common keys (excluding metadata)
    data_keys = set(data_dics[0].keys())
    
    for key in data_keys:
        merged_data = []
        for dic in data_dics:
            if key in dic:
                if isinstance(dic[key], np.ndarray):
                    merged_data.extend(dic[key])
                else:
                    merged_data.append(dic[key])
        merged_dic[key] = np.array(merged_data)
    
  
    
    return merged_dic



def bin_gravity_data(data_dic, n_channels):
    """
    Bin GRAVITY data by wavelength for each baseline.
    
    Parameters
    ----------
    data_dic : dict
        Data dictionary to bin
    n_channels : int
        Number of spectral channels per bin
        
    Returns
    -------
    dict
        Binned data dictionary
    """
    # Get unique baselines
    unique_baselines = np.unique(data_dic['B'])
    
    binned_data = {
        'VIS2': [], 'VIS2_err': [], 'VIS2_waves': [],
        'Bu': [], 'Bv': [], 'B': [], 'freqs': []
    }
    
    for baseline in unique_baselines:
        # Find data for this baseline
        baseline_idx = np.where(np.abs(data_dic['B'] - baseline) < 1e-6)[0]
        
        # Sort by wavelength
        wave_sort_idx = np.argsort(data_dic['VIS2_waves'][baseline_idx])
        baseline_idx = baseline_idx[wave_sort_idx]
        
        # Calculate number of bins
        n_bins = len(baseline_idx) // n_channels
        
        # Bin the data
        for i in range(n_bins):
            bin_idx = baseline_idx[i*n_channels:(i+1)*n_channels]
            
            # Calculate binned values
            binned_data['VIS2'].append(np.mean(data_dic['VIS2'][bin_idx]))
            binned_data['VIS2_err'].append(
                np.sqrt(np.sum(data_dic['VIS2_err'][bin_idx]**2)) / n_channels
            )
            binned_data['VIS2_waves'].append(np.mean(data_dic['VIS2_waves'][bin_idx]))
            binned_data['Bu'].append(np.mean(data_dic['Bu'][bin_idx]))
            binned_data['Bv'].append(np.mean(data_dic['Bv'][bin_idx]))
            binned_data['B'].append(baseline)
            binned_data['freqs'].append(np.mean(data_dic['freqs'][bin_idx]))
    
    # Convert to numpy arrays
    for key in binned_data:
        binned_data[key] = np.array(binned_data[key])
    
    # Copy metadata
    for key in ['polarization', 'fringe_tracker']:
        if key in data_dic:
            binned_data[key] = data_dic[key]
    
    return binned_data

def merge_dics(dics_arr):
    
    """Merges the elements of multiple dictionaries.
        
    Parameters
    ----------
    dics_arr : list
        List containing multiple dictionaries generated by create_data_dic.
    
    Returns
    -------
    merged_dic: dic
    
        Dictionary containing the merged data of all dictionaries in dics_arr.
    """
    merged_dic = {}

    # Collect all unique keys from all dictionaries
    all_keys = set()
    for dic in dics_arr:
        all_keys.update(dic.keys())

    for key in all_keys:
        merged_dic[key] = []
        for dic in dics_arr:
            if key in dic:
                merged_dic[key].extend(dic[key])
        merged_dic[key] = np.array(merged_dic[key])
    return merged_dic
    
if __name__ == "__main__":
    # Example usage
    print("GRAVITY OIFITS utilities loaded.")
    print("Available functions:")
    print("  - create_data_dic_GRAVITY()")
    print("  - create_data_dic_GRAVITY_split()")
    print("  - merge_polarization_data()")
    print("  - filter_gravity_data()")
    print("  - bin_gravity_data()")