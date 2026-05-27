#%%
import os
import numpy as np
from pathlib import Path
from astropy.io import fits
import vlti_loader.oifits_utils as oi_utl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.gridspec import GridSpec

class Observations:
    def __init__(self, path_obs):

        
        
        self.path_obs = path_obs
        self.data_type = None
        self.file_list = []
        
        # Check if path_obs is an array/list
        if isinstance(path_obs, (list, tuple, np.ndarray)):
            self.data_type = 'array'
            self.file_list = list(path_obs)
            # Validate that all elements are valid file paths
            for file_path in self.file_list:
                if not os.path.exists(file_path):
                    raise FileNotFoundError(f"File not found in array: {file_path}")
                if not str(file_path).lower().endswith('.fits'):
                    raise ValueError(f"Non-FITS file in array: {file_path}")
        
        # Check if path_obs is a string (file or directory path)
        elif isinstance(path_obs, (str, Path)):
            path_obj = Path(path_obs)
            
            if not path_obj.exists():
                raise FileNotFoundError(f"Path does not exist: {path_obs}")
            
            # Check if it's a FITS file
            if path_obj.is_file() and str(path_obs).lower().endswith('.fits'):
                self.data_type = 'fits_file'
                self.file_list = [str(path_obs)]
            
            # Check if it's a directory
            elif path_obj.is_dir():
                self.data_type = 'directory'
                # Find all FITS files in the directory
                fits_files = list(path_obj.glob('*.fits')) + list(path_obj.glob('*.FITS'))
                if not fits_files:
                    raise ValueError(f"No FITS files found in directory: {path_obs}")
                self.file_list = [str(f) for f in sorted(fits_files)]
            
            else:
                raise ValueError(f"Path is not a FITS file or directory: {path_obs}")
        
        else:
            raise TypeError(f"Invalid type for path_obs: {type(path_obs)}. Expected string, Path, list, or array.")
        
        print(f"Initialized Observations with {self.data_type}: {len(self.file_list)} FITS file(s)")

        self.get_data()
        
    def __str__(self):
        return f"Observations({self.data_type}, {len(self.file_list)} files)"

    def __repr__(self):
        return f"Observations({self.data_type}, {len(self.file_list)} files)"

    def __getattr__(self, name):
        # Allows data dict keys to be accessed as attributes, e.g. obs.V2
        if name != 'data' and hasattr(self, 'data') and name in self.data:
            return self.data[name]
        raise AttributeError(f"'Observations' object has no attribute '{name}'")

    def get_data(self):
        data_dic_array = []
        for file_path in self.file_list:
            header = oi_utl.read_header(file_path)
            if 'ESO INS ID' in header:
                ins_name = header['ESO INS ID'].split('/')[0]
            elif 'INSTRUME' in header:
                ins_name = header['INSTRUME'].split('/')[0]
            else:
                print(f"Warning: Instrument name not found in header of {file_path}. Setting PIONIER as default.")
                ins_name = 'PIONIER'
                # raise KeyError(f"Instrument name not found in header of {file_path}")
            with fits.open(file_path, mode='readonly', memmap=True) as hdul:
                print(f'LOADING VLTI/{ins_name} data: {file_path}')
                if ins_name == 'GRAVITY':
                    polarization_mode = header['ESO INS POLA MODE'].lower()
                    if polarization_mode == 'combined':
                        data_dic_array.append(oi_utl.create_data_dic_GRAVITY(hdul, fringe_tracker=False, polarization=polarization_mode))
                    elif polarization_mode=='split':
                        data_dic_array.append(oi_utl.create_data_dic_GRAVITY_split(hdul, fringe_tracker=False, flux_bool=False)[2])
                elif ins_name == 'MATISSE':
                    N_band = header['ESO DET CHIP NAME'] == 'AQUARIUS'
                    if N_band:
                        print('Loading MATISSE N-band data (AQUARIUS detector)')
                    data_dic_array.append(oi_utl.create_data_dic_MATISSE(hdul, N_band=N_band))
                elif ins_name == 'PIONIER':
                    data_dic_array.append(oi_utl.create_data_dic_PIONIER(hdul))
        
        self.data_dic_array = data_dic_array
        self.data = oi_utl.merge_dics(data_dic_array)
        self._raw_data = {k: v.copy() if isinstance(v, np.ndarray) else v
                          for k, v in self.data.items()}
        return self.data
    
    def filter_data(self, wave_ranges=None, baseline_ranges=None, freq_ranges=None,
                   vis2_err_ranges=None, t3_err_ranges=None,
                   min_wave=None, max_wave=None, min_baseline=None, 
                   max_baseline=None, min_freq=None, max_freq=None,
                   min_vis2_err=None, max_vis2_err=None, min_t3_err=None, max_t3_err=None):
        """
        Filter data by various criteria, supporting both single ranges and multiple ranges.
        Filters are applied to both V2 and T3 data where applicable.
        
        Parameters
        ----------
        wave_ranges : list of tuples, optional
            List of (min_wave, max_wave) tuples to keep, e.g. [(2.0e-6, 2.5e-6), (3.0e-6, 4.0e-6)]
        baseline_ranges : list of tuples, optional
            List of (min_baseline, max_baseline) tuples to keep
        freq_ranges : list of tuples, optional
            List of (min_freq, max_freq) tuples to keep
        vis2_err_ranges : list of tuples, optional
            List of (min_vis2_err, max_vis2_err) tuples to keep for V2 error filtering
        t3_err_ranges : list of tuples, optional
            List of (min_t3_err, max_t3_err) tuples to keep for T3 error filtering
        min_wave, max_wave : float, optional
            Single wavelength range in meters (legacy support)
        min_baseline, max_baseline : float, optional
            Single baseline range in meters (legacy support)
        min_freq, max_freq : float, optional
            Single spatial frequency range in rad^-1 (legacy support)
        min_vis2_err, max_vis2_err : float, optional
            Single V2 error range (legacy support)
        min_t3_err, max_t3_err : float, optional
            Single T3 error range (legacy support)
            
        Returns
        -------
        dict
            Filtered data dictionary
            
        Examples
        --------
        # Filter multiple wavelength ranges
        filtered = obs.filter_data(wave_ranges=[(2.0e-6, 2.5e-6), (3.0e-6, 4.0e-6)])
        
        # Filter by error ranges
        filtered = obs.filter_data(
            vis2_err_ranges=[(0, 0.1), (0.2, 0.3)],
            t3_err_ranges=[(0, 5), (10, 15)]
        )
        
        # Combine multiple filter types
        filtered = obs.filter_data(
            baseline_ranges=[(50, 100), (150, 200)],
            vis2_err_ranges=[(0, 0.05)]
        )
        """
        data_dic = self.data
        
        # Helper function to apply multiple range filters
        def apply_range_filter(data_array, ranges):
            if ranges is None:
                return np.ones(len(data_array), dtype=bool)
            
            range_idx = np.zeros(len(data_array), dtype=bool)
            for min_val, max_val in ranges:
                range_idx |= (data_array >= min_val) & (data_array <= max_val)
            return range_idx
        
        # === FILTER V2 DATA ===
        # Start with all indices for V2
        v2_idx = np.ones(len(data_dic['VIS2']), dtype=bool)
        
        # Apply wavelength filters to V2
        if wave_ranges is not None:
            v2_idx &= apply_range_filter(data_dic['VIS2_waves'], wave_ranges)
        else:
            # Legacy single range support
            if min_wave is not None:
                v2_idx &= (data_dic['VIS2_waves'] >= min_wave)
            if max_wave is not None:
                v2_idx &= (data_dic['VIS2_waves'] <= max_wave)
        
        # Apply baseline filters to V2
        if baseline_ranges is not None:
            v2_idx &= apply_range_filter(data_dic['B'], baseline_ranges)
        else:
            # Legacy single range support
            if min_baseline is not None:
                v2_idx &= (data_dic['B'] >= min_baseline)
            if max_baseline is not None:
                v2_idx &= (data_dic['B'] <= max_baseline)
        
        # Apply frequency filters to V2
        if freq_ranges is not None:
            v2_idx &= apply_range_filter(data_dic['freqs'], freq_ranges)
        else:
            # Legacy single range support
            if min_freq is not None:
                v2_idx &= (data_dic['freqs'] >= min_freq)
            if max_freq is not None:
                v2_idx &= (data_dic['freqs'] <= max_freq)
        
        # Apply V2 error filters
        if vis2_err_ranges is not None:
            v2_idx &= apply_range_filter(data_dic['VIS2_err'], vis2_err_ranges)
        else:
            # Legacy single range support
            if min_vis2_err is not None:
                v2_idx &= (data_dic['VIS2_err'] >= min_vis2_err)
            if max_vis2_err is not None:
                v2_idx &= (data_dic['VIS2_err'] <= max_vis2_err)
        
        # Create filtered dictionary for V2 data
        filtered_dic = {}
        v2_keys = ['VIS2', 'VIS2_err', 'VIS2_waves', 'Bu', 'Bv', 'B', 'freqs', 'u', 'v',
                   'VIS2_sta_idx_0', 'VIS2_sta_idx_1', 'VIS2_tel_name_0', 'VIS2_tel_name_1',
                   'INS_VIS2']
        
        for key in v2_keys:
            if key in data_dic:
                filtered_dic[key] = data_dic[key][v2_idx]
        
        # === FILTER T3 DATA (if present) ===
        if 'T3_PHI' in data_dic:
            # Start with all indices for T3
            t3_idx = np.ones(len(data_dic['T3_PHI']), dtype=bool)
            
            # Apply wavelength filters to T3
            if wave_ranges is not None:
                t3_idx &= apply_range_filter(data_dic['T3_waves'], wave_ranges)
            else:
                # Legacy single range support
                if min_wave is not None:
                    t3_idx &= (data_dic['T3_waves'] >= min_wave)
                if max_wave is not None:
                    t3_idx &= (data_dic['T3_waves'] <= max_wave)
            
            # Apply baseline filters to T3 (using max_base)
            if baseline_ranges is not None:
                t3_idx &= apply_range_filter(data_dic['max_base'], baseline_ranges)
            else:
                # Legacy single range support
                if min_baseline is not None:
                    t3_idx &= (data_dic['max_base'] >= min_baseline)
                if max_baseline is not None:
                    t3_idx &= (data_dic['max_base'] <= max_baseline)
            
            # Apply frequency filters to T3 (using max_base/wavelength)
            if 'max_base' in data_dic and 'T3_waves' in data_dic:
                t3_freqs = data_dic['max_base'] / data_dic['T3_waves']
                if freq_ranges is not None:
                    t3_idx &= apply_range_filter(t3_freqs, freq_ranges)
                else:
                    # Legacy single range support
                    if min_freq is not None:
                        t3_idx &= (t3_freqs >= min_freq)
                    if max_freq is not None:
                        t3_idx &= (t3_freqs <= max_freq)
            
            # Apply T3 error filters
            if 'T3_PHI_err' in data_dic:
                if t3_err_ranges is not None:
                    t3_idx &= apply_range_filter(data_dic['T3_PHI_err'], t3_err_ranges)
                else:
                    # Legacy single range support
                    if min_t3_err is not None:
                        t3_idx &= (data_dic['T3_PHI_err'] >= min_t3_err)
                    if max_t3_err is not None:
                        t3_idx &= (data_dic['T3_PHI_err'] <= max_t3_err)
            
            # Filter T3 data
            t3_keys = ['T3_PHI', 'T3_PHI_err', 'T3_waves', 'U1', 'V1', 'U2', 'V2', 'U3', 'V3',
                       'avg_base', 'max_base', 'INS_T3',
                       'T3_sta_idx_0', 'T3_sta_idx_1', 'T3_sta_idx_2',
                       'T3_tel_name_0', 'T3_tel_name_1', 'T3_tel_name_2']
            for key in t3_keys:
                if key in data_dic:
                    filtered_dic[key] = data_dic[key][t3_idx]
            
            print(f"Filtered T3 data: {np.sum(t3_idx)}/{len(t3_idx)} points kept ({np.sum(t3_idx)/len(t3_idx)*100:.1f}%)")
        
        # Copy flux data if present (flux data doesn't depend on wavelength/baseline filters)
        if 'FLUX' in data_dic:
            filtered_dic['FLUX'] = data_dic['FLUX']
            if 'FLUX_err' in data_dic:
                filtered_dic['FLUX_err'] = data_dic['FLUX_err']
            if 'FLUX_sta_idx' in data_dic:
                filtered_dic['FLUX_sta_idx'] = data_dic['FLUX_sta_idx']
        
        # Copy metadata
        for key in ['polarization', 'fringe_tracker', 'TEL_type', 'Telescopes']:
            if key in data_dic:
                filtered_dic[key] = data_dic[key]
        
        print(f"Filtered V2 data: {np.sum(v2_idx)}/{len(v2_idx)} points kept ({np.sum(v2_idx)/len(v2_idx)*100:.1f}%)")
        
        self.data = filtered_dic
        return filtered_dic

    def reset_data(self):
        """Restore self.data to the original state as loaded from disk."""
        self.data = {k: v.copy() if isinstance(v, np.ndarray) else v
                     for k, v in self._raw_data.items()}
        print(f"Data reset to {len(self.data['VIS2'])} VIS2 points")
        return self.data

    def summary(self):
        """Print a concise summary of the loaded data."""
        d = self.data
        instruments = np.unique(d['INS_VIS2']) if 'INS_VIS2' in d else ['Unknown']
        wave_min = d['VIS2_waves'].min() * 1e6
        wave_max = d['VIS2_waves'].max() * 1e6
        n_baselines = len(np.unique(d['B']))
        with np.errstate(invalid='ignore'):
            snr_v2 = d['VIS2'] / np.where(d['VIS2_err'] > 0, d['VIS2_err'], np.nan)
        print('=' * 47)
        print('  Observations Summary')
        print('=' * 47)
        print(f"  Files          : {len(self.file_list)}")
        print(f"  Instrument(s)  : {', '.join(instruments)}")
        print(f"  Wavelengths    : {wave_min:.3f} \u2013 {wave_max:.3f} \u00b5m")
        print(f"  VIS2 points    : {len(d['VIS2'])}")
        print(f"  Baselines      : {n_baselines}")
        print(f"  B range        : {d['B'].min():.1f} \u2013 {d['B'].max():.1f} m")
        print(f"  Median SNR(V\u00b2): {np.nanmedian(snr_v2):.1f}")
        if 'T3_PHI' in d:
            with np.errstate(invalid='ignore'):
                snr_t3 = np.abs(d['T3_PHI']) / np.where(d['T3_PHI_err'] > 0, d['T3_PHI_err'], np.nan)
            print(f"  T3 points      : {len(d['T3_PHI'])}")
            print(f"  Median SNR(CP) : {np.nanmedian(snr_t3):.1f}")
        print('=' * 47)

    def get_effective_resolution(self):
        """Print angular resolution lambda/B per unique baseline."""
        d = self.data
        unique_baselines = np.unique(d['B'])
        mas = 180 * 3600 * 1000 / np.pi  # rad -> mas conversion
        print(f"{'B (m)':>10}  {'lam_min (um)':>13}  {'lam_max (um)':>13}"
              f"  {'theta_min (mas)':>16}  {'theta_max (mas)':>16}")
        print('-' * 75)
        for B in unique_baselines:
            mask = d['B'] == B
            w_min = d['VIS2_waves'][mask].min()
            w_max = d['VIS2_waves'][mask].max()
            print(f"{B:>10.1f}  {w_min*1e6:>13.3f}  {w_max*1e6:>13.3f}"
                  f"  {w_min/B*mas:>16.2f}  {w_max/B*mas:>16.2f}")

    def bin_spectral_channels(self, n):
        """Bin every n spectral channels in-place, updating self.data."""
        d = self.data
        pairs = np.stack([d['VIS2_sta_idx_0'], d['VIS2_sta_idx_1']], axis=1)
        unique_pairs = np.unique(pairs, axis=0)

        scalar_keys = ['Bu', 'Bv', 'B']
        spec_keys   = ['VIS2', 'VIS2_err', 'VIS2_waves', 'freqs', 'u', 'v']
        str_keys    = ['VIS2_sta_idx_0', 'VIS2_sta_idx_1',
                       'VIS2_tel_name_0', 'VIS2_tel_name_1', 'INS_VIS2']
        out = {k: [] for k in spec_keys + scalar_keys + str_keys}

        for pair in unique_pairs:
            mask = (d['VIS2_sta_idx_0'] == pair[0]) & (d['VIS2_sta_idx_1'] == pair[1])
            idx  = np.where(mask)[0][np.argsort(d['VIS2_waves'][mask])]
            for b in range(len(idx) // n):
                sl = idx[b*n:(b+1)*n]
                out['VIS2'].append(np.mean(d['VIS2'][sl]))
                out['VIS2_err'].append(np.sqrt(np.sum(d['VIS2_err'][sl]**2)) / n)
                out['VIS2_waves'].append(np.mean(d['VIS2_waves'][sl]))
                for k in scalar_keys:
                    out[k].append(d[k][sl[0]])
                w_bin = out['VIS2_waves'][-1]
                out['freqs'].append(out['B'][-1] / w_bin)
                out['u'].append(out['Bu'][-1] / w_bin)
                out['v'].append(out['Bv'][-1] / w_bin)
                for k in str_keys:
                    out[k].append(d[k][sl[0]])

        new_data = {k: np.array(v) for k, v in out.items()}

        if 'T3_PHI' in d:
            trips = np.stack([d['T3_sta_idx_0'], d['T3_sta_idx_1'], d['T3_sta_idx_2']], axis=1)
            t3_scalar = ['U1', 'V1', 'U2', 'V2', 'U3', 'V3', 'avg_base', 'max_base']
            t3_str    = ['INS_T3', 'T3_sta_idx_0', 'T3_sta_idx_1', 'T3_sta_idx_2',
                         'T3_tel_name_0', 'T3_tel_name_1', 'T3_tel_name_2']
            t3_out = {k: [] for k in ['T3_PHI', 'T3_PHI_err', 'T3_waves'] + t3_scalar + t3_str}
            for trip in np.unique(trips, axis=0):
                mask = ((d['T3_sta_idx_0'] == trip[0]) & (d['T3_sta_idx_1'] == trip[1]) &
                        (d['T3_sta_idx_2'] == trip[2]))
                idx  = np.where(mask)[0][np.argsort(d['T3_waves'][mask])]
                for b in range(len(idx) // n):
                    sl = idx[b*n:(b+1)*n]
                    t3_out['T3_PHI'].append(np.mean(d['T3_PHI'][sl]))
                    t3_out['T3_PHI_err'].append(np.sqrt(np.sum(d['T3_PHI_err'][sl]**2)) / n)
                    t3_out['T3_waves'].append(np.mean(d['T3_waves'][sl]))
                    for k in t3_scalar + t3_str:
                        t3_out[k].append(d[k][sl[0]])
            for k, v in t3_out.items():
                new_data[k] = np.array(v)

        for key in d:
            if key not in new_data:
                new_data[key] = d[key]

        n_orig = len(d['VIS2'])
        self.data = new_data
        print(f"Binned {n_orig} \u2192 {len(new_data['VIS2'])} VIS2 points ({n}\u00d7 channels/bin)")
        return new_data

    def export_oifits(self, path):
        """Export current (possibly filtered) data to a minimal OIFITS 2 file.

        Parameters
        ----------
        path : str
            Output file path (will overwrite if exists).

        Notes
        -----
        Target coordinates, telescope positions, and timestamps are not stored
        in the data model and will be set to placeholder zeros.
        """
        from astropy.io import fits as pyfits
        d = self.data

        primary = pyfits.PrimaryHDU()
        primary.header['ORIGIN']  = 'vlti_loader'
        primary.header['CONTENT'] = 'OIFITS2'
        hdu_list = [primary]

        # OI_TARGET
        target_hdu = pyfits.BinTableHDU.from_columns([
            pyfits.Column(name='TARGET_ID', format='I',    array=np.array([1],          dtype=np.int16)),
            pyfits.Column(name='TARGET',    format='16A',  array=np.array(['UNKNOWN'])),
            pyfits.Column(name='RAEP0',     format='D',    unit='deg', array=np.array([0.0])),
            pyfits.Column(name='DECEP0',    format='D',    unit='deg', array=np.array([0.0])),
            pyfits.Column(name='EQUINOX',   format='E',    unit='yr',  array=np.array([2000.0], dtype=np.float32)),
            pyfits.Column(name='RA_ERR',    format='D',    unit='deg', array=np.array([0.0])),
            pyfits.Column(name='DEC_ERR',   format='D',    unit='deg', array=np.array([0.0])),
            pyfits.Column(name='SYSVEL',    format='D',    unit='m/s', array=np.array([0.0])),
            pyfits.Column(name='VELTYP',    format='8A',               array=np.array(['UNKNOWN'])),
            pyfits.Column(name='VELDEF',    format='8A',               array=np.array(['OPTICAL'])),
            pyfits.Column(name='PMRA',      format='D',    unit='deg/yr', array=np.array([0.0])),
            pyfits.Column(name='PMDEC',     format='D',    unit='deg/yr', array=np.array([0.0])),
            pyfits.Column(name='PMRA_ERR',  format='D',    unit='deg/yr', array=np.array([0.0])),
            pyfits.Column(name='PMDEC_ERR', format='D',    unit='deg/yr', array=np.array([0.0])),
            pyfits.Column(name='PARALLAX',  format='E',    unit='deg', array=np.array([0.0], dtype=np.float32)),
            pyfits.Column(name='PARA_ERR',  format='E',    unit='deg', array=np.array([0.0], dtype=np.float32)),
            pyfits.Column(name='SPECTYP',   format='16A',              array=np.array(['UNKNOWN'])),
        ])
        target_hdu.header['EXTNAME'] = 'OI_TARGET'
        hdu_list.append(target_hdu)

        # OI_ARRAY
        tel_names = np.unique(np.concatenate([d['VIS2_tel_name_0'], d['VIS2_tel_name_1']]))
        sta_indices = np.arange(1, len(tel_names) + 1, dtype=np.int16)
        tel_to_sta = {t: i for t, i in zip(tel_names, sta_indices)}
        tel_type = d.get('TEL_type', 'AT')
        if isinstance(tel_type, np.ndarray):
            tel_type = tel_type[0]
        diameter = 8.2 if tel_type == 'UT' else 1.8
        array_hdu = pyfits.BinTableHDU.from_columns([
            pyfits.Column(name='TEL_NAME',  format='16A', array=tel_names),
            pyfits.Column(name='STA_NAME',  format='16A', array=tel_names),
            pyfits.Column(name='STA_INDEX', format='I',   array=sta_indices),
            pyfits.Column(name='DIAMETER',  format='E',   unit='m',
                          array=np.full(len(tel_names), diameter, dtype=np.float32)),
            pyfits.Column(name='STAXYZ',    format='3D',  unit='m',
                          array=np.zeros((len(tel_names), 3))),
        ])
        array_hdu.header['EXTNAME'] = 'OI_ARRAY'
        array_hdu.header['ARRNAME'] = 'VLTI'
        array_hdu.header['FRAME']   = 'GEOCENTRIC'
        array_hdu.header['ARRAYX']  = 0.0
        array_hdu.header['ARRAYY']  = 0.0
        array_hdu.header['ARRAYZ']  = 0.0
        hdu_list.append(array_hdu)

        # OI_WAVELENGTH + OI_VIS2 per instrument
        for ins in np.unique(d['INS_VIS2']):
            ins_mask = d['INS_VIS2'] == ins
            waves_ins = np.unique(d['VIS2_waves'][ins_mask])
            n_waves = len(waves_ins)
            bw = float(np.diff(waves_ins).mean()) if n_waves > 1 else 1e-7

            wave_hdu = pyfits.BinTableHDU.from_columns([
                pyfits.Column(name='EFF_WAVE', format='E', unit='m',
                              array=waves_ins.astype(np.float32)),
                pyfits.Column(name='EFF_BAND', format='E', unit='m',
                              array=np.full(n_waves, bw, dtype=np.float32)),
            ])
            wave_hdu.header['EXTNAME'] = 'OI_WAVELENGTH'
            wave_hdu.header['INSNAME'] = ins
            hdu_list.append(wave_hdu)

            # build per-baseline rows: group by (tel0, tel1, Bu, Bv)
            ins_d = {k: v[ins_mask] for k, v in d.items()
                     if isinstance(v, np.ndarray) and len(v) == len(d['VIS2'])}
            row_keys, seen = [], {}
            for i, rk in enumerate(zip(ins_d['VIS2_tel_name_0'], ins_d['VIS2_tel_name_1'],
                                       np.round(ins_d['Bu'], 3), np.round(ins_d['Bv'], 3))):
                if rk not in seen:
                    seen[rk] = []
                    row_keys.append(rk)
                seen[rk].append(i)
            n_rows = len(row_keys)
            flags    = np.ones((n_rows, n_waves), dtype=bool)
            vis2data = np.zeros((n_rows, n_waves))
            vis2err  = np.zeros((n_rows, n_waves))
            ucoord   = np.zeros(n_rows)
            vcoord   = np.zeros(n_rows)
            sta_idx  = np.zeros((n_rows, 2), dtype=np.int16)
            for ri, rk in enumerate(row_keys):
                sl = seen[rk]
                for j in sl:
                    wi = np.searchsorted(waves_ins, ins_d['VIS2_waves'][j])
                    if wi < n_waves and np.isclose(waves_ins[wi], ins_d['VIS2_waves'][j], rtol=1e-6):
                        vis2data[ri, wi] = ins_d['VIS2'][j]
                        vis2err[ri, wi]  = ins_d['VIS2_err'][j]
                        flags[ri, wi]    = False
                ucoord[ri]  = ins_d['Bu'][sl[0]]
                vcoord[ri]  = ins_d['Bv'][sl[0]]
                sta_idx[ri] = [tel_to_sta[rk[0]], tel_to_sta[rk[1]]]
            vis2_hdu = pyfits.BinTableHDU.from_columns([
                pyfits.Column(name='TARGET_ID', format='I',              array=np.ones(n_rows, dtype=np.int16)),
                pyfits.Column(name='TIME',      format='D', unit='s',    array=np.zeros(n_rows)),
                pyfits.Column(name='MJD',       format='D', unit='day',  array=np.zeros(n_rows)),
                pyfits.Column(name='INT_TIME',  format='D', unit='s',    array=np.ones(n_rows)),
                pyfits.Column(name='VIS2DATA',  format=f'{n_waves}D',    array=vis2data),
                pyfits.Column(name='VIS2ERR',   format=f'{n_waves}D',    array=vis2err),
                pyfits.Column(name='UCOORD',    format='D', unit='m',    array=ucoord),
                pyfits.Column(name='VCOORD',    format='D', unit='m',    array=vcoord),
                pyfits.Column(name='STA_INDEX', format='2I',             array=sta_idx),
                pyfits.Column(name='FLAG',      format=f'{n_waves}L',    array=flags),
            ])
            vis2_hdu.header['EXTNAME'] = 'OI_VIS2'
            vis2_hdu.header['INSNAME'] = ins
            vis2_hdu.header['ARRNAME'] = 'VLTI'
            hdu_list.append(vis2_hdu)

        # OI_T3
        if 'T3_PHI' in d:
            for ins in np.unique(d['INS_T3']):
                t3_mask = d['INS_T3'] == ins
                waves_ins = np.unique(d['T3_waves'][t3_mask])
                n_waves   = len(waves_ins)
                t3_d = {k: v[t3_mask] for k, v in d.items()
                        if isinstance(v, np.ndarray) and len(v) == len(d['T3_PHI'])}
                row_keys, seen = [], {}
                for i, rk in enumerate(zip(t3_d['T3_tel_name_0'], t3_d['T3_tel_name_1'],
                                           t3_d['T3_tel_name_2'],
                                           np.round(t3_d['U1'], 3), np.round(t3_d['V1'], 3),
                                           np.round(t3_d['U2'], 3), np.round(t3_d['V2'], 3))):
                    if rk not in seen:
                        seen[rk] = []
                        row_keys.append(rk)
                    seen[rk].append(i)
                n_rows   = len(row_keys)
                t3flags  = np.ones((n_rows, n_waves), dtype=bool)
                t3phi    = np.zeros((n_rows, n_waves))
                t3phierr = np.zeros((n_rows, n_waves))
                u1c = np.zeros(n_rows); v1c = np.zeros(n_rows)
                u2c = np.zeros(n_rows); v2c = np.zeros(n_rows)
                sta3 = np.zeros((n_rows, 3), dtype=np.int16)
                for ri, rk in enumerate(row_keys):
                    sl = seen[rk]
                    for j in sl:
                        wi = np.searchsorted(waves_ins, t3_d['T3_waves'][j])
                        if wi < n_waves and np.isclose(waves_ins[wi], t3_d['T3_waves'][j], rtol=1e-6):
                            t3phi[ri, wi]    = t3_d['T3_PHI'][j]
                            t3phierr[ri, wi] = t3_d['T3_PHI_err'][j]
                            t3flags[ri, wi]  = False
                    u1c[ri] = t3_d['U1'][sl[0]]; v1c[ri] = t3_d['V1'][sl[0]]
                    u2c[ri] = t3_d['U2'][sl[0]]; v2c[ri] = t3_d['V2'][sl[0]]
                    sta3[ri] = [tel_to_sta.get(rk[0], 1), tel_to_sta.get(rk[1], 2),
                                tel_to_sta.get(rk[2], 3)]
                t3_hdu = pyfits.BinTableHDU.from_columns([
                    pyfits.Column(name='TARGET_ID', format='I',              array=np.ones(n_rows, dtype=np.int16)),
                    pyfits.Column(name='TIME',      format='D', unit='s',    array=np.zeros(n_rows)),
                    pyfits.Column(name='MJD',       format='D', unit='day',  array=np.zeros(n_rows)),
                    pyfits.Column(name='INT_TIME',  format='D', unit='s',    array=np.ones(n_rows)),
                    pyfits.Column(name='T3AMP',     format=f'{n_waves}D',    array=np.ones((n_rows, n_waves))),
                    pyfits.Column(name='T3AMPERR',  format=f'{n_waves}D',    array=np.zeros((n_rows, n_waves))),
                    pyfits.Column(name='T3PHI',     format=f'{n_waves}D',    unit='deg', array=t3phi),
                    pyfits.Column(name='T3PHIERR',  format=f'{n_waves}D',    unit='deg', array=t3phierr),
                    pyfits.Column(name='U1COORD',   format='D', unit='m',    array=u1c),
                    pyfits.Column(name='V1COORD',   format='D', unit='m',    array=v1c),
                    pyfits.Column(name='U2COORD',   format='D', unit='m',    array=u2c),
                    pyfits.Column(name='V2COORD',   format='D', unit='m',    array=v2c),
                    pyfits.Column(name='STA_INDEX', format='3I',             array=sta3),
                    pyfits.Column(name='FLAG',      format=f'{n_waves}L',    array=t3flags),
                ])
                t3_hdu.header['EXTNAME'] = 'OI_T3'
                t3_hdu.header['INSNAME'] = ins
                t3_hdu.header['ARRNAME'] = 'VLTI'
                hdu_list.append(t3_hdu)

        pyfits.HDUList(hdu_list).writeto(path, overwrite=True)
        print(f"Exported OIFITS to {path}")

    def plot(self, uv_bool=True, model_vis2=None, model_t3=None, error_bars_v2=None, error_bars_t3=None, 
             v2_ylim=None, cp_ylim=None, show=True):
        """
        Plot visibility and closure phase data.
        
        Parameters
        ----------
        uv_bool : bool, optional
            Whether to plot uv coverage (default: True)
        model_vis2 : array_like, optional
            Model visibility squared values to overlay
        model_t3 : array_like, optional
            Model closure phase values to overlay
        error_bars_v2 : array_like, optional
            Custom error bars for V2 plot
        error_bars_t3 : array_like, optional
            Custom error bars for CP plot
        v2_ylim : tuple, optional
            Y-axis limits for V2 plot as (ymin, ymax)
        cp_ylim : tuple, optional
            Y-axis limits for closure phase plot as (ymin, ymax)
        show : bool, optional
            Whether to display the plot (default: True)
            
        Returns
        -------
        matplotlib.figure.Figure
            The created figure object
        """
        data_dic = self.data
        if 'T3_PHI' in data_dic:
            N_rows = 2
            figsize_y = 9
            height_ratios = [1.7,1]
        else:
            N_rows = 1
            figsize_y = 5
            height_ratios = [1]

        if uv_bool:
            V2 = data_dic['VIS2']
            V2_err = np.array(data_dic['VIS2_err'])

            waves = data_dic['VIS2_waves']
            

            B_u = np.array(data_dic['Bu'])
            B_v = np.array(data_dic['Bv'])
            B = np.sqrt(B_u**2+B_v**2)

            fig = plt.figure(figsize=(30,figsize_y))

            gs1 = GridSpec(N_rows, 2, left=0.05, right=0.48, wspace=0.3, hspace=0.2, width_ratios=[1,2], height_ratios=height_ratios)
            ax1 = fig.add_subplot(gs1[0, 0])
            ax2 = fig.add_subplot(gs1[0,1])
            scatter_obs = ax2.scatter(B/waves, V2, c = waves, cmap='turbo', s=10)
            if error_bars_v2 is None:
                ax2.errorbar(B/waves, V2, V2_err, linestyle = '',  c='lightgrey', alpha = 0.5, zorder=0)
            else:
                ax2.errorbar(B/waves, V2, error_bars_v2, linestyle = '',  c='lightgrey', alpha = 0.5, zorder=0)

            ax2.grid(visible=True, which='both', axis='both')
            ax2.set_ylabel(r'${V^2}$')
            ax2.set_xlabel(r'f ${[rad^{-1}]}$')
            
            # Set V2 plot y-limits if specified
            if v2_ylim is not None:
                ax2.set_ylim(v2_ylim[0], v2_ylim[1])

            divider = make_axes_locatable(ax2)
            cax = divider.append_axes('right', size='5%', pad=0.05)

            ax1.scatter(B_u/waves, B_v/waves, c=waves, cmap='turbo')
            ax1.scatter(-B_u/waves, -B_v/waves, c=waves, cmap='turbo')
            ax1.set_ylabel(r'v $[rad^{-1}]$')
            ax1.set_xlabel(r'u $[rad^{-1}]$')
            ax1.set_xlim(ax1.get_xlim()[::-1])
            ax1.grid(visible=True, which='major', axis='both')

            fig.colorbar(scatter_obs, cax=cax, orientation='vertical')
            cax.set_title(r'$\lambda$ [m]')
            
            if not model_vis2 is None:
                ax2.scatter(B/waves, model_vis2, c = 'blue', s=10)
            
            if 'T3_PHI' in data_dic:
                B_max = data_dic['max_base']
                t3_phi = data_dic['T3_PHI']
                t3_phi_err = data_dic['T3_PHI_err']
                t3_waves = data_dic['T3_waves']
                
                ax3 = fig.add_subplot(gs1[1,1])
                scatter_obs = ax3.scatter(B_max/t3_waves, t3_phi, c = t3_waves, cmap='turbo', s=10)
                if error_bars_t3 is None:
                    ax3.errorbar(B_max/t3_waves, t3_phi, t3_phi_err, linestyle = '',  c='lightgrey', alpha = 0.5, zorder=0)
                else:
                    ax3.errorbar(B_max/t3_waves, t3_phi, error_bars_t3, linestyle = '',  c='lightgrey', alpha = 0.5, zorder=0)

                ax3.grid(visible=True, which='both', axis='both')
                ax3.set_ylabel('CP [deg]')
                ax3.set_xlabel(r'${B_{max}/\lambda}$ ${[rad^{-1}]}$')
                
                # Set CP plot y-limits if specified
                if cp_ylim is not None:
                    ax3.set_ylim(cp_ylim[0], cp_ylim[1])
                
                if not model_t3 is None:
                    scatter_obs = ax3.scatter(B_max/t3_waves, model_t3, c = 'blue', s=10)
            
            if show:
                plt.show()
            else:
                plt.close(fig)
            
            return fig
        else:
            print('The option you selected is not available yet :(')

    def plot_report_by_base(self, V2_min=0.6, V2_max=1.4, T3_min=-20, T3_max=20, show=True):
        """Plots V2 vs wavelength and closure phase vs wavelength on separate subplots,
        one per baseline / triangle. Also includes the UV-plane colored per baseline.
        Out-of-range points are indicated with arrows. An SNR(V²) panel sits below
        the visibility plots.

        Parameters
        ----------
        V2_min : float, optional
            Lower y-axis limit for V2, by default 0.6
        V2_max : float, optional
            Upper y-axis limit for V2, by default 1.4
        T3_min : float, optional
            Lower y-axis limit for closure phase [deg], by default -20
        T3_max : float, optional
            Upper y-axis limit for closure phase [deg], by default 20
        show : bool, optional
            Whether to display the figure, by default True

        Returns
        -------
        fig : matplotlib.figure.Figure
        """
        data_dic = self.data
        has_t3 = 'T3_PHI' in data_dic

        V2         = data_dic['VIS2']
        V2_err     = data_dic['VIS2_err']
        waves      = data_dic['VIS2_waves']
        baselines  = data_dic['B']
        B_u        = np.array(data_dic['Bu'])
        B_v        = np.array(data_dic['Bv'])
        tel_name_0 = data_dic['VIS2_tel_name_0']
        tel_name_1 = data_dic['VIS2_tel_name_1']

        # Group by sorted telescope pair label (e.g. "U1-U2") so multiple
        # baselines with the same pair end up on the same subplot
        pair_labels = np.array([
            '-'.join(sorted([a, b])) for a, b in zip(tel_name_0, tel_name_1)
        ])
        unique_pairs = np.unique(pair_labels)
        n_pairs = len(unique_pairs)

        if has_t3:
            t3_phi     = data_dic['T3_PHI']
            t3_phi_err = data_dic['T3_PHI_err']
            t3_waves   = data_dic['T3_waves']
            t3_tel0    = data_dic['T3_tel_name_0']
            t3_tel1    = data_dic['T3_tel_name_1']
            t3_tel2    = data_dic['T3_tel_name_2']
            triplet_labels = np.array(
                [f"{a}-{b}-{c}" for a, b, c in zip(t3_tel0, t3_tel1, t3_tel2)]
            )
            unique_triplets = np.unique(triplet_labels)
            n_triplets = len(unique_triplets)
        else:
            n_triplets = 0

        # ── Figure / GridSpec layout ──────────────────────────────────────────
        # Row 0 : UV plane (col 0) | V2 subplots (col 1) | CP subplots (col 2, if T3)
        # Row 1 : SNR aligned with the V2 column
        n_data_rows = max(n_pairs, n_triplets) if has_t3 else n_pairs
        n_cols = 3 if has_t3 else 2
        figsize_y = 16 if has_t3 else 11

        fig = plt.figure(figsize=(18, figsize_y))

        gs_outer = GridSpec(2, n_cols, figure=fig,
                            left=0.06, right=0.97,
                            hspace=0.15, wspace=0.35,
                            height_ratios=[3, 1],
                            width_ratios=[1] + [2] * (n_cols - 1))

        # Left column: UV plane pinned to top, sized to ~3 V2-subplot heights
        uv_rows = min(3, n_data_rows)
        gs_left = gs_outer[0, 0].subgridspec(n_data_rows, 1, hspace=0)
        ax_uv = fig.add_subplot(gs_left[:uv_rows, 0])

        # V2 column (row 0, col 1)
        gs_v2 = gs_outer[0, 1].subgridspec(n_data_rows, 1, hspace=0.15)
        colors_v2 = plt.get_cmap('tab10')(np.linspace(0, 1, n_pairs))

        # Draw UV plane colored per telescope pair
        for color_uv, pair in zip(colors_v2, unique_pairs):
            mask_uv = pair_labels == pair
            if not np.any(mask_uv):
                continue
            u_bl = B_u[mask_uv] / waves[mask_uv]
            v_bl = B_v[mask_uv] / waves[mask_uv]
            ax_uv.scatter( u_bl,  v_bl, color=color_uv, s=8, alpha=0.8)
            ax_uv.scatter(-u_bl, -v_bl, color=color_uv, s=8, alpha=0.8)
        ax_uv.set_xlabel(r'u $[rad^{-1}]$')
        ax_uv.set_ylabel(r'v $[rad^{-1}]$')
        ax_uv.set_aspect('equal', adjustable='datalim')
        ax_uv.invert_xaxis()
        ax_uv.grid(visible=True, which='major', axis='both')

        # CP column (row 0, col 2)
        if has_t3:
            gs_right = gs_outer[0, 2].subgridspec(n_data_rows, 1, hspace=0.15)

        # snr_data: per-pair SNR entries for pairs with > 1 unique baseline
        # snr_single: pooled SNR data for pairs with exactly 1 unique baseline
        snr_data = []    # entries: (list_of_w, list_of_snr, color, pair)
        snr_single = []  # entries: (w_array, snr_array, color, pair)  — pooled later

        for i, (color, pair) in enumerate(zip(colors_v2, unique_pairs)):
            pair_mask = pair_labels == pair

            ax = fig.add_subplot(gs_v2[i, 0])

            # Determine y-limits from all data for this pair
            if V2_min is not None and V2_max is not None:
                V2_min_lim, V2_max_lim = V2_min, V2_max
            else:
                V2_min_lim = np.min((V2 - V2_err)[pair_mask])
                V2_max_lim = np.max((V2 + V2_err)[pair_mask])
            ax.set_ylim(V2_min_lim, V2_max_lim)

            # Each unique baseline within the pair gets its own curve
            pair_baselines = np.unique(baselines[pair_mask])
            all_snr_w, all_snr_v = [], []
            for base_val in pair_baselines:
                bl_mask = pair_mask & (baselines == base_val)
                if not np.any(bl_mask):
                    continue
                w_bl   = waves[bl_mask] * 1e6
                v2_bl  = V2[bl_mask]
                err_bl = V2_err[bl_mask]
                snr_bl = v2_bl / np.where(err_bl > 0, err_bl, np.nan)
                order  = np.argsort(w_bl)
                w_bl, v2_bl, err_bl, snr_bl = w_bl[order], v2_bl[order], err_bl[order], snr_bl[order]

                ax.plot(w_bl, v2_bl, color=color, linewidth=1.2, alpha=0.9)
                ax.fill_between(w_bl, v2_bl - err_bl, v2_bl + err_bl,
                                color=color, alpha=0.2, linewidth=0)
                for x, y in zip(w_bl, v2_bl):
                    if y > V2_max_lim:
                        ax.annotate('', xy=(x, V2_max_lim), xytext=(x, V2_max_lim * 0.98),
                                    arrowprops=dict(facecolor=color, edgecolor=color,
                                                    shrink=0.05, width=2, headwidth=6))
                    elif y < V2_min_lim:
                        ax.annotate('', xy=(x, V2_min_lim),
                                    xytext=(x, V2_min_lim + (V2_max_lim - V2_min_lim) * 0.02),
                                    arrowprops=dict(facecolor=color, edgecolor=color,
                                                    shrink=0.05, width=2, headwidth=6))
                all_snr_w.append(w_bl)
                all_snr_v.append(snr_bl)

            ax.set_ylabel(r'$V^2$', fontsize=9)
            ax.text(0.02, 0.95, pair, transform=ax.transAxes,
                    fontsize=9, color=color, fontweight='bold',
                    va='top', ha='left')
            ax.grid(visible=True, which='both', axis='both')
            if i == n_data_rows - 1 or i == n_pairs - 1:
                ax.set_xlabel(r'$\lambda$ [$\mu$m]', fontsize=9)
            else:
                ax.set_xticklabels([])

            # Pairs with > 1 baseline get their own SNR subplot;
            # pairs with exactly 1 baseline are pooled into a shared SNR subplot.
            if len(pair_baselines) > 1 and all_snr_w:
                snr_data.append((all_snr_w, all_snr_v, color, pair))
            elif len(pair_baselines) == 1 and all_snr_w:
                snr_single.append((all_snr_w[0], all_snr_v[0], color, pair))

        # ── Per-pair SNR panels (pairs with > 1 unique baseline) ─────────────
        # Plus one pooled panel for all single-baseline pairs (if any).
        # If only the pooled panel exists, constrain it to the V2 column only.
        # Otherwise span V2 + CP columns for width, up to 2 rows.
        all_snr_entries = snr_data + ([snr_single] if snr_single else [])
        if all_snr_entries:
            n_snr = len(all_snr_entries)
            only_pooled = (len(snr_data) == 0)  # only the shared single-baseline panel
            n_snr_rows = min(2, n_snr)
            n_snr_cols = int(np.ceil(n_snr / n_snr_rows))
            snr_col_span = gs_outer[1, 1] if only_pooled else gs_outer[1, 1:]
            gs_snr = snr_col_span.subgridspec(n_snr_rows, n_snr_cols,
                                              wspace=0.35, hspace=0.45)
            for k, entry in enumerate(all_snr_entries):
                row_k, col_k = divmod(k, n_snr_cols)
                ax_snr = fig.add_subplot(gs_snr[row_k, col_k])

                if entry is snr_single:
                    # Pooled panel: plot all single-baseline pairs in their pair colour
                    for w_bl, snr_bl, color, pair in snr_single:
                        ax_snr.plot(w_bl, snr_bl, color=color, linewidth=1.2, alpha=0.9)
                else:
                    w_list, snr_list, color, pair = entry
                    for w_bl, snr_bl in zip(w_list, snr_list):
                        ax_snr.plot(w_bl, snr_bl, color=color, linewidth=1.2, alpha=0.9)

                ax_snr.set_ylabel('SNR(V²)', fontsize=9)
                ax_snr.set_xlabel(r'$\lambda$ [$\mu$m]', fontsize=9)
                ax_snr.grid(visible=True, which='both', axis='both')

        # ── Closure-phase subplots (column 2) ────────────────────────────────
        if has_t3:
            colors_t3 = plt.get_cmap('tab10')(np.linspace(0, 1, n_triplets))

            for j, (color, triplet) in enumerate(zip(colors_t3, unique_triplets)):
                triplet_mask = triplet_labels == triplet

                ax = fig.add_subplot(gs_right[j, 0])

                # Determine y-limits from all data for this triplet
                if T3_min is not None and T3_max is not None:
                    T3_min_lim, T3_max_lim = T3_min, T3_max
                else:
                    T3_min_lim = np.min((t3_phi - t3_phi_err)[triplet_mask])
                    T3_max_lim = np.max((t3_phi + t3_phi_err)[triplet_mask])
                ax.set_ylim(T3_min_lim, T3_max_lim)

                # Each unique triangle (by max_base) within the triplet gets its own curve
                triplet_max_bases = np.unique(data_dic['max_base'][triplet_mask])
                for base_val in triplet_max_bases:
                    bl_mask = triplet_mask & (data_dic['max_base'] == base_val)
                    if not np.any(bl_mask):
                        continue
                    w_tr   = t3_waves[bl_mask] * 1e6
                    cp_tr  = t3_phi[bl_mask]
                    err_tr = t3_phi_err[bl_mask]
                    order  = np.argsort(w_tr)
                    w_tr, cp_tr, err_tr = w_tr[order], cp_tr[order], err_tr[order]

                    ax.plot(w_tr, cp_tr, color=color, linewidth=1.2, alpha=0.9)
                    ax.fill_between(w_tr, cp_tr - err_tr, cp_tr + err_tr,
                                    color=color, alpha=0.2, linewidth=0)
                    for x, y in zip(w_tr, cp_tr):
                        if y > T3_max_lim:
                            ax.annotate('', xy=(x, T3_max_lim), xytext=(x, T3_max_lim * 0.98),
                                        arrowprops=dict(facecolor=color, edgecolor=color,
                                                        shrink=0.05, width=2, headwidth=6))
                        elif y < T3_min_lim:
                            ax.annotate('', xy=(x, T3_min_lim),
                                        xytext=(x, T3_min_lim + (T3_max_lim - T3_min_lim) * 0.02),
                                        arrowprops=dict(facecolor=color, edgecolor=color,
                                                        shrink=0.05, width=2, headwidth=6))

                ax.set_ylabel('CP [deg]', fontsize=9)
                ax.text(0.02, 0.95, triplet, transform=ax.transAxes,
                        fontsize=9, color=color, fontweight='bold',
                        va='top', ha='left')
                ax.grid(visible=True, which='both', axis='both')
                if j == n_data_rows - 1 or j == n_triplets - 1:
                    ax.set_xlabel(r'$\lambda$ [$\mu$m]', fontsize=9)
                else:
                    ax.set_xticklabels([])

        if show:
            plt.show()
        else:
            plt.close(fig)

        return fig

if __name__=='__main__':
    # path_obs_MAT = '/Users/prioletp/PhD/ESO_visitor_program/MATISSE_data/HD113766/HD113766_2024_03_29/MERGED/2024-03-29T014654_HD__113766A_U1U2U3U4_IR-LM_LOW_noChop_cal_oifits_0.fits'
    # path_obs_GRAV = '/Users/prioletp/PhD/ESO_visitor_program/GRAVITY_data/HD113766/HD113766_27_06_24/reduced/calibrated/calibrated/GRAVI.2025-01-28T06:33:51.886_singlescivis_singlesciviscalibrated.fits'
    
    
    # path_all = '/Users/prioletp/PhD/ESO_visitor_program/MATISSE_data/HD172555/HD172555_2022_04_22'
    # path_all = [path_obs_MAT, path_obs_GRAV]
    path_all = '/Users/prioletp/PhD/ESO_visitor_program/MATISSE_data/HD113766/HD113766_10_03_25/MERGED/2025-03-12T033445_HD113766A_A0B2D0C1_IR-LM_LOW_noChop_cal_oifits_0.fits'#'/Users/prioletp/PhD/ESO_visitor_program/MATISSE_data/HD172555/HD172555_2022_04_22/Iter4_OIFITS/2022-04-22T094100_HD_172555_U1U2U3U4_IR-LM_LOW_IN_IN_Chop.fits'
    # observations = Observations(path_obs_MAT)
    # observations_GRAV = Observations(path_obs_GRAV)
    observations = Observations(path_all)
    wave_range = [(3.2e-6, 3.6e-6)]
    observations.filter_data(wave_ranges=wave_range)
    observations.plot(uv_bool=True, v2_ylim=(0, 1.4), cp_ylim=(-20, 20))
    observations.plot_report_by_base(V2_min=None, V2_max=None, T3_min=-20, T3_max=20)
    # V2_err = observations.data['VIS2_err'][observations.data['VIS2_err']<0]
    # plt.plot(np.arange(0,len(V2_err), 1), V2_err)
    # print(np.min(V2_err))
    # observations.plot(v2_ylim=(0, 1), cp_ylim=(-50, 50))
    #%%
    # for elem in observations.data_dic_array:
    #     # print(elem)
    #     print(elem.keys())
    data = observations.data
    v2 = data['VIS2']
    v2_err = data['VIS2_err']
    baselines = data['B']
    waves = data['VIS2_waves']
    freqs = data['freqs']
    for i in np.unique(baselines):
        mask = baselines == i
        # print(f"Baseline {i} m: {np.sum(mask)} points, V2 range: {np.min(v2[mask]):.3f} - {np.max(v2[mask]):.3f}, wavelength range: {np.min(waves[mask])*1e6:.2f} - {np.max(waves[mask])*1e6:.2f} µm, freq range: {np.min(freqs[mask]):.2e} - {np.max(freqs[mask]):.2e} rad^-1")
        v2_masked = v2[mask]
        v2_err_masked = v2_err[mask]
        waves_masked = waves[mask]
        freqs_masked = freqs[mask]
        order  = np.argsort(waves_masked)

        # plt.plot(waves_masked[order]*1e6, v2_masked[order], label=f'Baseline {i} m')
        plt.plot(waves_masked[order], v2_masked[order], linewidth=1.2, alpha=0.9)
        plt.fill_between(waves_masked[order], v2_masked[order] - v2_err_masked[order], v2_masked[order] + v2_err_masked[order],
                         alpha=0.2, linewidth=0)
        plt.legend()
        plt.show()
        plt.clf()


# %%
