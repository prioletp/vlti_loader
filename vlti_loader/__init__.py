import os
import numpy as np
from pathlib import Path
from astropy.io import fits
from . import utils as oi_utl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.gridspec import GridSpec

class Observations:
    def __init__(self, path):
        """Load VLTI interferometric data from one or more OIFITS files.

        Parameters
        ----------
        path : str, Path, list, or tuple
            Source of OIFITS data. Accepted forms:

            * A path to a single ``.fits`` file.
            * A path to a directory — all ``.fits`` files inside are loaded
              and merged into a single data dictionary.
            * A list or tuple of file paths, useful for combining data from
              multiple instruments (e.g. MATISSE + GRAVITY together).

        Raises
        ------
        FileNotFoundError
            If the given path or any file in the list does not exist.
        ValueError
            If the path is a directory with no FITS files, or a file without
            a ``.fits`` extension.
        TypeError
            If *path* is not a ``str``, ``Path``, ``list``, or ``tuple``.

        Notes
        -----
        After construction, the loaded data is available via ``self.data`` (a
        flat ``dict`` of 1-D numpy arrays) or as attribute shortcuts, e.g.
        ``obs.VIS2``.  Use :meth:`filter_data` to restrict the data to a
        wavelength or baseline range, and :meth:`reset_data` to undo any
        filtering.

        Examples
        --------
        >>> obs = Observations('/data/star.fits')
        >>> obs = Observations('/data/MATISSE_reduced/')
        >>> obs = Observations(['/data/MATISSE.fits', '/data/GRAVITY.fits'])
        """
        self.path = path
        self.data_type = None
        self.file_list = []
        
        # Check if path is an array/list
        if isinstance(path, (list, tuple, np.ndarray)):
            self.data_type = 'array'
            self.file_list = list(path)
            # Validate that all elements are valid file paths
            for file_path in self.file_list:
                if not os.path.exists(file_path):
                    raise FileNotFoundError(f"File not found in array: {file_path}")
                if not str(file_path).lower().endswith('.fits'):
                    raise ValueError(f"Non-FITS file in array: {file_path}")
        
        # Check if path is a string (file or directory path)
        elif isinstance(path, (str, Path)):
            path_obj = Path(path)
            
            if not path_obj.exists():
                raise FileNotFoundError(f"Path does not exist: {path}")
            
            # Check if it's a FITS file
            if path_obj.is_file() and str(path).lower().endswith('.fits'):
                self.data_type = 'fits_file'
                self.file_list = [str(path)]
            
            # Check if it's a directory
            elif path_obj.is_dir():
                self.data_type = 'directory'
                # Find all FITS files in the directory
                fits_files = list(path_obj.glob('*.fits')) + list(path_obj.glob('*.FITS'))
                if not fits_files:
                    raise ValueError(f"No FITS files found in directory: {path}")
                self.file_list = [str(f) for f in sorted(fits_files)]
            
            else:
                raise ValueError(f"Path is not a FITS file or directory: {path}")
        
        else:
            raise TypeError(f"Invalid type for path: {type(path)}. Expected string, Path, list, or array.")
        
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
        """Read all FITS files in ``self.file_list`` and populate ``self.data``.

        Detects the instrument (GRAVITY, MATISSE, PIONIER) from the FITS
        header, dispatches to the appropriate parser in
        ``vlti_loader.utils``, and merges all files into a single flat
        data dictionary stored as ``self.data``.  A copy of the original data
        is kept in ``self._raw_data`` so that :meth:`reset_data` can restore
        the unfiltered state.

        Returns
        -------
        dict
            The merged data dictionary (same object as ``self.data``).

        Notes
        -----
        This method is called automatically by :meth:`__init__` and does not
        normally need to be called by the user.
        """
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
    
    def filter_data(self,
                   # ── Primary API ───────────────────────────────────────────
                   wave=None, baseline=None, freq=None, vis2_err=None, cp_err=None,
                   # ── Legacy *_ranges aliases ───────────────────────────────
                   wave_ranges=None, baseline_ranges=None, freq_ranges=None,
                   vis2_err_ranges=None, t3_err_ranges=None,
                   # ── Legacy single-bound aliases ───────────────────────────
                   min_wave=None, max_wave=None, min_baseline=None,
                   max_baseline=None, min_freq=None, max_freq=None,
                   min_vis2_err=None, max_vis2_err=None, min_t3_err=None, max_t3_err=None):
        """Filter the data in-place and update ``self.data``.

        Pass a ``(min, max)`` tuple (or a list of tuples for multiple disjoint
        windows) to each parameter.  Filters are combined with AND across
        parameters; multiple windows within one parameter are combined with OR.
        Use :meth:`reset_data` to restore the full unfiltered dataset.

        Parameters
        ----------
        wave : (float, float) or list of (float, float), optional
            Wavelength window(s) to keep in metres.
            ``(3.2e-6, 3.8e-6)`` keeps a single band;
            ``[(1.5e-6, 1.8e-6), (2.0e-6, 2.4e-6)]`` keeps two bands.
        baseline : (float, float) or list of (float, float), optional
            Projected baseline window(s) to keep in metres.
        freq : (float, float) or list of (float, float), optional
            Spatial-frequency window(s) to keep in rad⁻¹ (= B/λ).
        vis2_err : (float, float) or list of (float, float), optional
            V² uncertainty window(s) to keep.
        cp_err : (float, float) or list of (float, float), optional
            Closure-phase uncertainty window(s) to keep in degrees.
        wave_ranges, baseline_ranges, freq_ranges, vis2_err_ranges, t3_err_ranges : list of (float, float), optional
            Legacy aliases — prefer the shorthand parameters above.
        min_wave, max_wave, min_baseline, max_baseline, min_freq, max_freq, min_vis2_err, max_vis2_err, min_t3_err, max_t3_err : float, optional
            Legacy single-bound aliases.

        Returns
        -------
        dict
            The filtered data dictionary (same object as ``self.data``).

        Examples
        --------
        >>> # Single wavelength band
        >>> obs.filter_data(wave=(3.2e-6, 3.8e-6))

        >>> # Two bands + quality cuts
        >>> obs.filter_data(
        ...     wave=[(1.5e-6, 1.8e-6), (3.2e-6, 3.9e-6)],
        ...     vis2_err=(0, 0.15),
        ...     cp_err=(0, 20),
        ... )

        >>> # Baseline range
        >>> obs.filter_data(baseline=(50, 150))
        """
        def _resolve(shorthand, ranges, min_val, max_val):
            """Normalise any filter form to a list of (lo, hi) tuples, or None."""
            if shorthand is not None:
                if isinstance(shorthand[0], (int, float)):
                    return [tuple(shorthand)]
                return [tuple(r) for r in shorthand]
            if ranges is not None:
                return list(ranges)
            if min_val is not None or max_val is not None:
                lo = min_val if min_val is not None else -np.inf
                hi = max_val if max_val is not None else np.inf
                return [(lo, hi)]
            return None

        wave_filter   = _resolve(wave,     wave_ranges,     min_wave,     max_wave)
        base_filter   = _resolve(baseline, baseline_ranges, min_baseline, max_baseline)
        freq_filter   = _resolve(freq,     freq_ranges,     min_freq,     max_freq)
        v2err_filter  = _resolve(vis2_err, vis2_err_ranges, min_vis2_err, max_vis2_err)
        cp_err_filter = _resolve(cp_err,   t3_err_ranges,   min_t3_err,   max_t3_err)

        data_dic = self.data

        def apply_range_filter(data_array, ranges):
            if ranges is None:
                return np.ones(len(data_array), dtype=bool)
            mask = np.zeros(len(data_array), dtype=bool)
            for lo, hi in ranges:
                mask |= (data_array >= lo) & (data_array <= hi)
            return mask

        # === FILTER V2 DATA ===
        v2_idx = np.ones(len(data_dic['VIS2']), dtype=bool)
        v2_idx &= apply_range_filter(data_dic['VIS2_waves'], wave_filter)
        v2_idx &= apply_range_filter(data_dic['B'],          base_filter)
        v2_idx &= apply_range_filter(data_dic['freqs'],      freq_filter)
        v2_idx &= apply_range_filter(data_dic['VIS2_err'],   v2err_filter)

        filtered_dic = {}
        v2_keys = ['VIS2', 'VIS2_err', 'VIS2_waves', 'Bu', 'Bv', 'B', 'freqs', 'u', 'v',
                   'VIS2_sta_idx_0', 'VIS2_sta_idx_1', 'VIS2_tel_name_0', 'VIS2_tel_name_1',
                   'INS_VIS2']
        for key in v2_keys:
            if key in data_dic:
                filtered_dic[key] = data_dic[key][v2_idx]

        # === FILTER T3 DATA (if present) ===
        if 'T3_PHI' in data_dic:
            t3_idx = np.ones(len(data_dic['T3_PHI']), dtype=bool)
            t3_idx &= apply_range_filter(data_dic['T3_waves'], wave_filter)
            t3_idx &= apply_range_filter(data_dic['max_base'], base_filter)
            if 'max_base' in data_dic and 'T3_waves' in data_dic:
                t3_freqs = data_dic['max_base'] / data_dic['T3_waves']
                t3_idx &= apply_range_filter(t3_freqs, freq_filter)
            if 'T3_PHI_err' in data_dic:
                t3_idx &= apply_range_filter(data_dic['T3_PHI_err'], cp_err_filter)

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
        """Restore ``self.data`` to its original state as loaded from disk.

        Undoes all calls to :meth:`filter_data` and :meth:`bin_spectral_channels`
        by replacing ``self.data`` with a fresh copy of ``self._raw_data``.

        Returns
        -------
        dict
            The restored data dictionary (same object as ``self.data``).

        Examples
        --------
        >>> obs.filter_data(wave=(3.2e-6, 3.6e-6))
        >>> obs.reset_data()   # back to full wavelength range
        """
        self.data = {k: v.copy() if isinstance(v, np.ndarray) else v
                     for k, v in self._raw_data.items()}
        print(f"Data reset to {len(self.data['VIS2'])} VIS2 points")
        return self.data

    def flag_v2(self, baselines=None, telescopes=None):
        """Remove specific baselines from the V\u00b2 data in-place.

        Only V\u00b2 data is affected; T3 data is left unchanged.
        Use :meth:`reset_data` to undo.

        Parameters
        ----------
        baselines : str, list of str, or list of (str, str), optional
            Baseline pair(s) to remove (order-insensitive). Accepted forms:

            * ``'AT1-AT2'`` (any non-alphanumeric separator works)
            * ``['AT1-AT2', 'AT3-AT4']``
            * ``[('AT1', 'AT2'), ('AT3', 'AT4')]``
        telescopes : str or list of str, optional
            Remove all V\u00b2 baselines that involve any of these telescopes.

        Returns
        -------
        dict
            The updated data dictionary (same object as ``self.data``).

        Examples
        --------
        >>> obs.flag_v2(baselines='AT1-AT2')
        >>> obs.flag_v2(baselines=['AT1-AT2', 'AT3-AT4'])
        >>> obs.flag_v2(telescopes='AT1')   # all baselines involving AT1
        """
        import re
        d = self.data

        excluded_pairs = set()
        if baselines is not None:
            if isinstance(baselines, str):
                baselines = [baselines]
            for b in baselines:
                if isinstance(b, str):
                    excluded_pairs.add(frozenset(re.split(r'[^A-Za-z0-9]+', b)))
                else:
                    excluded_pairs.add(frozenset(b))

        excluded_tels = set()
        if telescopes is not None:
            excluded_tels = {telescopes} if isinstance(telescopes, str) else set(telescopes)

        t0 = d['VIS2_tel_name_0']
        t1 = d['VIS2_tel_name_1']
        v2_keep = np.ones(len(d['VIS2']), dtype=bool)
        for pair_fs in excluded_pairs:
            pl = list(pair_fs)
            a, b = (pl[0], pl[1]) if len(pl) == 2 else (pl[0], pl[0])
            v2_keep &= ~(((t0 == a) & (t1 == b)) | ((t0 == b) & (t1 == a)))
        for tel in excluded_tels:
            v2_keep &= ~((t0 == tel) | (t1 == tel))

        v2_keys = ['VIS2', 'VIS2_err', 'VIS2_waves', 'Bu', 'Bv', 'B', 'freqs', 'u', 'v',
                   'VIS2_sta_idx_0', 'VIS2_sta_idx_1', 'VIS2_tel_name_0', 'VIS2_tel_name_1',
                   'INS_VIS2']
        for key in v2_keys:
            if key in d:
                self.data[key] = d[key][v2_keep]
        print(f"flag_v2: removed {(~v2_keep).sum()}/{len(v2_keep)} V\u00b2 points "
              f"({v2_keep.sum()} kept)")
        return self.data

    def flag_t3(self, triangles=None, telescopes=None):
        """Remove specific closure-phase triangles from the T3 data in-place.

        Only T3 data is affected; V\u00b2 data is left unchanged.
        Use :meth:`reset_data` to undo.

        Parameters
        ----------
        triangles : str, list of str, or list of (str, str, str), optional
            Triangle(s) to remove (order-insensitive). Accepted forms:

            * ``'AT1-AT2-AT3'`` (any non-alphanumeric separator works)
            * ``['AT1-AT2-AT3', 'AT2-AT3-AT4']``
            * ``[('AT1', 'AT2', 'AT3')]``
        telescopes : str or list of str, optional
            Remove all T3 triangles that involve any of these telescopes.

        Returns
        -------
        dict
            The updated data dictionary (same object as ``self.data``).

        Examples
        --------
        >>> obs.flag_t3(triangles='AT1-AT2-AT3')
        >>> obs.flag_t3(triangles=['AT1-AT2-AT3', 'AT2-AT3-AT4'])
        >>> obs.flag_t3(telescopes='AT1')   # all triangles involving AT1
        """
        import re
        d = self.data

        if 'T3_PHI' not in d:
            print("flag_t3: no T3 data present")
            return self.data

        excluded_trips = set()
        if triangles is not None:
            if isinstance(triangles, str):
                triangles = [triangles]
            for t in triangles:
                if isinstance(t, str):
                    excluded_trips.add(frozenset(re.split(r'[^A-Za-z0-9]+', t)))
                else:
                    excluded_trips.add(frozenset(t))

        excluded_tels = set()
        if telescopes is not None:
            excluded_tels = {telescopes} if isinstance(telescopes, str) else set(telescopes)

        t3_t0 = d['T3_tel_name_0']
        t3_t1 = d['T3_tel_name_1']
        t3_t2 = d['T3_tel_name_2']
        t3_keep = np.ones(len(d['T3_PHI']), dtype=bool)

        for trip_fs in excluded_trips:
            tl = list(trip_fs)
            if len(tl) == 3:
                a, b, c = tl
                # A point matches if its three telescope names equal the target set
                t3_keep &= ~(
                    ((t3_t0 == a) | (t3_t1 == a) | (t3_t2 == a)) &
                    ((t3_t0 == b) | (t3_t1 == b) | (t3_t2 == b)) &
                    ((t3_t0 == c) | (t3_t1 == c) | (t3_t2 == c))
                )
        for tel in excluded_tels:
            t3_keep &= ~((t3_t0 == tel) | (t3_t1 == tel) | (t3_t2 == tel))

        t3_keys = ['T3_PHI', 'T3_PHI_err', 'T3_waves', 'U1', 'V1', 'U2', 'V2', 'U3', 'V3',
                   'avg_base', 'max_base', 'INS_T3',
                   'T3_sta_idx_0', 'T3_sta_idx_1', 'T3_sta_idx_2',
                   'T3_tel_name_0', 'T3_tel_name_1', 'T3_tel_name_2']
        for key in t3_keys:
            if key in d:
                self.data[key] = d[key][t3_keep]
        print(f"flag_t3: removed {(~t3_keep).sum()}/{len(t3_keep)} T3 points "
              f"({t3_keep.sum()} kept)")
        return self.data

    def summary(self):
        """Print a concise summary of the currently loaded data.

        Displays the number of files, instrument(s), wavelength range,
        number of VIS2 and T3 points, baseline range, and median
        signal-to-noise ratios for V² and closure phase.

        Examples
        --------
        >>> obs.summary()
        ===============================================
          Observations Summary
        ===============================================
          Files          : 1
          Instrument(s)  : MATISSE
          Wavelengths    : 3.200 - 3.800 µm
          ...
        """
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
        """Print the effective angular resolution λ/B for every unique baseline.

        For each unique projected baseline length *B* present in the data,
        displays the minimum and maximum wavelength covered and the
        corresponding angular resolution λ/B in milli-arcseconds (mas).

        Notes
        -----
        The resolution is defined as λ/B, which corresponds to the fringe
        spacing of a two-element interferometer.  The full-width at half
        maximum of the central fringe envelope is ≈ 1.22 λ/B for a uniform
        aperture.

        Examples
        --------
        >>> obs.get_effective_resolution()
             B (m)   lam_min (um)   lam_max (um)   theta_min (mas)   theta_max (mas)
        ---------------------------------------------------------------------------
              46.6         3.200         3.800             14.18             16.84
        """
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

    def bin_spectral_channels(self, bin_size):
        """Bin consecutive spectral channels in-place, updating ``self.data``.

        For each telescope pair, wavelength-sorted channels are grouped into
        non-overlapping bins of *bin_size* channels.  Channels that do not form a
        complete bin at the end of a sequence are discarded.  V² is averaged;
        V² errors are combined in quadrature (divided by *bin_size*).  The same
        binning is applied to T3 data when present.

        Parameters
        ----------
        bin_size : int
            Number of spectral channels to combine into each output channel.
            For example, ``bin_size=5`` reduces 100 channels to 20.

        Returns
        -------
        dict
            The updated data dictionary (same object as ``self.data``).

        Examples
        --------
        >>> obs.bin_spectral_channels(bin_size=5)   # 5 channels → 1
        Binned 600 → 120 VIS2 points (5× channels/bin)
        """
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
            for b in range(len(idx) // bin_size):
                sl = idx[b*bin_size:(b+1)*bin_size]
                out['VIS2'].append(np.mean(d['VIS2'][sl]))
                out['VIS2_err'].append(np.sqrt(np.sum(d['VIS2_err'][sl]**2)) / bin_size)
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
                for b in range(len(idx) // bin_size):
                    sl = idx[b*bin_size:(b+1)*bin_size]
                    t3_out['T3_PHI'].append(np.mean(d['T3_PHI'][sl]))
                    t3_out['T3_PHI_err'].append(np.sqrt(np.sum(d['T3_PHI_err'][sl]**2)) / bin_size)
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
        print(f"Binned {n_orig} \u2192 {len(new_data['VIS2'])} VIS2 points ({bin_size}\u00d7 channels/bin)")
        return new_data

    def fit_uniform_disk(self, theta_init=1.0):
        """Fit a uniform-disk (UD) model to the V\u00b2 data.

        Uses a \u03c7\u00b2 minimisation (via ``scipy.optimize.curve_fit``) of the
        standard UD visibility model:

        .. math::

            V(f) = \\frac{2\\,J_1(\\pi\\,\\theta\\,f)}{\\pi\\,\\theta\\,f}

        where *f = B/\u03bb* is the spatial frequency in rad\u207b\u00b9 and \u03b8 is the
        angular diameter in radians.  Only data points with positive
        uncertainties are included in the fit.

        Parameters
        ----------
        theta_init : float, optional
            Initial guess for the angular diameter in mas.  Default is
            ``1.0`` mas.

        Returns
        -------
        dict
            A result dictionary with the following keys:

            * ``'theta_mas'`` \u2014 best-fit angular diameter in mas.
            * ``'theta_err_mas'`` \u2014 1-\u03c3 uncertainty in mas.
            * ``'chi2_red'`` \u2014 reduced \u03c7\u00b2 of the fit.
            * ``'model_vis2'`` \u2014 model V\u00b2 evaluated at every data-point
              spatial frequency; suitable for passing to :meth:`plot` as
              ``model_vis2``.

        Examples
        --------
        >>> result = obs.fit_uniform_disk(theta_init=1.0)
        Uniform disk fit: \u03b8 = 1.234 \u00b1 0.012 mas  (\u03c7\u00b2_red = 1.05)
        >>> fig = obs.plot(model_vis2=result['model_vis2'])
        """
        from scipy.special import j1
        from scipy.optimize import curve_fit

        d = self.data
        freqs = d['freqs']
        v2    = d['VIS2']
        v2_err = d['VIS2_err']

        mas_to_rad = np.pi / (180.0 * 3600.0 * 1000.0)

        def _ud_vis2(f, theta_mas):
            x = np.pi * theta_mas * mas_to_rad * f
            with np.errstate(invalid='ignore', divide='ignore'):
                vis = np.where(x == 0.0, 1.0, 2.0 * j1(x) / x)
            return vis ** 2

        valid = v2_err > 0
        popt, pcov = curve_fit(
            _ud_vis2, freqs[valid], v2[valid],
            p0=[theta_init], sigma=v2_err[valid], absolute_sigma=True,
            bounds=(0, np.inf),
        )
        theta_best = float(popt[0])
        theta_err  = float(np.sqrt(pcov[0, 0]))

        model_v2   = _ud_vis2(freqs, theta_best)
        residuals  = (v2[valid] - _ud_vis2(freqs[valid], theta_best)) / v2_err[valid]
        chi2_red   = float(np.sum(residuals ** 2) / max(valid.sum() - 1, 1))

        print(f"Uniform disk fit: \u03b8 = {theta_best:.3f} \u00b1 {theta_err:.3f} mas  "
              f"(\u03c7\u00b2_red = {chi2_red:.2f})")
        return {
            'theta_mas':     theta_best,
            'theta_err_mas': theta_err,
            'chi2_red':      chi2_red,
            'model_vis2':    model_v2,
        }

    def export_oifits(self, path):
        """Export the current (possibly filtered/binned) data to a minimal OIFITS 2 file.

        Writes one ``OI_WAVELENGTH`` and one ``OI_VIS2`` binary-table extension
        per instrument found in the data.  An ``OI_T3`` extension is written
        when closure-phase data are present.  Wavelength channels that were
        removed by :meth:`filter_data` are written back as flagged rows
        (``FLAG=True``) so that the output wavelength grid remains contiguous.

        Parameters
        ----------
        path : str
            Destination file path.  An existing file at this path will be
            overwritten.

        Notes
        -----
        * Target coordinates, telescope positions, and observation timestamps
          are not stored in the internal data model; they are written as
          placeholder zeros in the output file.
        * The ``ARRNAME`` keyword is hardcoded to ``'VLTI'``.

        Examples
        --------
        >>> obs.filter_data(wave_ranges=[(3.2e-6, 3.6e-6)])
        >>> obs.export_oifits('/data/star_filtered.fits')
        Exported OIFITS to /data/star_filtered.fits
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

    def plot_wavelength(self, v2_ylim=None, cp_ylim=None, show=True):
        """Plot V\u00b2 and closure phase as a function of wavelength.

        Each baseline (V\u00b2) and each maximum-baseline value (CP) is drawn as a
        separate line colour-coded by baseline length using the ``'viridis'``
        colourmap.  Shaded bands show \u00b11\u03c3 uncertainties.

        Parameters
        ----------
        v2_ylim : (float, float), optional
            Y-axis limits for the V\u00b2 panel.
        cp_ylim : (float, float), optional
            Y-axis limits for the closure-phase panel in degrees.
        show : bool, optional
            Call ``plt.show()`` at the end when ``True`` (default).

        Returns
        -------
        matplotlib.figure.Figure

        Examples
        --------
        >>> fig = obs.plot_wavelength(v2_ylim=(0, 1.2), cp_ylim=(-30, 30))
        >>> fig.savefig('vis_vs_wave.pdf')
        """
        d = self.data
        has_t3 = 'T3_PHI' in d

        n_rows = 2 if has_t3 else 1
        fig, axes = plt.subplots(n_rows, 1, figsize=(10, 4 * n_rows), squeeze=False)
        ax_v2 = axes[0, 0]

        B_vals = np.unique(d['B'])
        cmap   = plt.get_cmap('viridis')
        norm   = plt.Normalize(vmin=B_vals.min(), vmax=B_vals.max())

        for B_val in B_vals:
            mask  = d['B'] == B_val
            w     = d['VIS2_waves'][mask] * 1e6
            v2    = d['VIS2'][mask]
            v2err = d['VIS2_err'][mask]
            order = np.argsort(w)
            w, v2, v2err = w[order], v2[order], v2err[order]
            color = cmap(norm(B_val))
            ax_v2.plot(w, v2, color=color, linewidth=1.2)
            ax_v2.fill_between(w, v2 - v2err, v2 + v2err,
                               color=color, alpha=0.2, linewidth=0)

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        fig.colorbar(sm, ax=ax_v2, label='Baseline (m)')
        ax_v2.set_xlabel(r'$\lambda$ [$\mu$m]')
        ax_v2.set_ylabel(r'$V^2$')
        ax_v2.grid(visible=True, which='both')
        if v2_ylim is not None:
            ax_v2.set_ylim(*v2_ylim)

        if has_t3:
            ax_t3   = axes[1, 0]
            B_maxes = np.unique(d['max_base'])
            norm_t3 = plt.Normalize(vmin=B_maxes.min(), vmax=B_maxes.max())

            for B_val in B_maxes:
                mask  = d['max_base'] == B_val
                w     = d['T3_waves'][mask] * 1e6
                cp    = d['T3_PHI'][mask]
                cperr = d['T3_PHI_err'][mask]
                order = np.argsort(w)
                w, cp, cperr = w[order], cp[order], cperr[order]
                color = cmap(norm_t3(B_val))
                ax_t3.plot(w, cp, color=color, linewidth=1.2)
                ax_t3.fill_between(w, cp - cperr, cp + cperr,
                                   color=color, alpha=0.2, linewidth=0)

            sm_t3 = plt.cm.ScalarMappable(cmap=cmap, norm=norm_t3)
            sm_t3.set_array([])
            fig.colorbar(sm_t3, ax=ax_t3, label='Max baseline (m)')
            ax_t3.axhline(0, color='k', linewidth=0.5, linestyle='--')
            ax_t3.set_xlabel(r'$\lambda$ [$\mu$m]')
            ax_t3.set_ylabel('CP [deg]')
            ax_t3.grid(visible=True, which='both')
            if cp_ylim is not None:
                ax_t3.set_ylim(*cp_ylim)

        fig.tight_layout()
        if show:
            plt.show()
        else:
            plt.close(fig)
        return fig

    def plot(self, show_uv=True, model_vis2=None, model_t3=None, error_bars_v2=None, error_bars_t3=None,
             v2_ylim=None, cp_ylim=None, show=True, color_by='wavelength'):
        """Plot squared visibilities and closure phases as a function of spatial frequency.

        Produces a summary figure with three panels: UV coverage (left),
        V² vs spatial frequency (centre), and — when T3 data are present —
        closure phase vs B_max/λ (bottom right).  Points are colour-coded by
        wavelength using the ``'turbo'`` colourmap.

        Parameters
        ----------
        show_uv : bool, optional
            When ``True`` (default), include the UV-coverage panel on the
            left side of the figure.
        model_vis2 : array_like, optional
            Model V² values (same length as ``self.data['VIS2']``) to overlay
            as blue scatter points on the V² panel.
        model_t3 : array_like, optional
            Model closure-phase values (same length as ``self.data['T3_PHI']``)
            to overlay as blue scatter points on the CP panel.
        error_bars_v2 : array_like, optional
            Custom error bars for V².  If ``None``, uses ``self.data['VIS2_err']``.
        error_bars_t3 : array_like, optional
            Custom error bars for CP.  If ``None``, uses ``self.data['T3_PHI_err']``.
        v2_ylim : tuple of float, optional
            ``(ymin, ymax)`` limits for the V² panel.
        cp_ylim : tuple of float, optional
            ``(ymin, ymax)`` limits for the closure-phase panel in degrees.
        show : bool, optional
            Call ``plt.show()`` at the end when ``True`` (default).  Pass
            ``False`` to suppress display (e.g. when saving to file).
        color_by : {'wavelength', 'baseline'}, optional
            Colour scheme for data points.

            * ``'wavelength'`` (default) — points are coloured by wavelength
              using the ``'turbo'`` colourmap; a colour bar is shown.
            * ``'baseline'`` — each telescope pair (V²) and each triangle
              (CP) receives a distinct colour from ``'tab10'``; the UV plane
              uses matching colours, and a legend shows the telescope names.

        Returns
        -------
        matplotlib.figure.Figure
            The figure object, which can be saved with ``fig.savefig(...)``.

        Examples
        --------
        >>> fig = obs.plot(v2_ylim=(0, 1.2), cp_ylim=(-30, 30))
        >>> fig = obs.plot(color_by='baseline', v2_ylim=(0, 1.2))
        >>> fig.savefig('visibility.pdf')
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

        if show_uv:
            V2 = data_dic['VIS2']
            V2_err = np.array(data_dic['VIS2_err'])

            waves = data_dic['VIS2_waves']
            

            B_u = np.array(data_dic['Bu'])
            B_v = np.array(data_dic['Bv'])
            B = np.sqrt(B_u**2+B_v**2)

            fig = plt.figure(figsize=(30,figsize_y))

            gs1 = GridSpec(N_rows, 2, left=0.05, right=0.48, wspace=0.3, hspace=0.2, width_ratios=[1,2], height_ratios=height_ratios)
            ax1 = fig.add_subplot(gs1[0, 0])
            ax2 = fig.add_subplot(gs1[0, 1])

            if color_by == 'baseline':
                # ── Per-baseline / per-triangle colour mode ──────────────────────
                pair_labels = np.array([
                    '-'.join(sorted([a, b]))
                    for a, b in zip(data_dic['VIS2_tel_name_0'], data_dic['VIS2_tel_name_1'])
                ])
                unique_pairs = np.unique(pair_labels)
                n_pairs  = len(unique_pairs)
                cmap_bl  = plt.get_cmap('tab10')
                pair_color = {p: cmap_bl(i / max(n_pairs - 1, 1)) for i, p in enumerate(unique_pairs)}

                for pair, color in pair_color.items():
                    mask = pair_labels == pair
                    err  = (V2_err if error_bars_v2 is None else np.array(error_bars_v2))[mask]
                    ax2.scatter(B[mask] / waves[mask], V2[mask], color=color, s=10, label=pair, zorder=2)
                    ax2.errorbar(B[mask] / waves[mask], V2[mask], err,
                                 linestyle='', color=color, alpha=0.5, zorder=1)
                    ax1.scatter( B_u[mask] / waves[mask],  B_v[mask] / waves[mask],
                                color=color, s=8, alpha=0.8)
                    ax1.scatter(-B_u[mask] / waves[mask], -B_v[mask] / waves[mask],
                                color=color, s=8, alpha=0.8)

                ax2.legend(fontsize=8, title='Baseline', loc='upper right')
                if model_vis2 is not None:
                    ax2.scatter(B / waves, model_vis2, marker='*', color='k', s=20, zorder=3, label='model')

                if 'T3_PHI' in data_dic:
                    B_max      = data_dic['max_base']
                    t3_phi     = data_dic['T3_PHI']
                    t3_phi_err = data_dic['T3_PHI_err']
                    t3_waves   = data_dic['T3_waves']
                    trip_labels = np.array([
                        f"{a}-{b}-{c}"
                        for a, b, c in zip(data_dic['T3_tel_name_0'],
                                           data_dic['T3_tel_name_1'],
                                           data_dic['T3_tel_name_2'])
                    ])
                    unique_trips = np.unique(trip_labels)
                    n_trips = len(unique_trips)
                    trip_color = {t: cmap_bl(i / max(n_trips - 1, 1))
                                  for i, t in enumerate(unique_trips)}

                    ax3 = fig.add_subplot(gs1[1, 1])
                    for triplet, color in trip_color.items():
                        mask = trip_labels == triplet
                        err  = (t3_phi_err if error_bars_t3 is None
                                else np.array(error_bars_t3))[mask]
                        ax3.scatter(B_max[mask] / t3_waves[mask], t3_phi[mask],
                                    color=color, s=10, label=triplet, zorder=2)
                        ax3.errorbar(B_max[mask] / t3_waves[mask], t3_phi[mask], err,
                                     linestyle='', color=color, alpha=0.5, zorder=1)
                    ax3.legend(fontsize=7, title='Triangle', loc='upper right')
                    ax3.grid(visible=True, which='both', axis='both')
                    ax3.set_ylabel('CP [deg]')
                    ax3.set_xlabel(r'${B_{max}/\lambda}$ ${[rad^{-1}]}$')
                    if cp_ylim is not None:
                        ax3.set_ylim(cp_ylim[0], cp_ylim[1])
                    if model_t3 is not None:
                        ax3.scatter(B_max / t3_waves, model_t3,
                                    marker='*', color='k', s=20, zorder=3)

            else:
                # ── Wavelength colourmap mode (default) ─────────────────────────
                scatter_obs = ax2.scatter(B / waves, V2, c=waves, cmap='turbo', s=10)
                if error_bars_v2 is None:
                    ax2.errorbar(B / waves, V2, V2_err,
                                 linestyle='', c='lightgrey', alpha=0.5, zorder=0)
                else:
                    ax2.errorbar(B / waves, V2, error_bars_v2,
                                 linestyle='', c='lightgrey', alpha=0.5, zorder=0)

                divider = make_axes_locatable(ax2)
                cax = divider.append_axes('right', size='5%', pad=0.05)
                ax1.scatter( B_u / waves,  B_v / waves, c=waves, cmap='turbo')
                ax1.scatter(-B_u / waves, -B_v / waves, c=waves, cmap='turbo')
                fig.colorbar(scatter_obs, cax=cax, orientation='vertical')
                cax.set_title(r'$\lambda$ [m]')

                if model_vis2 is not None:
                    ax2.scatter(B / waves, model_vis2, c='blue', s=10)

                if 'T3_PHI' in data_dic:
                    B_max      = data_dic['max_base']
                    t3_phi     = data_dic['T3_PHI']
                    t3_phi_err = data_dic['T3_PHI_err']
                    t3_waves   = data_dic['T3_waves']

                    ax3 = fig.add_subplot(gs1[1, 1])
                    scatter_obs = ax3.scatter(B_max / t3_waves, t3_phi,
                                             c=t3_waves, cmap='turbo', s=10)
                    if error_bars_t3 is None:
                        ax3.errorbar(B_max / t3_waves, t3_phi, t3_phi_err,
                                     linestyle='', c='lightgrey', alpha=0.5, zorder=0)
                    else:
                        ax3.errorbar(B_max / t3_waves, t3_phi, error_bars_t3,
                                     linestyle='', c='lightgrey', alpha=0.5, zorder=0)
                    ax3.grid(visible=True, which='both', axis='both')
                    ax3.set_ylabel('CP [deg]')
                    ax3.set_xlabel(r'${B_{max}/\lambda}$ ${[rad^{-1}]}$')
                    if cp_ylim is not None:
                        ax3.set_ylim(cp_ylim[0], cp_ylim[1])
                    if model_t3 is not None:
                        ax3.scatter(B_max / t3_waves, model_t3, c='blue', s=10)

            # ── Shared axis formatting ─────────────────────────────────────────
            ax2.grid(visible=True, which='both', axis='both')
            ax2.set_ylabel(r'${V^2}$')
            ax2.set_xlabel(r'f ${[rad^{-1}]}$')
            if v2_ylim is not None:
                ax2.set_ylim(v2_ylim[0], v2_ylim[1])
            ax1.set_ylabel(r'v $[rad^{-1}]$')
            ax1.set_xlabel(r'u $[rad^{-1}]$')
            ax1.set_xlim(ax1.get_xlim()[::-1])
            ax1.grid(visible=True, which='major', axis='both')

            if show:
                plt.show()
            else:
                plt.close(fig)

            return fig
        else:
            print('The option you selected is not available yet :(')

    def plot_report_by_base(self, v2_min=0.6, v2_max=1.4, cp_min=-20, cp_max=20, show=True):
        """Plot V² and closure phase vs wavelength, one subplot per baseline / triangle.

        Produces a multi-panel diagnostic figure organised as follows:

        * **Left column** — UV-plane coloured by telescope pair.
        * **Centre column** — One V² vs λ subplot per unique telescope pair.
          Multiple baselines within the same pair are drawn as separate curves.
          Out-of-range points are indicated with small arrows at the panel edge.
        * **Right column** (T3 data only) — Closure phase vs λ, one subplot
          per unique telescope triplet.
        * **Bottom row** — SNR(V²) vs λ panels.

        Parameters
        ----------
        v2_min : float or None, optional
            Lower y-axis limit for V² panels, by default ``0.6``.  Pass
            ``None`` together with *v2_max=None* to auto-scale.
        v2_max : float or None, optional
            Upper y-axis limit for V² panels, by default ``1.4``.
        cp_min : float or None, optional
            Lower y-axis limit for closure-phase panels in degrees, by default
            ``-20``.  Pass ``None`` together with *cp_max=None* to auto-scale.
        cp_max : float or None, optional
            Upper y-axis limit for closure-phase panels in degrees, by default
            ``20``.
        show : bool, optional
            Call ``plt.show()`` at the end when ``True`` (default).  Pass
            ``False`` to suppress display (e.g. when saving to file).

        Returns
        -------
        matplotlib.figure.Figure
            The figure object.

        Examples
        --------
        >>> fig = obs.plot_report_by_base(v2_min=0, v2_max=1.2, cp_min=-30, cp_max=30)
        >>> fig.savefig('report.pdf', bbox_inches='tight')
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
            if v2_min is not None and v2_max is not None:
                V2_min_lim, V2_max_lim = v2_min, v2_max
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
                if cp_min is not None and cp_max is not None:
                    T3_min_lim, T3_max_lim = cp_min, cp_max
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


