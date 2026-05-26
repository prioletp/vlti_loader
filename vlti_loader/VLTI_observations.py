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
                    data_dic_array.append(oi_utl.create_data_dic_MATISSE(hdul))
        
        self.data_dic_array = data_dic_array
        self.data = oi_utl.merge_dics(data_dic_array)    
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
        v2_idx = np.ones(len(data_dic['Vis2']), dtype=bool)
        
        # Apply wavelength filters to V2
        if wave_ranges is not None:
            v2_idx &= apply_range_filter(data_dic['wave_vis'], wave_ranges)
        else:
            # Legacy single range support
            if min_wave is not None:
                v2_idx &= (data_dic['wave_vis'] >= min_wave)
            if max_wave is not None:
                v2_idx &= (data_dic['wave_vis'] <= max_wave)
        
        # Apply baseline filters to V2
        if baseline_ranges is not None:
            v2_idx &= apply_range_filter(data_dic['Baselines'], baseline_ranges)
        else:
            # Legacy single range support
            if min_baseline is not None:
                v2_idx &= (data_dic['Baselines'] >= min_baseline)
            if max_baseline is not None:
                v2_idx &= (data_dic['Baselines'] <= max_baseline)
        
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
            v2_idx &= apply_range_filter(data_dic['Vis2_err'], vis2_err_ranges)
        else:
            # Legacy single range support
            if min_vis2_err is not None:
                v2_idx &= (data_dic['Vis2_err'] >= min_vis2_err)
            if max_vis2_err is not None:
                v2_idx &= (data_dic['Vis2_err'] <= max_vis2_err)
        
        # Create filtered dictionary for V2 data
        filtered_dic = {}
        v2_keys = ['Vis2', 'Vis2_err', 'wave_vis', 'u', 'v', 'Baselines', 'freqs', 
                   'Vis2_sta_idx_0', 'Vis2_sta_idx_1', 'Vis2_tel_name_0', 'Vis2_tel_name_1']
        
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
            V2 = data_dic['Vis2']
            V2_err = np.array(data_dic['Vis2_err'])

            waves = data_dic['wave_vis']
            

            B_u = np.array(data_dic['u'])
            B_v = np.array(data_dic['v'])
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
                print(B_max.shape, t3_waves.shape, t3_phi.shape)
                
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
        print('USING NEW VERSION OF THE FUNCTION report_by_base, CHECK IT OUT!')
        data_dic = self.data
        has_t3 = 'T3_PHI' in data_dic

        V2         = data_dic['Vis2']
        V2_err     = data_dic['Vis2_err']
        waves      = data_dic['wave_vis']
        baselines  = data_dic['Baselines']
        B_u        = np.array(data_dic['u'])
        B_v        = np.array(data_dic['v'])
        tel_name_0 = data_dic['Vis2_tel_name_0']
        tel_name_1 = data_dic['Vis2_tel_name_1']

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
    # V2_err = observations.data['Vis2_err'][observations.data['Vis2_err']<0]
    # plt.plot(np.arange(0,len(V2_err), 1), V2_err)
    # print(np.min(V2_err))
    # observations.plot(v2_ylim=(0, 1), cp_ylim=(-50, 50))
    #%%
    # for elem in observations.data_dic_array:
    #     # print(elem)
    #     print(elem.keys())
    data = observations.data
    v2 = data['Vis2']
    v2_err = data['Vis2_err']
    baselines = data['Baselines']
    waves = data['wave_vis']
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
