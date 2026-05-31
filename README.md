# vlti_loader

A Python package for loading, inspecting, filtering, binning, and exporting VLTI interferometric data from OIFITS files.

**Supported instruments:** GRAVITY (K-band ~2.0–2.4 µm), MATISSE (L/M ~3–5 µm, N ~8–13 µm), PIONIER (H-band ~1.5–1.8 µm).

---

## Installation

Clone the repository and install in editable mode:

```bash
git clone https://github.com/prioletp/vlti_loader.git
cd vlti_loader
pip install -e .
```

**Dependencies** (installed automatically): `numpy`, `astropy`, `matplotlib`.

---

## Quick start

```python
from vlti_loader import Observations

# Single file, directory of .fits files, or list of files from multiple instruments
obs = Observations("/path/to/data/")

obs.summary()                        # print a per-instrument overview
obs.get_effective_resolution()       # print λ/B resolution table (mas)

obs.filter_data(
    wave=[(1.5e-6, 1.8e-6), (2.0e-6, 2.4e-6), (3.2e-6, 3.9e-6)],
    vis2_err=(0, 0.15),
    cp_err=(0, 20),
)

fig = obs.plot(show_uv=True)                          # UV plane + V² + CP, coloured by λ
fig = obs.plot(color_by='baseline', v2_ylim=(0, 1.2)) # same, coloured by baseline with legend
fig = obs.plot_wavelength()          # V² and CP vs wavelength, coloured by baseline length

obs.flag_v2(baselines='AT1-AT2')        # remove a V² baseline by name (T3 unchanged)
obs.flag_t3(triangles='AT1-AT2-AT3')    # remove a T3 triangle by name (V² unchanged)

result = obs.fit_uniform_disk(theta_init=1.0)   # fit UD model; returns θ ± err + χ²_red
fig = obs.plot(model_vis2=result['model_vis2'])  # overlay model on the V² plot

obs.bin_spectral_channels(bin_size=5)  # combine channels, errors in quadrature
obs.reset_data()                       # restore original (no file re-read)

obs.export_oifits("output.fits")     # write OIFITS 2 file
```

All keys of `obs.data` are also accessible as attributes:

```python
obs.VIS2          # squared visibilities
obs.VIS2_waves    # wavelengths for each V² point
obs.B             # baseline lengths (m)
```

---

## Tutorial notebook

A step-by-step notebook covering all features is in [`examples/tutorial.ipynb`](examples/tutorial.ipynb).

---

## `Observations` API

### `Observations(path)`

Load interferometric data from OIFITS file(s).

| Argument | Type | Default | Description |
|---|---|---|---|
| `path` | str / Path / list / tuple | — | Path to a `.fits` file, a directory of `.fits` files, or a list of file paths (useful for combining instruments) |

---

### `summary()`

Print a concise overview: instrument(s), wavelength range, number of V² and T3 points, baseline range, and median SNR. No arguments.

---

### `get_effective_resolution()`

Print the angular resolution λ/B (in mas) for every unique projected baseline. No arguments.

---

### `filter_data(...)`

Filter the data in-place. Pass a `(min, max)` tuple, or a list of tuples for multiple disjoint windows. Filters are combined with AND across parameters; multiple windows within one parameter are combined with OR.

| Argument | Type | Default | Description |
|---|---|---|---|
| `wave` | (float, float) or list of tuples | `None` | Wavelength window(s) in metres — `(3.2e-6, 3.8e-6)` or `[(1.5e-6, 1.8e-6), (2.0e-6, 2.4e-6)]` |
| `baseline` | (float, float) or list of tuples | `None` | Projected baseline window(s) in metres |
| `freq` | (float, float) or list of tuples | `None` | Spatial frequency window(s) in rad⁻¹ (= B/λ) |
| `vis2_err` | (float, float) or list of tuples | `None` | V² uncertainty window(s) to keep |
| `cp_err` | (float, float) or list of tuples | `None` | Closure-phase uncertainty window(s) to keep in degrees |

> **Legacy aliases** (still supported): `wave_ranges`, `baseline_ranges`, `freq_ranges`, `vis2_err_ranges`, `t3_err_ranges` (list of tuples); and single-bound forms `min_wave`, `max_wave`, `min_baseline`, `max_baseline`, `min_freq`, `max_freq`, `min_vis2_err`, `max_vis2_err`, `min_t3_err`, `max_t3_err`.

```python
# Single band
obs.filter_data(wave=(3.2e-6, 3.8e-6))

# Two bands + quality cuts
obs.filter_data(
    wave=[(1.5e-6, 1.8e-6), (3.2e-6, 3.9e-6)],
    vis2_err=(0, 0.15),
    cp_err=(0, 20),
)
```

---

### `reset_data()`

Restore `self.data` to the original unfiltered state loaded from disk. No arguments.

---

### `flag_v2(baselines=None, telescopes=None)`

Remove specific baselines from the V² data in-place. T3 data is not affected. Use `reset_data()` to undo.

| Argument | Type | Default | Description |
|---|---|---|---|
| `baselines` | str, list of str, or list of (str, str) | `None` | Baseline pair(s) to remove — `'AT1-AT2'`, `['AT1-AT2', 'AT3-AT4']`, or `[('AT1','AT2')]`; order-insensitive |
| `telescopes` | str or list of str | `None` | Remove all V² baselines involving any of these telescopes |

```python
obs.flag_v2(baselines='AT1-AT2')              # single pair
obs.flag_v2(baselines=['AT1-AT2', 'AT3-AT4']) # multiple pairs
obs.flag_v2(telescopes='AT1')                 # all V² baselines involving AT1
```

---

### `flag_t3(triangles=None, telescopes=None)`

Remove specific closure-phase triangles from the T3 data in-place. V² data is not affected. Use `reset_data()` to undo.

| Argument | Type | Default | Description |
|---|---|---|---|
| `triangles` | str, list of str, or list of (str, str, str) | `None` | Triangle(s) to remove — `'AT1-AT2-AT3'`, `['AT1-AT2-AT3', 'AT2-AT3-AT4']`, or `[('AT1','AT2','AT3')]`; order-insensitive |
| `telescopes` | str or list of str | `None` | Remove all T3 triangles involving any of these telescopes |

```python
obs.flag_t3(triangles='AT1-AT2-AT3')               # single triangle
obs.flag_t3(triangles=['AT1-AT2-AT3', 'AT2-AT3-AT4']) # multiple triangles
obs.flag_t3(telescopes='AT1')                      # all T3 triangles involving AT1
```

---

### `bin_spectral_channels(bin_size)`

Average consecutive spectral channels in-place; errors are combined in quadrature.

| Argument | Type | Default | Description |
|---|---|---|---|
| `bin_size` | int | — | Number of channels to merge into one (e.g. `5` turns 100 channels into 20) |

---

### `fit_uniform_disk(theta_init=1.0)`

Fit the standard uniform-disk (UD) model $V(f) = 2J_1(\pi\theta f) / (\pi\theta f)$ to the V² data using χ² minimisation (`scipy.optimize.curve_fit`). Only points with positive uncertainties are included.

| Argument | Type | Default | Description |
|---|---|---|---|
| `theta_init` | float | `1.0` | Initial guess for the angular diameter in mas |

Returns a `dict` with keys `theta_mas`, `theta_err_mas`, `chi2_red`, and `model_vis2` (model V² at each data point — suitable for `plot(model_vis2=...)`)

```python
result = obs.fit_uniform_disk(theta_init=1.0)
print(result['theta_mas'], result['theta_err_mas'])  # best-fit ± 1σ in mas
fig = obs.plot(model_vis2=result['model_vis2'])
```

---

### `export_oifits(path)`

Write the current (possibly filtered/binned) data to a minimal OIFITS 2 file.

| Argument | Type | Default | Description |
|---|---|---|---|
| `path` | str | — | Output file path; existing files are overwritten |

---

### `plot_wavelength(v2_ylim=None, cp_ylim=None, show=True)`

Plot V² and closure phase as a function of wavelength. Each line corresponds to one baseline (or one max-baseline for CP), colour-coded by baseline length via the `viridis` colourmap.

| Argument | Type | Default | Description |
|---|---|---|---|
| `v2_ylim` | (float, float) | `None` | Y-axis limits for the V² panel |
| `cp_ylim` | (float, float) | `None` | Y-axis limits for the CP panel (deg) |
| `show` | bool | `True` | Call `plt.show()`; pass `False` to get the figure without displaying it |

---

### `plot(...)`

Overview plot: UV coverage panel (optional), V² vs spatial frequency, and closure phase vs B_max/λ.

| Argument | Type | Default | Description |
|---|---|---|---|
| `show_uv` | bool | `True` | Include the UV-coverage panel |
| `color_by` | `'wavelength'` or `'baseline'` | `'wavelength'` | `'wavelength'` colours points by λ with a turbo colourbar; `'baseline'` assigns a distinct tab10 colour to each telescope pair (V²) and each triangle (CP) and shows a legend with the telescope names |
| `model_vis2` | array_like | `None` | Model V² values to overlay as blue points |
| `model_t3` | array_like | `None` | Model closure-phase values to overlay as blue points |
| `error_bars_v2` | array_like | `None` | Custom V² error bars; defaults to `VIS2_err` |
| `error_bars_t3` | array_like | `None` | Custom CP error bars; defaults to `T3_PHI_err` |
| `v2_ylim` | (float, float) | `None` | Y-axis limits for the V² panel |
| `cp_ylim` | (float, float) | `None` | Y-axis limits for the closure-phase panel (deg) |
| `show` | bool | `True` | Call `plt.show()`; pass `False` to get the figure without displaying it |

```python
fig = obs.plot(show_uv=True, v2_ylim=(0, 1.2))               # wavelength colormap
fig = obs.plot(color_by='baseline', v2_ylim=(0, 1.2))        # per-baseline colours + legend
```

---

### `plot_report_by_base(...)`

Per-baseline / per-triangle diagnostic figure: one V² vs λ subplot per telescope pair, one CP vs λ subplot per triplet, UV-plane, and SNR panels.

| Argument | Type | Default | Description |
|---|---|---|---|
| `v2_min` | float or None | `0.6` | Lower y-axis limit for V² panels; `None` to auto-scale |
| `v2_max` | float or None | `1.4` | Upper y-axis limit for V² panels; `None` to auto-scale |
| `cp_min` | float or None | `-20` | Lower y-axis limit for closure-phase panels (deg); `None` to auto-scale |
| `cp_max` | float or None | `20` | Upper y-axis limit for closure-phase panels (deg); `None` to auto-scale |
| `show` | bool | `True` | Call `plt.show()`; pass `False` to get the figure without displaying it |

---

## Data dictionary keys

All data are stored as flat 1-D NumPy arrays in `obs.data`. Every point
corresponds to one (baseline, wavelength) or (triangle, wavelength) sample.

### Squared visibilities (V²)

| Key | Description |
|---|---|
| `VIS2` | Squared visibility |
| `VIS2_err` | Uncertainty on V² |
| `VIS2_waves` | Wavelength of each V² point (m) |
| `Bu` | Projected baseline — East component (m) |
| `Bv` | Projected baseline — North component (m) |
| `B` | Baseline length √(Bu²+Bv²) (m) |
| `freqs` | Spatial frequency B/λ (rad⁻¹) |
| `u` | Spatial frequency East component Bu/λ (rad⁻¹) |
| `v` | Spatial frequency North component Bv/λ (rad⁻¹) |
| `VIS2_sta_idx_0` | Station index of telescope 1 |
| `VIS2_sta_idx_1` | Station index of telescope 2 |
| `VIS2_tel_name_0` | Name of telescope 1 |
| `VIS2_tel_name_1` | Name of telescope 2 |
| `INS_VIS2` | Instrument name for each V² point |

### Closure phases (T3)

| Key | Description |
|---|---|
| `T3_PHI` | Closure phase (deg) |
| `T3_PHI_err` | Uncertainty on closure phase (deg) |
| `T3_waves` | Wavelength of each T3 point (m) |
| `U1`, `V1` | UV coordinates of baseline 1 (m) |
| `U2`, `V2` | UV coordinates of baseline 2 (m) |
| `U3`, `V3` | UV coordinates of baseline 3 (m) |
| `avg_base` | Average baseline of the triangle (m) |
| `max_base` | Maximum baseline of the triangle (m) |
| `T3_sta_idx_0/1/2` | Station indices of the three telescopes |
| `T3_tel_name_0/1/2` | Names of the three telescopes |
| `INS_T3` | Instrument name for each T3 point |

### Metadata

| Key | Description |
|---|---|
| `TEL_type` | Telescope type: `'UT'` (8.2 m) or `'AT'` (1.8 m) |
| `Telescopes` | Array of telescope names used in the observation |


