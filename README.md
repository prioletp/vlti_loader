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
from vlti_loader.VLTI_observations import Observations

# Single file, directory of .fits files, or list of files from multiple instruments
obs = Observations("/path/to/data/")

obs.summary()                        # print a per-instrument overview
obs.get_effective_resolution()       # print λ/B resolution table (mas)

obs.filter_data(
    wave_ranges=[(1.5e-6, 1.8e-6), (2.0e-6, 2.4e-6), (3.2e-6, 3.9e-6)],
    vis2_err_ranges=[(0, 0.15)],
    t3_err_ranges=[(0, 20)],
)

fig = obs.plot(uv_bool=True)         # UV plane + V² + closure phase

obs.bin_spectral_channels(n=5)       # combine channels, errors in quadrature
obs.reset_data()                     # restore original (no file re-read)

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

| Method | Description |
|---|---|
| `Observations(path)` | Load from file, directory, or list of files |
| `summary()` | Print per-instrument stats (N points, wavelength range, baseline range) |
| `get_effective_resolution()` | Print angular resolution λ/B per baseline (in mas) |
| `filter_data(...)` | Filter by wavelength, baseline, spatial frequency, or error |
| `reset_data()` | Restore data to the original loaded state |
| `bin_spectral_channels(n)` | Average every *n* spectral channels (errors in quadrature) |
| `export_oifits(path)` | Write current data to a minimal OIFITS 2 file |
| `plot(...)` | Overview plot: UV coverage, V² vs λ, closure phase vs λ |
| `plot_report_by_base(...)` | Per-baseline V² and closure phase panels |

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


