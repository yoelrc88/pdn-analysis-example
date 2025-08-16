# Touchstone Plot Example

This repository contains Jupyter notebooks that demonstrate RF network analysis using the [`scikit-rf`](https://scikit-rf.org) package.

## Notebooks

### 1. Basic S-Parameter Plotting (`plot_sparameters.ipynb`)

A simple notebook that demonstrates how to visualize RF network parameters from Touchstone `.s2p` files.

The notebook produces three plots:
- **Return Loss** – magnitude of `S11` in dB
- **Insertion Loss** – magnitude of `S21` in dB
- **Input Impedance** – real and imaginary parts of `Z11`

### 2. PDN Impedance Analysis (`pdn_impedance_analysis_refactored.ipynb`)

A comprehensive notebook that demonstrates Power Delivery Network (PDN) impedance analysis with:
- Systematic analysis workflow from component modeling to final impedance calculation
- Y-space calculations with 1Ω normalization for numerical stability
- Synthetic capacitor models with realistic parasitics
- VRM modeling options (R-L, S1P, S2P)
- Multiple visualization approaches and engineering insights
- Automatic export to standalone Python script

An example `.s2p` file is included for quick testing.

## Requirements

- Python 3.10+
- `scikit-rf`
- `matplotlib`
- `jupyter`
- `numpy`

Install dependencies with:

```bash
pip install scikit-rf matplotlib jupyter numpy
```

## Usage

### Basic S-Parameter Plotting

Launch the basic plotting notebook:

```bash
jupyter notebook plot_sparameters.ipynb
```

### PDN Impedance Analysis

Launch the PDN analysis notebook:

```bash
jupyter notebook pdn_impedance_analysis_refactored.ipynb
```

The notebooks generate PNG plots in the `plots/` directory and export themselves to standalone Python scripts.

### Running Python Scripts

Both notebooks automatically export to Python scripts that can be run without Jupyter:

```bash
# Run basic S-parameter plotting
python plot_sparameters.py

# Run PDN impedance analysis  
python pdn_impedance_analysis_refactored.py
```

### Regenerating Python Scripts

To regenerate the Python scripts from the notebooks, run the last cell of each notebook or use:

```bash
jupyter nbconvert --to script plot_sparameters.ipynb
jupyter nbconvert --to script pdn_impedance_analysis_refactored.ipynb
```
