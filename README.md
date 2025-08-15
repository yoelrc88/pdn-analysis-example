# Touchstone Plot Example

This repository contains a small Jupyter notebook that demonstrates how to
visualise RF network parameters from a Touchstone `.s2p` file using the
[`scikit-rf`](https://scikit-rf.org) package.

The notebook produces three plots:

- **Return Loss** – magnitude of `S11` in dB
- **Insertion Loss** – magnitude of `S21` in dB
- **Input Impedance** – real and imaginary parts of `Z11`

An example `.s2p` file is included for quick testing.

## Requirements

- Python 3.10+
- `scikit-rf`
- `matplotlib`
- `jupyter`

Install dependencies with:

```bash
pip install scikit-rf matplotlib jupyter
```

## Usage

Launch the notebook and run the cells to generate the plots:

```bash
jupyter notebook plot_sparameters.ipynb
```

The generated PNG plots will be written to the `plots/` directory.

The last cell also exports the notebook to a regular Python script
(`plot_sparameters.py`), allowing you to rerun the analysis outside of
Jupyter if desired.
