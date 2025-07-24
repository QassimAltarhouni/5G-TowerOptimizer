# 5G Tower Optimizer

The **5G Tower Optimizer** project contains utilities for visualising and optimising mobile tower deployments using genetic algorithms. Data is sourced from the OpenCellID database along with population rasters for generating realistic user distributions.

## Repository layout

- `data/` – raw OpenCellID tower dumps (`*.csv.gz`) and population raster files (`*.tif`).
- `src/` – Python modules used for loading data, generating users, computing fitness metrics and running the optimiser.
- `notebooks/` – Jupyter notebooks for exploratory analysis.
- `outputs/` – cleaned tower/user data, result CSVs and generated maps.

## Main components

- **Data loading** (`data_loader.py`)
- **User generation** (`simulator.py`, `real_user_generator.py`)
- **Fitness calculation** (`fitness_function.py`)
- **Genetic optimisation** (`genetic_optimizer.py`, `knowledge_utils.py`)
- **Experiment pipeline** (`prepare_all_instances.py`, `run_experiments.py`)
- **Visualisation** (`visualizer.py`)

## Running

A typical workflow is to use `prepare_all_instances.py` for a single dataset or `run_experiments.py` to evaluate several files. For example:

```bash
python src/prepare_all_instances.py
```

This will load tower data, generate user populations, compute baseline fitness and run the genetic algorithms. Results and plots are placed inside the `outputs/` directory.

## Dependencies

The code relies on common scientific Python packages such as `pandas`, `numpy`, `rasterio`, `scipy` and `folium`. Ensure they are installed in your environment before running the scripts.

## License

This project is provided as-is under the repository's default license.
