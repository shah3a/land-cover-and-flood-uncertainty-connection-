# Influence of Land Use and Land Cover Change on Urban Flooding
## A Deep Learning-Based Uncertainty Analysis (AAG 2025)

This repository provides a reproducible geospatial deep learning workflow for evaluating how land use and land cover (LULC) change influences urban flooding using a CNN-based susceptibility modeling framework, uncertainty evaluation, and scenario-based experiments.

## What this repo does
Pipeline:
1) Prepare and align raster predictors (LULC + terrain + hydrologic + ancillary layers)
2) Build a multiband raster stack aligned to a common grid
3) Generate training samples from labels (points or polygons) and extract per-sample feature vectors
4) Train a 1D-CNN flood susceptibility model
5) Evaluate with standard metrics and export results
6) Sensitivity analysis using leave-one-variable-out experiments
7) Scenario analysis by modifying LULC and re-running prediction
8) Predict full-area flood susceptibility maps and export GeoTIFF outputs

## Setup
Python 3.10+

Install:
pip install -r requirements.txt

## Configure
Edit:
configs/config.json

You must set:
- raster_paths: list of input rasters (GeoTIFF)
- label_points_csv or label_polygons_path
- target_column and label encoding
- output folders and model settings

## Run end-to-end
python -m src.run_pipeline

Outputs:
- data/processed/stack/stack.tif
- data/processed/samples/samples.parquet
- data/outputs/models/best_model.pt
- data/outputs/metrics/metrics.json
- data/outputs/maps/flood_susceptibility.tif
- data/outputs/metrics/sensitivity_leave_one_out.csv
- data/outputs/maps/scenarios/*.tif

## Data notes
This repo does not include raw datasets. Provide your own rasters and labels matching the study area and projection.
All rasters must be numeric. Categorical LULC should be encoded as integers or provided as one-hot layers.

## Citation
Abdullah, S., and Magliocca, N. (2025). Influence of land use and land cover change on urban flooding: A Deep Learning-Based Uncertainty Analysis. AAG Annual Meeting.

## License
MIT
