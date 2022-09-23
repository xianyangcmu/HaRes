# Harmonization-guided cloud imputation (HaRes)
This repository is an official PyTorch implementation of the submitted paper **"Harmonization-guided deep residual network for imputing under clouds with multi-sensor satellite imagery"** from **WACV 2023**.
We provide scripts for reproducing all the results from our paper. All models can be trained from scratch.
## Dependencies
* Python 3.6
* PyTorch >= 1.0.0

## Quickstart (Demo)
Before running the code, a dataset needs to be created from public resources ([Landsat](https://landsat.gsfc.nasa.gov/satellites/landsat-8/) and [Sentinel-2](https://sentinel.esa.int/web/sentinel/missions/sentinel-2)).
To run the code, 
```bash
python train.py --config configs/[MODELNAME].yaml
'''
