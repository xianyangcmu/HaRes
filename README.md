# Harmonization-guided cloud imputation (HaRes)
This repository is an official PyTorch implementation of the submitted paper **"Harmonization-guided deep residual network for imputing under clouds with multi-sensor satellite imagery"** from **WACV 2023**.
We provide scripts for reproducing the pipeline from our paper. The benchmark multi-sensor dataset we used in this paper will be included once published. All models can be trained from scratch.
## Dependencies
* Python 3.6
* PyTorch 1.9.1
* Numpy 1.20.1

## Quickstart (Demo)
Before running the code, a dataset needs to be created from public resources ([Landsat](https://landsat.gsfc.nasa.gov/satellites/landsat-8/) and [Sentinel-2](https://sentinel.esa.int/web/sentinel/missions/sentinel-2)). Similar to ``dataset/ara_allBands_dataset.py``, a ``torch.util.data`` object needs to be created for loading 5 different numpy arrays including ``a Sentinel-2 image, a Landsat-8 image, a cloudy image, a binary cloud mask, and the ground-truth cloud-free image``.
To run the code, simply run
```bash
python train.py --config configs/[MODELNAME].yaml
```
