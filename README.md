# Applying Generative Adversarial Networks to Forecast High-Likelihood Fishing Grounds for Northeast Atlantic Mackerel

Repository with code for my masters thesis.

### Requirements
```bash
pip install -r requirements.txt
```

Does not contain the `xesmf` used to upsample bio data. That should be used with a conda environment and installed following [this](https://xesmf.readthedocs.io/en/stable/installation.html) tutorial.



## Repository Sections

1. **Analysis** – Data analysis and validation notebooks.
2. **Data-Exploration** – Scripts and notebook used for exploring data.
3. **DataFrames** – CSV files for correlation testing and a complete catch Data Frame with fused VMS and ERS catch locations from "test" area.
4. **Datasets** – Datasets used in the report, also datasets from previous areas and earlier iterations.
5. **GAN-model** – GAN implementation, training, and evaluation scripts.
6. **Prepare-Dataset** – Scripts and notebooks for downloading data from Copernicus marine, preprocessing and creation of datasets.


### Prepare-Dataset

**Scripts for dataset creation:**
1. **download_cfd.py** – Downloads Copernicus marine data for specified dates and locations from a CSV file.
2. **resample_bio_data** – Resamples and aligns biological data to match the required temporal and spatial resolution.
3. **transform_dataset.py** – Transforms and normalizes the downloaded data, applying statistical methods for further analysis.
4. **make_pictures.py** – Converts processed datasets into image formats suitable for machine learning models.


### GAN-model
- **train.py** – Training script with all hyperparameters 
- **models.py** – All models used: (Generator: baseline, *Generator_minimal_residual*: residual)
- **losses.py** – Loss functions used: all are logged to tensorboard.

### DataFrames 
[TEST_MAC_ERS_VMS_COMPLETE.csv](DataFrames/TEST_MAC_ERS_VMS_COMPLETE.csv) contains all catches form 2011 to 2024 for the test area (ERS and VMS)