# Soil Phosphorus Estimation from Hyperspectral Spectra with GCNs and CNNs

This repository contains the code accompanying the manuscript:

> **"Investigating the Use of Graph Convolutional Networks on Soil Phosphorus Estimation from Hyperspectral Sensor Data"**  
> Metin et al., 2025.

The goal is to predict soil phosphorus concentration from hyperspectral reflectance spectra (200–900 nm).  
We compare a Graph Convolutional Network (GCN) with a 1D Convolutional Neural Network (CNN) baseline under a consistent preprocessing and evaluation pipeline:

- mutual information–based band selection (top 300 bands),
- Gaussian-noise data augmentation,
- per-sample Min–Max scaling,
- k-NN graph construction for the GCN,
- 3-fold cross-validation and hyperparameter grid search.

---

## Dataset

The experiments are based on the dataset:

> Rivadeneira-Bolaños, F., et al.  
> *Automated prediction of phosphorus concentration in soils using reflectance spectroscopy and machine learning algorithms*, MethodsX, 2024.

The dataset provides:

- 152 soil samples,  
- reflectance spectra from 200 to 900 nm (3501 bands),  
- laboratory phosphorus measurements.

In this repository, the dataset is provided in two Excel files:

- `data.xlsx` – spectral data (bands × samples),  
- `labels.xlsx` – phosphorus measurements for each sample.

These files are directly derived from the dataset published by Rivadeneira-Bolaños et al.  
They are included here **only for academic and reproducibility purposes**, and all rights remain with the original authors and publisher.


---

## Repository structure
```text
.
├── FosforGNN_BestMetrics&Plots.py
├── FosforGNN_GridSearch.py
├── FosforCNN_BestMetrics&Plots.py
├── FosforCNN_GridSearch.py
├── data.xlsx
├── labels.xlsx
├── plot_spectral_figures.py
├── requirements.txt
├── selected_band_indices.txt
└── statistics_of_data_and_pca.py
```


### GCN code

- FosforGNN_GridSearch.py  
  3-fold cross-validated grid search for the GCN model.  
  The script:

  - reads data.xlsx and labels.xlsx,  
  - selects the top 300 informative bands using mutual information regression,  
  - applies per-sample Min–Max scaling,  
  - performs Gaussian-noise augmentation of the training data,  
  - builds a k-NN graph for each fold using Euclidean distances in the 300-dimensional space,  
  - trains a 2-layer GCN with ReLU, dropout and a fully connected output layer,  
  - evaluates each hyperparameter combination with validation and test R²,  
  - logs the results to a text file (e.g. gnn_training_log.txt).

- FosforGNN_BestMetrics&Plots.py  
  Training script for the **best GNN configuration**.  
  It trains the final GNN model and generates prediction and residual plots,
  analogous to the CCN figures, for direct comparison.
  
### CNN code

- FosforCNN_GridSearch.py  
  Hyperparameter grid search for the 1D CNN baseline.  
  The script uses the **same 300 selected bands** and the same data augmentation and scaling scheme as the GCN:

  - builds a 1D CNN + fully connected regression head,  
  - runs 3-fold cross-validation over a grid of CNN hyperparameters  
    (e.g. number of channels, kernel size, dropout, learning rate),  
  - reports validation and test R² for each configuration.

- FosforCNN_BestMetrics&Plots.py  
  Training script for the **best CNN configuration**.  
  It trains the final CNN model and generates prediction and residual plots,
  analogous to the GCN figures, for direct comparison.

### Dataset figures and statistics

- plot_spectral_figures.py  
  Standalone script to reproduce the spectral figures used in the data description section.  
  It expects:

  - data.xlsx, labels.xlsx in the repository root,  
  - selected_band_indices.txt containing the indices of the 300 bands selected by mutual information.

  The script:

  - plots raw spectra for a subset of samples (spectra_raw.png),  
  - highlights the MIR-selected 300 bands on the raw spectrum of one sample  
    (spectra_raw_with_selected_bands_sample0.png),  
  - plots the same subset of spectra after per-sample Min–Max normalization  
    (spectra_normalized.png),  
  - plots example spectra for different phosphorus values, coloured and labeled by P content  
    (reflectance_vs_phosphorus.png).

  All plots are saved to the spectral_plots/ folder.

- statistics_of_data_and_pca.py  
  Script that computes and visualizes dataset-level statistics and exploratory plots:

  - descriptive statistics of the phosphorus labels, saved as  
    dataset_stats_plots/label_statistics.xlsx,  
  - label histogram and boxplot:  
    label_histogram.png, label_boxplot.png,  
  - per-band spectral statistics (mean, std, min, max) over all samples:  
    spectral_statistics_by_band.xlsx,  
  - example spectra for the first 10 samples and the mean spectrum with ±1 standard deviation:  
    example_spectra_first10.png, mean_spectrum_with_std.png,  
  - band–label Pearson correlation curve and corresponding table:  
    band_label_correlation_curve.png, band_label_correlation.xlsx,  
  - comparison of mean spectra for low- and high-phosphorus subsets:  
    low_vs_high_P_mean_spectra.png,  
  - PCA analysis of the spectra (after band selection), including:  
    pca_scree_plot.png and pca_pc1_pc2_scatter_Pcolormap.png.

  All outputs are saved into the dataset_stats_plots/ folder.

- selected_band_indices.txt  
  Text file containing the indices of the 300 most informative spectral bands selected by mutual information regression.  
  Used by plot_spectral_figures.py and (optionally) by other scripts.

---

## Installation

It is recommended to use a virtual environment:

python -m venv venv
source venv/bin/activate      # on Linux/macOS

venv\Scripts\activate       # on Windows

pip install -r requirements.txt

Note: Installing torch-geometric may require following the official installation instructions  
for your specific PyTorch and CUDA version.

---

## How to run the experiments

### 1. Prepare the data

The repository already contains:

- `data.xlsx` – spectral data with shape (bands × samples),  
- `labels.xlsx` – phosphorus measurements (one value per sample).

If you cloned the repository as-is, no extra data preparation is required.  


### 2. Reproducing the GCN baseline

python FosforGNN_GridSearch.py

This will:

- run 3-fold cross-validation over all GCN hyperparameter combinations,
- apply mutual information band selection, scaling and augmentation,
- construct k-NN graphs for each fold,
- log validation and test R² for each configuration.

python FosforGNN_BestMetrics&Plots.py

This script:

- trains the GCN with the selected best hyperparameters,
- saves loss curves and prediction/residual plots for validation and test data.

### 3. Reproducing the CNN baseline

python FosforCNN_GridSearch.py
python FosforCNN_BestMetrics&Plots.py

The first script performs the CNN hyperparameter grid search,  
the second trains the final CNN model and generates evaluation plots.

### 4. Dataset figures and statistics

To generate the dataset description figures used in the manuscript:

python plot_spectral_figures.py
python statistics_of_data_and_pca.py

All resulting plots and Excel tables will be saved into  
spectral_plots/ and dataset_stats_plots/.

---

## Reproducibility

All scripts:

- set a fixed random seed for random, numpy and torch (and CUDA when available),  
- use K-fold cross-validation with shuffle=True and a fixed random_state,  

to improve reproducibility of the reported results.

---

## Citation

If you use this code in your research, please cite the manuscript:

```bibtex
@article{metin2025soilP_GCN,
  author  = {Metin, Neslihan Pelin and Panahi, Nazila and Cimtay, Yucel},
  title   = {Investigating the Use of Graph Convolutional Networks on Soil Phosphorus Estimation from Hyperspectral Sensor Data},
  year    = {2025},
  journal = {to appear}
}
```

You may also want to cite the original dataset paper by Rivadeneira-Bolaños et al.

---

## Contact

For questions or issues about the code, please open a GitHub issue or contact:

- neslihanpelin.metin@mail.polimi.it
