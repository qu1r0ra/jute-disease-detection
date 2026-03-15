# jute-disease-detection <!-- omit from toc -->

![title](./assets/figures/dl/grad_cam.png)

<!-- Refer to <https://shields.io/badges> for usage -->

![Year, Term, Course](https://img.shields.io/badge/AY2526--T2-CSC713M-blue)
![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white) ![PyTorch](https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white) ![Lightning](https://img.shields.io/badge/Lightning-792ee5?logo=lightning&logoColor=white) ![Flask](https://img.shields.io/badge/Flask-3babc3?logo=flask&logoColor=white) ![Jupyter](https://img.shields.io/badge/Jupyter-f37626?logo=jupyter&logoColor=white)

An exploration of deep learning on merged jute leaf disease datasets. Created for CSC713M (Machine Learning for MSCS).

## Table of Contents <!-- omit from toc -->

- [1. Introduction](#1-introduction)
- [2. Project Structure](#2-project-structure)
- [3. Running the Project](#3-running-the-project)
  - [3.1. Prerequisites](#31-prerequisites)
  - [3.2. Reproducing the Results](#32-reproducing-the-results)
- [4. References](#4-references)

## 1. Introduction

To be written.

## 2. Project Structure

A high-level overview of the repository organization:

```text
.
├── artifacts/          # Generated checkpoints, models, logs, and project context
├── configs/            # Training configurations (.yaml) for Lightning CLI
├── docs/               # Technical documentation
│   ├── agents/         # AI agent-specific directives
│   └── ARCHITECTURE.md # Core technical design
├── notebooks/          # Jupyter notebooks for EDA and analysis
├── scripts/            # Automation scripts (batch training, grid search)
├── src/
│   ├── annotator/      # Image annotation tool (Flask)
│   └── jute_disease/   # Main package
│       ├── data/       # DataModules, Transforms, Datasets
│       ├── engines/    # Entry points (DL CLI, ML Training)
│       ├── models/     # Model architectures (e.g., MobileNetV2, RF, SVM)
│       └── utils/      # Logging, Seeding, Constants
└── tests/              # Structured test suite
    ├── annotator/      # Tests for the web app
    └── jute_disease/   # Tests for the core library
└── AGENTS.md           # AI assistant entry point
```

For a detailed look at the internal design, public APIs, and architectural decisions, see [ARCHITECTURE.md](docs/ARCHITECTURE.md).

## 3. Running the Project

### 3.1. Prerequisites

To reproduce our results, you will need the following installed:

1. **Git:** Used to clone this repository.

2. **Python:** We require Python `>=3.11` for this project. You do not need to install the specific version as it will be installed by `uv`.

3. **uv:** The package manager we used. Installation instructions can be found at <https://docs.astral.sh/uv/getting-started/installation/>.

### 3.2. Reproducing the Results

1. Clone this repository:

   ```bash
   git clone https://github.com/qu1r0ra/jute-disease-detection
   ```

2. Navigate to the project root and install all dependencies:

   ```bash
   cd jute-disease-detection
   uv sync
   ```

3. Run through the Jupyter notebooks in `notebooks/reproducibility/` in numerical order:
   1. `01_Exploratory_Data_Analysis.ipynb`
   2. `02_Model_Selection_Training_DL.ipynb`
   3. `02_Model_Selection_Training_ML.ipynb`
   4. ...

   _Notes_
   - When running a notebook, select `.venv` in root as the kernel.
   - Follow the instructions found in each notebook.

## 4. References

[1] Ansel, J., Yang, E., He, H., Gimelshein, N., Jain, A., Voznesensky, M., Bao, B., Bell, P., Berard, D., Burovski, E., Chauhan, G., Chourdia, A., Constable, W., Desmaison, A., DeVito, Z., Ellison, E., Feng, W., Gong, J., Gschwind, M., ... Chintala, S. (2024). PyTorch 2: Faster machine learning through dynamic Python bytecode transformation and graph compilation. In _Proceedings of the 29th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 2_ (pp. 929-947). <https://doi.org/10.1145/3620665.3640366>

[2] Biewald, L. (2020). _Experiment tracking with weights and biases_ [Computer software]. <https://www.wandb.com/>

[3] Bradski, G. (2000). The OpenCV Library. _Dr. Dobb's Journal of Software Tools_.

[4] Buslaev, A., Iglovikov, V. I., Khvedchenya, E., Parinov, A., Druzhinin, M., & Kalinin, A. A. (2020). Albumentations: Fast and flexible image augmentations. _Information, 11_(2), 125. <https://doi.org/10.3390/info11020125>

[5] Coenen, A., & Pearce, A. (2019). _Understanding UMAP_. Google PAIR. <https://pair-code.github.io/understanding-umap/>

[6] Falcon, W., & The PyTorch Lightning team. (2026). _PyTorch Lightning_ (Version 2.6.1) [Computer software]. Zenodo. <https://doi.org/10.5281/zenodo.18432694>

[7] Haque, R., Miah, M. M., Sultana, S., Fardin, H., Noman, A. A., Al-Sakib, A., Hasan, M. K., Rafy, A., Shihabur, R. M., & Rahman, S. (2024, September). Advancements in jute leaf disease detection: A comprehensive study utilizing machine learning and deep learning techniques. In _2024 IEEE International Conference on Power, Electrical, Electronics and Industrial Applications (PEEIACON)_ (pp. 248-253). IEEE. <https://doi.org/10.1109/PEEIACON63629.2024.10800378>

[8] Harris, C. R., Millman, K. J., van der Walt, S. J., Gommers, R., Virtanen, P., Cournapeau, D., Wieser, E., Taylor, J., Berg, S., Smith, N. J., Kern, R., Picus, M., Hoyer, S., van Kerkwijk, M. H., Brett, M., Haldane, A., Fernández del Río, J., Wiebe, M., Peterson, P., ... Oliphant, T. E. (2020). Array programming with NumPy. _Nature, 585_(7825), 357-362. <https://doi.org/10.1038/s41586-020-2649-2>

[9] Hunter, J. D. (2007). Matplotlib: A 2D graphics environment. _Computing in Science & Engineering, 9_(3), 90-95. <https://doi.org/10.1109/MCSE.2007.55>

[10] Islam, M. M., & Sheikh, M. R. (2026). A comprehensive image dataset of jute diseases. _Data in Brief, 64_, 112334. <https://doi.org/10.1016/j.dib.2025.112334>

[11] Islam, M. R., & Naimur Rahman, K. M. (2026). _Jute Leaf Disease Classification Dataset_ (Vol. 1) [Data set]. Mendeley Data. <https://doi.org/10.17632/5294gb7b5p.1>

[12] Jannat, M., Uddin, M. S., Hasan, M. A., Alam, M. S., Paul, A., Chowdhury, M. E. H., & Haider, J. (2025). Real-time jute leaf disease classification using an explainable lightweight CNN via a supervised and semi-supervised self-training approach. _Frontiers in Plant Science, 16_. <https://doi.org/10.3389/fpls.2025.1647177>

[13] Kokhlikyan, N., Miglani, V., Martin, M., Wang, E., Alsallakh, B., Reynolds, J., Melnikov, A., Kliushkina, N., Araya, C., Yan, S., & Reblitz-Richardson, O. (2020). _Captum: A unified and generic model interpretability library for PyTorch_. arXiv. <https://arxiv.org/abs/2009.07896>

[14] McInnes, L., Healy, J., Saul, N., & Grossberger, L. (2018). UMAP: Uniform Manifold Approximation and Projection. _Journal of Open Source Software, 3_(29), 861.

[15] Mridha, M. H. (2024). _Jute Plant Leaves_ (Vol. 1) [Data set]. Mendeley Data. <https://doi.org/10.17632/z87b9hnkh7.1>

[16] Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., & Duchesnay, E. (2011). Scikit-learn: Machine learning in Python. _Journal of Machine Learning Research, 12_, 2825-2830.

[17] The pandas development team. (2026). _pandas-dev/pandas: Pandas_ (Version v3.0.1) [Computer software]. Zenodo. <https://doi.org/10.5281/zenodo.18675244>

[18] van der Walt, S., Schönberger, J. L., Nunez-Iglesias, J., Boulogne, F., Warner, J. D., Yager, N., Gouillart, E., Yu, T., & the scikit-image contributors. (2014). scikit-image: image processing in Python. _PeerJ, 2_, e453. <https://doi.org/10.7717/peerj.453>

[19] Waskom, M. L. (2021). seaborn: statistical data visualization. _Journal of Open Source Software, 6_(60), 3021. <https://doi.org/10.21105/joss.03021>

[20] Wattenberg, M., Viégas, F., & Johnson, I. (2016). How to use t-SNE effectively. _Distill_. <https://doi.org/10.23915/distill.00002>

[21] Wightman, R. (2019). _PyTorch Image Models_ (Version 1.0.11) [Computer software]. Zenodo. <https://doi.org/10.5281/zenodo.4414861>
