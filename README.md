# jute-disease-pest-detection <!-- omit from toc -->

![title](./readme/title.png)

<!-- Refer to https://shields.io/badges for usage -->

![Year, Term, Course](https://img.shields.io/badge/AY2526--T2-CSC713M-blue)
![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white) ![PyTorch](https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white) ![Lightning](https://img.shields.io/badge/Lightning-792ee5?logo=lightning&logoColor=white) ![Flask](https://img.shields.io/badge/Flask-3babc3?logo=flask&logoColor=white) ![Jupyter](https://img.shields.io/badge/Jupyter-f37626?logo=jupyter&logoColor=white)

An exploration of deep learning on combined jute disease and pest datasets. Created for CSC713M (Machine Learning for MSCS).

## Table of Contents <!-- omit from toc -->

- [1. Introduction](#1-introduction)
- [2. Running the Project](#2-running-the-project)
  - [2.1. Prerequisites](#21-prerequisites)
  - [2.2. Reproducing the Results](#22-reproducing-the-results)
- [3. References](#3-references)

## 1. Introduction

To be written.

## 2. Running the Project

### 2.1. Prerequisites

To reproduce our results, you will need the following installed:

1. **Git:** Used to clone this repository.

2. **Python:** We require Python `3.11.14` for this project. You do not need to install the specific version as it will be installed by `uv`.

3. **uv:** The package manager we used. Installation instructions can be found at <https://docs.astral.sh/uv/getting-started/installation/>.

### 2.2. Reproducing the Results

1. Clone this repository:

   ```bash
   git clone https://github.com/qu1r0ra/jute-disease-pest-detection
   ```

2. Navigate to the project root and install all required dependencies:

   ```bash
   uv sync
   ```

3. Run through the Jupyter notebooks in `notebooks/reproducibility/` in numerical order:
   1. `00_Setup.ipynb`
   2. `01_EDA.ipynb`
   3. ...

   Notes
   - When running a notebook, select `.venv` in root as the kernel.
   - More instructions can be found in each notebook.

## 3. References

[1] Md. M. Islam and Md. R. Sheikh, “A comprehensive image dataset of jute diseases,” _Data in Brief_, vol. 64, p. 112334, Feb. 2026. DOI: [10.1016/j.dib.2025.112334](https://doi.org/10.1016/j.dib.2025.112334).
