
This repository contains the code for reproducing results in the paper by Ryu & Lewis (*in prep*). 
It includes scripts for generating psycholinguistic metrics, running models, and creating the plots presented in the paper.

Note: Due to size limitations, the raw and annotated data from the naturalistic reading experiments cannot be uploaded here. However, they are available upon request.

## Psycholinguistic Phenomena

### Section 4: **Relative Clauses and Center Embeddings**

To generate data related to psycholinguistic phenomena (e.g., center embedding vs. right-branching sentences), run the following:

1. **Data Generation for Psycholinguistic Phenomena**
   - Run `psycholinguistic-phenomena/main.py`.

2. **Figures and Plots**
   - **Figure 5 & Figure 8**: Generate by running `psycholinguistic-phenomena/plot-generation.R`.
   - **Figure 6**: Generate by running `psycholinguistic-phenomena/relative-clause-plot.ipynb`.
   - **Figure 7**: Generate by running `psycholinguistic-phenomena/center-embedding-plot.ipynb`.

---

## Natural Readings

### Section 5: **Relative Clauses and Center Embeddings**

1. **GECO Eye-Tracking Corpus**
   - Run `natural-readings/geco_data_generation.py` to generate the GECO data.

2. **Natural Stories Corpus**
   - Run `natural-readings/ns_data_generation.py` to generate the Natural Stories data.

3. **Bayesian Models (Table 3-4 & Figure 10)**
   - Fit the models by running `natural-readings/regressions.R`.
   - **Note**: You’ll need the raw data and frequency data (`freq-dict.json`) to run these models. Raw data, frequency data, annotated data (results from geco_data_generation.py and ns_data_generation.py) and fitted models can be shared upon request.

---

## Data Requests

Due to the large size of the raw and annotated data from the experiments, these cannot be uploaded to the repository. Please contact us (soohyunr@umich.edu) if you would like to request access to the following:

- Raw data from naturalistic reading experiments
- Annotated data for naturalistic reading experiments
- Fitted models for naturalistic reading experiments
- Frequency data (`freq-dict.json` you see in the source code.)
