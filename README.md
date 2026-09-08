# Molecular Solubility Prediction with RDKit and Machine Learning

## Overview

This project implements an end-to-end cheminformatics and molecular
machine-learning pipeline for predicting aqueous solubility (logS)
from molecular structure.

The workflow includes:

- molecular data auditing and curation
- SMILES parsing and canonicalization with RDKit
- duplicate structure handling
- physicochemical descriptor generation
- Morgan fingerprint generation
- chemical-space-aware dataset splitting
- molecular representation comparison
- Random Forest hyperparameter optimization
- applicability-domain analysis
- feature interpretation
- held-out test evaluation


## Dataset

The project uses the Delaney ESOL dataset containing experimentally
measured aqueous solubility values for 1,128 small molecules.

Initial records: 1,128

After molecular curation: 1,116 unique usable structures.


## Data Curation

The raw dataset was audited for:

- missing structures
- missing/non-numeric solubility measurements
- invalid SMILES
- duplicate molecular structures
- disconnected molecular fragments
- formal charge
- molecular-size extremes
- target-value extremes

22 records represented 11 duplicate molecular structures.

One notable conflict involved mannitol and sorbitol, which were encoded
with identical non-stereochemical SMILES despite being distinct
stereoisomers. This structure was excluded because the available
representation could not distinguish the two compounds while their
measured solubilities differed substantially.


## Molecular Representations

Three representations were evaluated.

### Physicochemical descriptors

10 RDKit descriptors:

- Molecular Weight
- LogP
- TPSA
- Hydrogen-bond donors
- Hydrogen-bond acceptors
- Rotatable bonds
- Ring count
- Aromatic ring count
- Heavy atom count
- Fraction Csp3

### Morgan fingerprint

- radius = 2
- 2048 bits
- chirality enabled

### Combined representation

10 descriptors + 2048 Morgan fingerprint bits.


## Chemical-Space Splitting

A strict Bemis-Murcko scaffold split was initially investigated.

However:

- 315 molecules (28.2%) had no ring scaffold
- 253 molecules shared a simple benzene scaffold

This produced excessively large scaffold groups.

Therefore Morgan fingerprints and Tanimoto similarity were used with
Butina clustering to generate chemically related groups.

The final dataset contained:

- 422 chemical clusters
- largest cluster: 59 molecules
- 268 singleton clusters

Final split:

| Split | Molecules |
|------|-----------|
| Train | 892 |
| Validation | 111 |
| Test | 113 |

Clusters were kept entirely within individual splits to reduce
chemical similarity leakage.


## Representation Comparison

Random Forest regression was used with identical initial model
settings across representations.

Validation results:

| Representation | RMSE | MAE | R² |
|---|---:|---:|---:|
| Combined | 0.783 | 0.602 | 0.836 |
| Descriptors | 0.810 | 0.622 | 0.824 |
| Morgan | 1.566 | 1.223 | 0.343 |

After hyperparameter optimization:

| Representation | RMSE | MAE | R² |
|---|---:|---:|---:|
| Combined | 0.755 | 0.582 | 0.847 |
| Descriptors | 0.759 | 0.577 | 0.846 |

The performance difference between the combined and descriptor
representations was negligible.

The descriptor model was selected because it achieved comparable
performance using only 10 interpretable molecular features rather
than 2,058 features.


## Final Model

RandomForestRegressor

- n_estimators = 500
- max_depth = 20
- max_features = sqrt
- min_samples_leaf = 1
- min_samples_split = 2
- random_state = 42

The final model was retrained using the combined training and
validation sets (1,003 molecules).


## Held-Out Test Results

| Model | RMSE | MAE | R² |
|---|---:|---:|---:|
| Mean baseline | 2.218 | 1.813 | -0.198 |
| Descriptor Random Forest | 0.853 | 0.625 | 0.823 |

The final model substantially outperformed the constant mean baseline.


## Applicability-Domain Analysis

Maximum Morgan-Tanimoto similarity between each validation molecule
and the training set was evaluated.

No significant monotonic relationship between nearest-neighbor
similarity and prediction error was observed:

Descriptor model:

Spearman rho = -0.037
p = 0.698

Combined model:

Spearman rho = -0.084
p = 0.383

This indicates that nearest-neighbor fingerprint similarity alone was
not a reliable uncertainty indicator for this model and dataset.


## Pipeline

Raw ESOL
    ↓
Data audit
    ↓
Molecular curation
    ↓
RDKit descriptors / Morgan fingerprints
    ↓
Chemical clustering
    ↓
Train / validation / test split
    ↓
Representation comparison
    ↓
Hyperparameter optimization
    ↓
Model interpretation
    ↓
Held-out test evaluation


## Usage

Create the environment:

conda env create -f environment.yml

Activate:

conda activate molml

Run:

./run_pipeline.sh


## Limitations

- ESOL is a relatively small benchmark dataset.
- Solubility measurements can depend on experimental conditions such
  as pH, temperature, salt state, and assay protocol.
- Source SMILES do not always preserve complete stereochemical
  information.
- The current model uses fixed physicochemical descriptors rather than
  learned molecular representations.
- Model predictions should be treated as in-silico estimates and
  require experimental validation.


## Future Work

Planned extensions include:

- additional ADMET endpoints
- graph neural networks
- uncertainty estimation
- property-guided molecular generation
- VAE-based de novo molecular generation
- diffusion-based molecular generation