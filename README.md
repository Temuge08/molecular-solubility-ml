# Molecular Solubility Prediction and De Novo Molecule Generation

An end-to-end molecular machine learning project combining **cheminformatics, property prediction, SMILES-based generative modeling, and multi-objective molecular screening**.

The project starts with aqueous solubility prediction using RDKit molecular representations and extends the workflow to de novo molecule generation using a PyTorch Variational Autoencoder (VAE).

---

## Project Overview

### Molecular Property Prediction

```text
ESOL dataset
    ↓
Data curation
    ↓
RDKit descriptors / Morgan fingerprints
    ↓
Chemical-space-aware splitting
    ↓
Random Forest regression
    ↓
Held-out solubility prediction
````

The final solubility model uses 10 interpretable RDKit physicochemical descriptors.

Held-out chemical-cluster test performance:

| Metric | Result |
| ------ | -----: |
| RMSE   |  0.853 |
| MAE    |  0.625 |
| R²     |  0.823 |

Descriptor-based modeling substantially outperformed Morgan fingerprints alone on this dataset.

---

### De Novo Molecular Generation

```text
MOSES molecules
    ↓
SMILES tokenization
    ↓
GRU Variational Autoencoder
    ↓
Latent-space sampling
    ↓
Novel SMILES generation
    ↓
RDKit validation
    ↓
Property prediction
    ↓
Multi-objective ranking
```

A SMILES-aware tokenizer and recurrent VAE were implemented in PyTorch using:

* token embeddings
* GRU encoder/decoder
* variational latent space
* KL annealing
* autoregressive generation

A subset of **100,000 MOSES molecules** was used for VAE development.

---

## Generated Molecule Evaluation

Generated molecules are evaluated for:

* validity
* uniqueness
* novelty
* physicochemical properties
* QED
* Lipinski compliance
* predicted aqueous solubility

The generated molecules showed similar broad property distributions to the MOSES training molecules.

For example:

| Property         | Training Mean | Generated Mean |
| ---------------- | ------------: | -------------: |
| Molecular Weight |        307.15 |         307.65 |
| LogP             |          2.43 |           2.47 |
| TPSA             |         65.92 |          65.66 |
| QED              |         0.806 |          0.798 |
| Predicted logS   |        -3.756 |         -3.792 |

---

## Multi-Objective Candidate Selection

Novel molecules are screened using multiple criteria rather than a single property.

The current pipeline combines:

```text
Drug-likeness (QED)
+
Predicted solubility
+
Lipinski properties
+
Applicability-domain filtering
+
Chemical diversity
```

Pareto analysis is used to preserve molecules representing different trade-offs between drug-likeness and predicted solubility.

Current results:

```text
Candidates after screening: 8,696
Pareto-optimal candidates: 35
Diverse candidates selected: 100
```

Morgan fingerprints and Tanimoto similarity are used to avoid selecting many nearly identical molecules.

---

## Repository Structure

```text
.
├── data/
├── figures/
├── models/
├── reports/
├── src/
│   ├── 01_audit_data.py
│   ├── ...
│   ├── 19_visualize_candidates.py
│   ├── smiles_tokenizer.py
│   └── smiles_vae.py
├── environment.yml
├── run_pipeline.sh
└── README.md
```

---

## Main Technologies

* Python
* PyTorch
* RDKit
* scikit-learn
* pandas / NumPy
* SciPy
* Matplotlib
* Conda
* Linux
* Git

---

## Installation

```bash
conda env create -f environment.yml
conda activate molml
```

The individual pipeline stages can then be run sequentially from `src/`.

---

## Example Workflow

```text
Molecular data
      ↓
Cheminformatics / curation
      ↓
Property prediction
      ↓
Generative modeling
      ↓
De novo molecules
      ↓
Virtual property screening
      ↓
Pareto ranking
      ↓
Diverse candidate set
```

---

## Current Limitations

This project is a computational molecular-design workflow rather than a complete drug-discovery platform.

Current candidate ranking does not yet include:

* target binding affinity
* toxicity prediction
* metabolism / broader ADMET endpoints
* molecular docking
* 3D protein-ligand interactions

Generated molecules should therefore be interpreted as **virtual candidates for further computational screening**, not validated drug leads.

---

## Planned Extensions

Future work may include:

* additional ADMET models
* toxicity prediction
* target-specific affinity prediction
* molecular docking
* property-guided generation
* graph or diffusion-based molecular generators
* reinforcement learning / GFlowNet optimization


ESOL:  Estimating Aqueous Solubility Directly from Molecular Structure 
John S. Delaney, J. Chem. Inf. Comput. Sci., 2004, 44, 1000 - 1005
https://pubs.acs.org/doi/abs/10.1021/ci034243x
