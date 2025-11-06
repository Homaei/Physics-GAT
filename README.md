# 🔱 Physics-GAT: Graph Attention Networks with Physical Constraints for Anomaly Detection

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![PyG](https://img.shields.io/badge/PyTorch%20Geometric-green?style=flat&logo=pytorch&logoColor=white)](https://pyg.readthedocs.io/en/latest/)
[![ICT Express](https://img.shields.io/badge/Journal-ICT%20Express-blue.svg)](https://www.sciencedirect.com/journal/ict-express)

## 📖 Overview

This repository contains the official implementation of the paper:

> **"Graph Attention Networks with Physical Constraints for Anomaly Detection"**
> Authors: Mohammadhossein Homaei, Iman Khazrak, Ruben Molano, Andres Caro, Mar Avila
> Journal: *Preprint* (Arxiv)


![GAT Architecture](https://github.com/Homaei/Physics-GAT/blob/main/GAT.png?raw=true)


### The Core Idea

The Physics-GAT framework is proposed to enhance anomaly detection in **Water Distribution Systems (WDSs)** by combining two critical elements often ignored by purely data-driven models: **Network Topology** and **Hydraulic Physics**.

We introduce **Physics-Informed (PI) Features** derived from the normalized violations of the Mass and Energy conservation laws. These features, combined with a Graph Attention Network (GAT) for spatial reasoning and a Bidirectional LSTM (BiLSTM) for temporal analysis, lead to a highly accurate, robust, and explainable anomaly detector. 

### Key Contributions

* **Physics-Informed Features:** Using normalized conservation law residuals ($\phi_{\text{mass}}, \phi_{\text{energy}}$) as critical input features, significantly boosting detection accuracy (5.0\% F1 gain).
* **Spatio-Temporal Model:** A GAT-BiLSTM core that captures both network structure dependencies (GAT attention $\alpha_{ij}$) and temporal patterns.
* **Robustness:** Demonstrated stable performance under $\pm 15\%$ uncertainty in hydraulic parameters, a critical flaw of classical model-based methods.
* **Explainability:** Built-in mechanism using GAT attention coefficients and direct attribution of physics violations.

## ⚙️ Methodology Implementation Details

The model implementation strictly follows the methodology described in Section 3 of the paper.

### 1. PI Data Representation

The input features $\mathbf{h}_t(v_i)$ are composed of raw SCADA data (pressure, flow, level), statistical summaries, and the two PI features:

* **Mass Balance Violation ($\phi_{\text{mass}}$):** Calculated at each node $v_i$ using observed flows and estimated demand $D_i(t)$, normalized by total inflow (Equation 1).
* **Energy Gradient Violation ($\phi_{\text{energy}}$):** Calculated across each pipe $(i, j)$ using observed heads ($p_i + z_i$) and theoretical head loss $h_L(Q_{ij})$, normalized by maximum head (Equation 2).
* **Unmeasured Nodes:** Feature vectors for nodes without sensors are approximated using pressure-driven inverse distance weighting interpolation among measured neighbors.

### 2. Spatio-Temporal Core

* **GAT:** 3 layers ($L=3$) with 8 attention heads ($K=8$) and 128 hidden units. Uses network topology to learn adaptive attention $\alpha_{ij}$.
* **BiLSTM:** Processes the GAT output sequence over a 24-hour input window ($w=24$) to capture bidirectional temporal dependencies.

### 3. Multi-Scale Detection

The final anomaly score $a_t^{\text{final}}(v_i)$ is an adaptive fusion of three scales:
$$
a_t^{\text{final}}(v_i) = \lambda_1 a_t^{\text{micro}}(v_i) + \lambda_2 a_t^{\text{meso}}(\mathcal{C}(i)) + \lambda_3 a_t^{\text{macro}}
$$
Where $\lambda_k$ are learned softmax weights, and $\mathcal{C}(i)$ is the hydraulic cluster (obtained via Louvain algorithm).

### 4. Training Objective

The model is optimized using a composite loss function (Equation 4):
$$
\mathcal{L} = \mathcal{L}_{\text{BCE}} + \lambda_p \mathcal{L}_{\text{physics}} + \lambda_c \mathcal{L}_{\text{consist}}
$$
* **$\mathcal{L}_{\text{BCE}}$:** Binary Cross-Entropy (Primary detection loss).
* **$\mathcal{L}_{\text{physics}}$:** Regularizes against high physical violations during normal, non-anomalous operation ($\lambda_p=0.1$).
* **$\mathcal{L}_{\text{consist}}$:** Enforces spatio-temporal coherence between adjacent nodes ($\lambda_c=0.05$).

## 💾 Datasets

This project uses the following Water Distribution System (WDS) benchmarks:

| Dataset | Type | Nodes ($N$) | Purpose |
| :--- | :--- | :--- | :--- |
| **BATADAL (C-Town)** | Controlled | 128 | Primary benchmark and training set. |
| **D-Town** | Real-World | $\approx 200$ | Generalization test. |
| **L-Town** | Large-Scale | $> 300$ | Scalability and complexity test. |
| **Modena** | Realism | $\approx 180$ | Real-world scenario test. |

### Data Download

The raw SCADA data, network topology (EPANET .inp files), and attack labels for C-Town are sourced from the official BATADAL website / related publications.

To replicate the study, please download and organize the data in the following structure:

data/
├── c-town/
│ ├── C-town.inp
│ ├── scada_training.csv
│ └── attack_labels.csv
├── d-town/
├── l-town/
└── modena/


## 🚀 Getting Started

### Prerequisites

* Python 3.8+
* PyTorch and PyTorch Geometric
* EPANET (for hydraulic analysis, highly recommended for feature calculation)

### Installation

Clone the repository and install dependencies:

```bash
git clone [https://github.com/Homaei/Physics-GAT.git](https://github.com/Homaei/Physics-GAT.git)
cd Physics-GAT
```
# Install core libraries and PyTorch Geometric dependencies (using requirements.txt)
```bash
pip install -r requirements.txt
```

Running the ModelPreprocessing: 

1. Run the feature generation script to calculate the PI features ($\phi_{\text{mass}}, \phi_{\text{energy}}$) and interpolate unmeasured nodes.
```bash
python src/1_preprocess_data.py --network c-town
```
# This generates preprocessed_features.pt in the data/ folder.

2. Training: Start the training process on the C-Town dataset.
```bash
python src/2_train_model.py --network c-town --epochs 50 --lr 0.001
```
3. Evaluation: Test the trained model on the BATADAL test set.
```bash
python src/3_evaluate_model.py --network c-town --checkpoint checkpoints/best_model.pt
```

Replicating Multi-Network Validation
To replicate the Zero-Shot and Fine-Tuned transfer learning experiments (Section 4.2.2), use the dedicated transfer script:
# Zero-Shot Transfer (Train on C-Town, Test on D-Town)
```bash
python src/4_transfer_learning.py --source c-town --target d-town --mode zero-shot
```
# Fine-Tuned Transfer (Train on C-Town, Fine-tune on 10% D-Town)
```bash
python src/4_transfer_learning.py --source c-town --target d-town --mode fine-tune --finetune_ratio 0.1
```
📊 Results and Benchmarks
The full results are detailed in the paper (Table 1 and 2).
| Method                  |  F1-score [95% CI] | TTD (hours) |
| :---------------------- | :----------------: | :---------: |
| Physics-GAT (Full)      | 0.979 [.971, .986] |     1.44    |
| B1 (Model-Based Winner) | 0.946 [.932, .958] |     1.61    |
| GCN + BiLSTM            | 0.941 [.931, .951] |     1.69    |


Ablation Study (Key Finding):

| Model Variant            | F1-score |   Δ   |
| :----------------------- | :------: | :---: |
| Full Model               |   0.979  |   —   |
| w/o ϕ (Physics Features) |   0.930  | −5.0% |


Explainability and Visualization
The model provides two interpretability outputs for operator decision support:

GAT Attention Heatmaps: Visualize learned spatial attention (αᵢⱼ).
A strong correlation (ρ = 0.81) between attention flow and hydraulic paths confirms that the model captures physically consistent relationships.

Physics Violation Attribution: During anomaly detection, the model attributes alarms to Mass Violation or Energy Violation, providing immediate diagnostic insight.

Scripts for generating heatmaps and temporal profiles (Figures in Appendix) are available in:
notebooks/visualization.ipynb

Contribution and Citation
If you find this work useful, please cite:

@article{Homaei2024PhysicsGAT,
title={Graph Attention Networks with Physical Constraints for Anomaly Detection},
author={Homaei, Mohammadhossein and Khazrak, Iman and Molano, Ruben and Caro, Andres and Avila, Mar},
journal={preprint},
year={2024},
publisher={arXiv}
}

Contact
Hubert Homaei
Email: Homaei@ieee.org
