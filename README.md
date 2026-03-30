<div align="center">

# 🏥 EBRT-DL: Deep Learning Surrogate for External Beam Radiation Therapy

**End-to-end pipeline: Monte Carlo simulation → 3D TransUNet + FiLM surrogate → Treatment optimization**

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org)
[![GEANT4](https://img.shields.io/badge/GEANT4-OpenGATE-green.svg)](https://opengate.readthedocs.io)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

[📄 Monte Carlo Notebook on Kaggle](https://www.kaggle.com/code/ibongarciagomez/ebrt-montecarlo-simulation) · [🧠 Neural Network Notebook on Kaggle](https://www.kaggle.com/code/ibongarciagomez/red-neuronal-ebrt)

</div>

---

## 📋 Abstract

External Beam Radiation Therapy (EBRT) treatment planning requires accurate dose calculations that currently take **3–5 minutes per plan** using clinical algorithms (e.g., Varian AcurosXB). This project replaces that bottleneck with a deep learning surrogate that predicts 3D dose distributions in **< 50 ms** — a **>1,000× speedup** — while maintaining clinical accuracy.

The pipeline is fully self-contained:

1. **Monte Carlo Ground Truth** — Full GEANT4-based simulation of a Varian TrueBeam STx with HD 120 MLC
2. **3D TransUNet + FiLM** — A physics-conditioned neural network trained on 9,146 MC simulations
3. **Differentiable Treatment Optimizer** — L-BFGS inverse planning through the surrogate using ICRU-83 clinical objectives

> **Clinical context**: Designed for the Varian TrueBeam SVC 3.0 linear accelerator (6/10/15 MV, FFF mode), which delivers > 60% of EBRT treatments worldwide.

---

## 🎯 Key Results

| Metric | Value | Clinical Threshold | Status |
|--------|-------|-------------------|--------|
| **DoseAcc 3%** | **73.0%** | — | ✅ Dose zone accuracy |
| **DoseAcc 5%** | **87.0%** | — | ✅ Dose zone accuracy |
| **D_max error** | **2.97%** | < 5% | ✅ Within tolerance |
| **Inference time** | **< 50 ms** | — | ✅ Real-time capable |
| **Speedup vs AcurosXB** | **> 1,000×** | — | ✅ |
| **Training dataset** | **9,146 samples** | — | 3 patient anatomies |

> **DoseAcc 3%**: Percentage of voxels in the dose zone (> 10% D_max) where the absolute error is < 3% of D_max. DoseAcc 5% uses a 5% threshold. Both metrics exclude background (~99.5% of the volume) to focus on clinically relevant regions.

---

## 🏗️ Architecture

### Stage 1: 3D TransUNet + FiLM Surrogate

```
                    ┌─────────────────────────────────────────────────┐
                    │            10 Beam Parameters                   │
                    │  (energy, field size, SSD, gantry angle, ...)   │
                    └────────────────────┬────────────────────────────┘
                                         │
                                    ┌────▼────┐
                                    │  FiLM   │  MLP → (γ, β) per level
                                    │Generator│
                                    └────┬────┘
                                         │ γ₁,β₁  γ₂,β₂  γ₃,β₃  γ₄,β₄
                                         │   │      │      │      │
    ┌──────────┐    ENCODER              │   │      │      │  DECODER
    │ Density  │                         │   │      │      │
    │ Volume   │──►[Conv32]──►[Conv64]──►[Conv128]──►[Conv256]
    │(48×64×64)│      │          │          │          │
    └──────────┘      │          │          │     ┌────▼─────┐
                      │          │          │     │Transformer│ 4 layers
                      │          │          │     │Bottleneck │ 8 heads
                      │          │          │     └────┬──────┘
                      │          │          │          │
                      │          │     skip ├────►[Up+FiLM₃]──►[Up+FiLM₂]──►[Up+FiLM₁]──►[Up+FiLM₀]
                      │          │──────────┘                                                  │
                      │──────────┘                                                        ┌───▼───┐
                      └───────────────────────────────────────────────────────────────────►│Sigmoid│
                                                                                          └───┬───┘
                                                                                              │
                                                                                    ┌─────────▼─────────┐
                                                                                    │   Predicted Dose   │
                                                                                    │    (48×64×64)      │
                                                                                    └────────────────────┘
```

**Key design choices:**
- **FiLM conditioning** (Feature-wise Linear Modulation): Beam parameters modulate decoder features via learned affine transforms γ·x + β, enabling a single model to predict dose for any beam configuration
- **3D Transformer bottleneck**: Captures long-range spatial dependencies critical for dose transport physics (buildup, scatter, penumbra)
- **InstanceNorm3D**: More stable than BatchNorm for medical imaging with small batch sizes
- **Deep supervision**: Auxiliary losses at 3 decoder levels for stable gradient flow
- **Physics-informed loss**: 7-component loss function (weighted MSE + gradient + SSIM + DVH + dose fall-off + cosine similarity + L1)

### Stage 2: Differentiable Treatment Optimizer

```
Clinical Objectives (ICRU-83)          Surrogate Model
         │                                    │
         ▼                                    ▼
  ┌──────────────┐    ┌──────────────┐   ┌─────────┐
  │ D95, D98,    │◄───│  L-BFGS      │──►│TransUNet│──► 3D Dose
  │ V95, HI,     │    │  Optimizer   │   │ + FiLM  │
  │ Dmax_OAR,    │    │ (6 branches) │   └─────────┘
  │ Dmean_OAR    │    └──────────────┘
  └──────────────┘          │
                     Sigmoid reparametrization
                     p = p_min + (p_max - p_min) · σ(raw)
```

The optimizer explores **6 branches** (3 energies × 2 filter modes) with **3 multi-start restarts** each, evaluating 18 candidate plans in seconds. Post-optimization uncertainty is quantified via **MC Dropout** (20 stochastic forward passes).

---

## 🔬 Monte Carlo Simulation

The training data is generated using a **full GEANT4 simulation** of the Varian TrueBeam treatment head:

### Simulation Chain

```
Point Source (SAD = 100 cm)
    │  Bremsstrahlung spectrum (from commissioning data)
    ▼
┌─────────────────┐
│   Jaws Y (W)    │  7.8 cm tungsten
└────────┬────────┘
         ▼
┌─────────────────┐
│   Jaws X (W)    │  7.8 cm tungsten
└────────┬────────┘
         ▼
┌─────────────────┐
│  HD 120 MLC     │  120 individual tungsten leaves
│  (60 pairs)     │  32 × 2.5 mm (central) + 28 × 5.0 mm (outer)
└────────┬────────┘
         ▼
┌─────────────────┐
│ Patient Phantom │  40 × 40 × 40 cm
│ (water + struct)│  Bone, tumor, OAR, lung (anatomy-dependent)
└────────┬────────┘
         ▼
    200³ dose grid (2 mm spacing)
```

### Physics
- **Framework**: GEANT4 (via OpenGATE Python API)
- **Physics list**: QGSP_BIC_EMZ (electromagnetic precision, production cuts 0.1 mm)
- **Interactions**: Photoelectric, Compton (Klein-Nishina), pair production, bremsstrahlung
- **Statistics**: 1M primary photons per simulation (~30 min/sim on 4 CPU cores)
- **Beam energies**: 6, 10, 15 MV (with and without flattening filter)

### Patient Anatomies

| Type | Tumor location | Structures | Challenge |
|------|---------------|------------|-----------|
| **Pelvis** | Deep (6–10 cm) | Bone slab + OAR | Bone heterogeneity |
| **Head & Neck** | Superficial (2–5 cm) | Bone slab + OAR | Small target, critical OAR |
| **Lung** | Surrounded by air | Bone + lung box + OAR | Extreme density contrast |

### Generated Dataset

- **9,146 unique simulations** across 70+ Kaggle sessions
- **Stratified sampling**: Clinical correlations between energy, field size, and patient type
- **10 normalized beam parameters**: energy, field size, SSD, gantry, tumor depth/radius, OAR distance/radius, bone thickness, patient type

---

## 📊 Training Details

### Loss Function (7 Components)

| Component | Weight | Purpose |
|-----------|--------|---------|
| Weighted MSE | 1.0 | Dose-weighted pixel error |
| 3D Gradient | 0.5 | Penumbra and buildup accuracy |
| 3D SSIM | 0.2 | Structural similarity |
| DVH loss | 0.3 | Dose-volume histogram consistency |
| Dose fall-off | 0.2 | Boundary accuracy at target edge |
| Cosine similarity | 0.15 | Global directional agreement |
| L1 (MAE) | 0.4 | Robust global error |

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW (lr=1.5e-4, wd=5e-5) |
| Scheduler | OneCycleLR (200 ep/cycle, warm restarts, decay=0.7) |
| Batch size | 8 |
| Mixed precision | FP16 (AMP) |
| Weight averaging | SWA (from epoch 400, every 5 epochs) |
| Early stopping | Dual criterion: val_loss + DoseAcc3% |
| Data augmentation | 12-step pipeline (geometric + intensity + coarse dropout) |
| Gradient clipping | max_norm = 1.0 |
| Hardware | NVIDIA P100 16GB (Kaggle) |

### Augmentation Pipeline

| Step | Transform | Parameters | Probability |
|------|-----------|-----------|-------------|
| 1 | Gaussian noise | σ=0.01 | 0.30 |
| 2 | Intensity scaling | ×0.97–1.03 | 0.30 |
| 3 | Gaussian blur | σ=0.5–1.5 | 0.15 |
| 4 | Elastic deformation | α=2, σ=15 | 0.15 |
| 5 | Contrast adjustment | 0.9–1.1 | 0.15 |
| 6 | Gamma correction | 0.8–1.2 | 0.15 |
| 7 | Motion artifacts | σ=0.3, shift≤2 | 0.15 |
| 8 | Local intensity variation | — | 0.15 |
| 9 | Sharpening | α=0.3 | 0.15 |
| 10 | Brightness shift | ±0.02 | 0.15 |
| 11 | 3D Coarse dropout | 4–8 voxel cubes | 0.05 |
| 12 | Random flips | Disabled | — |

---

## 🗂️ Repository Structure

```
├── README.md                          # This file
├── LICENSE                            # Apache 2.0
├── requirements.txt                   # Python dependencies
│
├── ebrt_montecarlo_simulation.ipynb   # Stage 0: MC data generation (GEANT4/OpenGATE)
├── consolidate_datasets_EBRT.ipynb    # Dataset consolidation across sessions
├── neural_network_EBRT.ipynb          # Stage 1+2: Surrogate training + optimizer
│
└── docs/
    └── references.md                  # Key papers and references
```

### Notebook Descriptions

| # | Notebook | Purpose | Kaggle |
|---|----------|---------|--------|
| 1 | `ebrt_montecarlo_simulation.ipynb` | Full TrueBeam head Monte Carlo simulation with GEANT4. Generates training data across 3 patient anatomies with stratified clinical parameter sampling. | [🔗 Link](https://www.kaggle.com/code/ibongarciagomez/ebrt-montecarlo-simulation) |
| 2 | `consolidate_datasets_EBRT.ipynb` | Merges partial HDF5 datasets from multiple Kaggle sessions. Handles deduplication, corruption detection, and incremental consolidation. | — |
| 3 | `neural_network_EBRT.ipynb` | 3D TransUNet + FiLM training, evaluation (Gamma Index, DoseAcc, DVH), and L-BFGS treatment optimization with ICRU-83 objectives. | [🔗 Link](https://www.kaggle.com/code/ibongarciagomez/red-neuronal-ebrt) |

---

## 🚀 Quick Start

### Requirements

```bash
pip install -r requirements.txt
```

### Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU VRAM | 12 GB | 16 GB (P100/T4/A100) |
| RAM | 32 GB | 64+ GB (for dataset preloading) |
| Disk | 50 GB | 100+ GB (HDF5 datasets) |
| CPU cores | 4 | 8+ (for MC simulations) |

### Running on Kaggle (Recommended)

Both notebooks are designed for Kaggle's free GPU environment:

1. Fork the [Monte Carlo notebook](https://www.kaggle.com/code/ibongarciagomez/ebrt-montecarlo-simulation) and run to generate training data
2. Use the consolidation notebook to merge results across sessions
3. Fork the [Neural Network notebook](https://www.kaggle.com/code/ibongarciagomez/red-neuronal-ebrt) and attach the consolidated dataset
4. Training automatically resumes across Kaggle's 12-hour sessions — checkpoint-compatible

---

## 🔑 Innovations

1. **FiLM-conditioned dose prediction** — A single model handles all beam configurations through feature-wise linear modulation, unlike prior work requiring separate models per energy/geometry

2. **7-component physics-informed loss** — Domain-specific loss combining dose-weighted MSE, 3D gradient, SSIM, DVH, dose fall-off, cosine similarity, and L1 — capturing both voxel-level and distribution-level accuracy

3. **Full TrueBeam head simulation** — Complete GEANT4 model including 120 individual HD MLC tungsten leaves with clinically accurate leaf positioning algorithms

4. **End-to-end differentiable treatment planning** — L-BFGS optimization of beam parameters through the trained surrogate, with ICRU-83 clinical objectives (D95, D98, V95, HI, OAR doses)

5. **Epistemic uncertainty quantification** — MC Dropout during inference provides per-voxel uncertainty maps for clinical decision support

6. **Three patient anatomies with clinical correlations** — Pelvis, head-and-neck, and lung phantoms with anatomy-dependent beam parameter sampling reflecting real clinical protocols

---

## 📈 Training Dynamics

The model is trained with OneCycleLR using warm restarts (200 epochs/cycle) and Stochastic Weight Averaging (SWA). The dual early stopping criterion monitors both `val_loss` and `DoseAcc 3%`, preventing premature stopping when loss components compensate each other but clinical metrics continue improving.

A collapse detection system monitors for sudden DoseAcc drops (> 15 pp from best) and automatically restores the best checkpoint, providing resilience against training instabilities common in 3D medical imaging with small batch sizes.

---

## 📚 References

1. Chen, J. et al. (2021). *TransUNet: Transformers Make Strong Encoders for Medical Image Segmentation*. arXiv:2102.04306
2. Perez, E. et al. (2018). *FiLM: Visual Reasoning with a General Conditioning Layer*. AAAI 2018
3. Isensee, F. et al. (2021). *nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation*. Nature Methods, 18(2), 203–211
4. Saravanakumar, A. et al. (2025). *Commissioning and beam data validation of Varian TrueBeam*. Reports of Practical Oncology and Radiotherapy
5. Klarenberg, J. (2025). *Deep learning dose prediction for breast cancer EBRT*. JACMP
6. Shojaei, M. et al. (2025). *Semi-supervised dose prediction using nnU-Net with style-guided deformable augmentation*. PMB
7. Smith, L. N. (2019). *Super-Convergence: Very Fast Training of Neural Networks Using Large Learning Rates*. arXiv:1708.07120
8. ICRU Report 83 (2010). *Prescribing, Recording, and Reporting Photon-Beam IMRT*

---

## 👤 Authors

- **Ibon Garcia Gomez** — Neural network surrogate, treatment optimizer, dataset consolidation, Monte Carlo simulation pipeline
  - GitHub: [@igarciagomez](https://github.com/igarciagomez)
  - Kaggle: [ibongarciagomez](https://www.kaggle.com/ibongarciagomez)

- **Asier Lopez Mantecon** — Monte Carlo simulation pipeline (co-author) .

---

## 📄 License

This project is licensed under the Apache License 2.0 — see the [LICENSE](LICENSE) file for details.[@asipowa](https://github.com/asipowa)

---

<div align="center">

*Built for advancing radiation therapy treatment planning through deep learning.*

*Interested in collaboration? Feel free to reach out.*

</div>
