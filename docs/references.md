# References

## Core Architecture

1. **TransUNet** — Chen, J., Lu, Y., Yu, Q., Luo, X., Adeli, E., Wang, Y., ... & Zhou, Y. (2021). *TransUNet: Transformers Make Strong Encoders for Medical Image Segmentation*. arXiv:2102.04306.
   - Foundation for our 3D TransUNet encoder-decoder architecture
   - Key insight: combining CNN local feature extraction with Transformer global attention

2. **FiLM** — Perez, E., Strub, F., de Vries, H., Dumoulin, V., & Bengio, A. (2018). *FiLM: Visual Reasoning with a General Conditioning Layer*. AAAI 2018.
   - Feature-wise Linear Modulation for conditioning on beam parameters
   - Enables a single model to handle all beam configurations

3. **nnU-Net** — Isensee, F., Jaeger, P. F., Kohl, S. A., Petersen, J., & Maier-Hein, K. H. (2021). *nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation*. Nature Methods, 18(2), 203–211.
   - Reference for augmentation pipeline and training best practices
   - Self-configuring approach that informed our hyperparameter choices

## Medical Physics

4. **TrueBeam commissioning** — Saravanakumar, A. et al. (2025). *Commissioning and beam data validation of Varian TrueBeam STx linear accelerator*. Reports of Practical Oncology and Radiotherapy.
   - Source for bremsstrahlung spectra across 5 beam qualities
   - Validated PDD and profile data used for synthetic data generation

5. **ICRU Report 83** (2010). *Prescribing, Recording, and Reporting Photon-Beam Intensity-Modulated Radiation Therapy (IMRT)*.
   - Clinical standard defining D95, D98, V95, HI, and OAR dose constraints
   - Foundation for our Stage 2 clinical objective function

## Deep Learning Dose Prediction

6. **Klarenberg, J.** (2025). *Deep learning dose prediction for breast cancer EBRT*. Journal of Applied Clinical Medical Physics (JACMP).
   - EBRT-specific dose prediction with deep learning
   - Reference for augmentation strategies (elastic deformation, intensity scaling)

7. **Shojaei, M. et al.** (2025). *Semi-supervised dose prediction using nnU-Net with style-guided deformable augmentation*. Physics in Medicine & Biology (PMB).
   - Advanced augmentation for dose prediction (sgDefAug)
   - Semi-supervised learning approach for limited labeled data

8. **Zhang, Y. et al.** (2024). *3D dose prediction using radiomics features*. AI in Medical Learning Research (AIMLR).
   - 3D dose prediction methodology
   - Radiomics-informed features for dose estimation

9. **Foster, D. et al.** (2025). *Multi-task learning for fluence and dose prediction*. PMB.
   - Multi-task approach combining fluence map and dose prediction
   - Reference for combined loss function design

## Training Techniques

10. **Super-Convergence** — Smith, L. N. & Topin, N. (2019). *Super-Convergence: Very Fast Training of Neural Networks Using Large Learning Rates*. arXiv:1708.07120.
    - OneCycleLR scheduler with warm restarts
    - Enables faster convergence with higher learning rates

11. **SWA** — Izmailov, P., Podoprikhin, D., Garipov, T., Vetrov, D., & Wilson, A. G. (2018). *Averaging Weights Leads to Wider Optima and Better Generalization*. UAI 2018.
    - Stochastic Weight Averaging for improved generalization
    - Implemented from epoch 400 with updates every 5 epochs

12. **MC Dropout** — Gal, Y. & Ghahramani, Z. (2016). *Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning*. ICML 2016.
    - Epistemic uncertainty quantification via stochastic forward passes
    - Used for treatment plan confidence assessment in Stage 2

## Monte Carlo Simulation

13. **GEANT4** — Agostinelli, S. et al. (2003). *GEANT4—a simulation toolkit*. Nuclear Instruments and Methods in Physics Research Section A, 506(3), 250–303.
    - Core particle transport engine

14. **OpenGATE** — Sarrut, D. et al. (2022). *The OpenGATE ecosystem for Monte Carlo simulation in medical physics*. Physics in Medicine & Biology, 67(18), 184001.
    - Python API for GEANT4
    - Simplified geometry construction and dose scoring

15. **Varian HD 120 MLC** — Varian Medical Systems. *TrueBeam Technical Reference Guide*. Palo Alto, CA.
    - HD 120 MLC specifications: 60 leaf pairs (32 × 2.5 mm + 28 × 5.0 mm)
    - Tungsten leaf thickness: 6.5 cm
    - Maximum field size: 40 × 22 cm (at isocenter)
