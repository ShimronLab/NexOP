# NexOP
## Official Implementation
**Paper Title:** NexOP: Joint Optimization of NEX-Aware k-space Sampling and Image Reconstruction for Low-Field MRI

**Authors:** Tal Oved, Efrat Shimron

[![DOI](https://zenodo.org/badge/1173554603.svg)](https://doi.org/10.5281/zenodo.19588428)

---

## Project Structure
The repository is organized as follows:

```text
NexOP
├── core_models
│   ├── M4RawData               # Dataset files
│   ├── utils                   # Helper functions 
│   ├── Compare_M4Raw.py        # Comparison script for different reconstruction models
│   ├── M4Raw_Trainer.py        # Main training script
│   ├── M4RawDataset.py         # Data loader implementation
│   ├── ReconModule.py          # Unrolled multi-NEX recon model
│   ├── Multi_To_Single.py      # ESPIRiT-based data preprocessing
│   ├── NexOP_model.py          # Core NexOP network architecture
│   ├── subsample_fastmri.py        
│   └── Test_Statistics.py      # Calculating evaluation metrics
├── LICENSE
└── README.md
```

## Data
This project utilizes the M4Raw dataset by [Lyu et al.](https://www.nature.com/articles/s41597-023-02181-4)

## Checkpoints
Pre-trained model weights are hosted publicly on Figshare.

[Download](https://doi.org/10.6084/m9.figshare.32008749) Checkpoints. 


## Citation

If you use this code or the paper's results, please cite:

[NexOP: Joint Optimization of NEX-Aware k-space Sampling and Image Reconstruction for Low-Field MRI](https://arxiv.org/abs/2605.11583) 

```bibtex
@article{oved2026nexop,
  title={NexOP: Joint Optimization of NEX-Aware k-space Sampling and Image Reconstruction for Low-Field MRI},
  author={Oved, Tal and Shimron, Efrat},
  journal={arXiv preprint arXiv:2605.11583},
  year={2026}
}