# NexOP
## Official Implementation
**Paper Title:** NexOP: Joint Optimization of NEX-Aware k-space Sampling and Image Reconstruction for Low-Field MRI

**Authors:** Tal Oved, Efrat Shimron

---

## Project Structure
The repository is organized as follows:

```text
NexOP
├── core_models
│   ├── M4RawData               # Dataset files
│   ├── unet                    # U-Net architecture components
│   └── utils                   # Helper functions 
├── Compare_M4Raw.py            # Comparison script for different reconstruction models
├── M4Raw_Trainer.py            # Main training script
├── M4RawDataset.py             # Data loader implementation
├── modl.py                     # unrolled model architecture implementation
├── Multi_To_Single.py          # ESPIRiT-based data preprocessing
├── NexOP_model.py              # Core NexOP network architecture
├── subsample_fastmri.py        
├── Test_Statistics.py          # Script for calculating evaluation metrics
├── LICENSE
└── README.md
```

## Data
This project utilizes the M4Raw dataset by [Lyu et al.](https://www.nature.com/articles/s41597-023-02181-4)

