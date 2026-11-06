# ASTRA-Net Repository Structure

```text
ASTRA-Net-main/
├── README.md
├── .gitignore
├── src/
│   ├── Demo_inference_standalone_ver.ipynb
│   ├── Training_Shell_Script.ipynb
│   ├── requirements.txt
│   ├── FILE_DIR.md
│   ├── astra-airway/
│   │   ├── pyproject.toml
│   │   ├── README.md
│   │   ├── astra_infer/                # from astra_infer import AWInferer
│   │   └── model/                      # GitHub Release checkpoints (ignored)
│   ├── common/
│   ├── preprocessing/
│   ├── src_T1/                        # Paper implementation
│   ├── src_T2/                        # ATM26 challenge baseline, not paper
│   ├── SH/
│   ├── MD/
│   ├── py/
│   └── logs/                          # Runtime logs (ignored)
├── data/
│   ├── raw/                           # ATM22/ATM26 inputs and metadata
│   ├── Dataset/                       # Preprocessed training/evaluation cases
│   ├── Fig/                           # Generated figures
│   └── ATM26_Track1/                  # Optional standalone demo input
└── res/                               # Research checkpoints/results (ignored)
    └── Demo_Output/                   # Standalone demo output (ignored)
```

Source files in `src/` resolve raw data from `../data/raw`, processed data from `../data/Dataset`, and model experiment outputs from `../res`.

The repository keeps `.gitkeep` placeholders only under `data/` and `res/`, not original CT/GT images, case-info sheets, generated results, or model checkpoints.
Download the original CT/GT and associated metadata from official ATM22 or ATM26 sources and pretrained weights/enhanced annotations from the ASTRA-Net GitHub Releases.
See `README.md` at repository root and `src/MD/Airway_v27_ThreeFolderLayout.md` for usage and migration details.
