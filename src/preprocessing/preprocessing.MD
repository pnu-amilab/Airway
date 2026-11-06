# Preprocessing

The preprocessing root path is controlled by `HP_Preprocessing` in `src_T1/options/hyper_parameters.py`.

```python
@dataclass
class HP_Preprocessing:
    path_main: str = '...'
```

Raw datasets are expected under `<path_main>/raw`, and processed datasets are written under `<path_main>/Dataset`.

## Scripts

```text
Data_Processing_ATM22_1.py
Data_Processing_ATM22_2.py
Data_Processing_ATM22_Enhanced.py
Data_Processing_ATM22_Valid.py
Data_Processing_ATM26_Enhanced.py
Data_Processing_ATM26_Track_1.py
Data_Processing_ATM26_Track_2.py
Data_Processing_BAS.py
Data_Processing_EXACT.py
Data_Processing_PNUYH.py
Plot_Preprocessed.py
```

Run preprocessing from the repository root. For example:

```bash
tput rmam
python preprocessing/Data_Processing_ATM26_Track_1.py --GPU_NUM 0 --idx_s 0 --idx_n 20
tput smam
```

The scripts accept `--idx_s` and `--idx_n`, allowing a dataset to be split across multiple preprocessing jobs.

## Processed case layout

Depending on the dataset and task, each processed case contains the volumes and metadata required by ASTRA-Net, including:

```text
CT.h5
MK.h5
LM.h5
CV.h5
SK.h5
MW.h5
MK_parsing.h5
Idx_SK.pt
LM_pos.pt
MK_pos.pt
meta_info.pt
```

`CT` is the input volume, `MK` is the binary airway mask, `LM` is the lung mask, `CV` is the vessel/context volume, and `SK`, `MW`, and `MK_parsing` provide training targets used by the ASTRA-Net pipeline.

Track 2 preprocessing additionally generates the anatomical airway labels used by `src_T2`.

## Dataset selection

Training and evaluation datasets are selected through `hp.dset_preset` in the corresponding hyper-parameter file. Dataset paths should be changed there rather than inside preprocessing, training, or inference scripts.
