# ASTRA-Net

**Discovery of Peripheral Airway Beyond Incomplete CT Annotations for Navigational Bronchoscopy**  
*IEEE Transactions on Medical Imaging*, 2026.

## Quick start: airway inference

Give `AWInferer` a 3D CT in **HU**. For ASTRA-Net, it runs R231 LungMask, builds vessel context and lung-relative positional encoding, and returns binary NumPy `uint8` masks `{'LM', 'CV', 'AW'}`. For CT-only comparison models, it skips all three context computations and returns only `{'AW'}`. Original CT orientation and shape are preserved.

> **Scope:** `src/src_T1/` is the paper's binary airway implementation. `src/src_T2/` is a separate ATM26 multiclass reference, **not a result of this paper**.

### Installation

From this repository:

```bash
pip install -e "./src/astra-airway[demo]"
```

After publishing to PyPI, users will be able to run `pip install astra-airway`. Until it is published, install from the Git repo with `pip install "git+https://github.com/guhong3648/ASTRA-Net.git#subdirectory=src/astra-airway"` (after the changes are pushed). PyTorch may need a platform-specific CUDA installation.

### Pretrained checkpoints (GitHub Releases)

The pretrained weights are hosted in [GitHub Releases](https://github.com/guhong3648/ASTRA-Net/releases), not in the Git repository or PyPI wheel
`AWInferer` automatically downloads the checkpoint for the selected model **only when it is missing** and reuses it on subsequent runs
In the cloned repository, models are stored in `src/astra-airway/model/`
When installed with pip, the default location is `~/.cache/astra-airway/model/`
You can override either location with `model_dir`, `model_path`, or the `ASTRA_MODEL_DIR` environment variable

```text
src/astra-airway/model/
├── ASTRA-Net_Phase1_Final.pt
├── ASTRA-Net_Phase2_Final.pt     # default
├── WingsNet_Final.pt
├── UNet_Final.pt
├── UDADS-Net_Final.pt
└── FANN_Final.pt
```

Only the selected model is downloaded
Automatic downloads require access to the public `v0.3.0` Release

### Enhanced annotations

Our `ATM26_Enhanced_Labels.zip` Release asset contains the project-created Enhanced airway annotations, **not original CT images or original labels**
To download and extract it into `data/raw/ATM26_Enhanced/`, run from `src/`:

```bash
python astra-airway/astra_infer/release_assets.py --data-root ../data
```

For original CT volumes and original annotations, download the datasets from the official [ATM22](https://atm22.grand-challenge.org/) and [ATM26](https://atm26.grand-challenge.org/registration/) sources under their terms
**ATM26 is recommended for reproduction and evaluation**
ATM22 and ATM26 largely share the patient cohort and original labels, with the main difference being the region of interest (ROI)

### Python example

```python
import SimpleITK as sitk
from astra_infer import AWInferer

ct_image = sitk.ReadImage('chest_ct.mha')
CT = sitk.GetArrayFromImage(ct_image)  # (D, H, W), HU

inferer = AWInferer(model_name='ASTRA-Net_Phase2')  # downloads checkpoint if missing
output = inferer(CT)
AW = output['AW']  # uint8 binary mask, same shape/orientation as CT

aw_image = sitk.GetImageFromArray(AW)
aw_image.CopyInformation(ct_image)
sitk.WriteImage(aw_image, 'AW.nii.gz')
```

Model selection: `ASTRA-Net_Phase1`, `ASTRA-Net_Phase2` (default), `WingsNet`, `UNet`, `UDADS-Net`, `FANN`. Defaults: `overlap=32` (stride 32), `batch_size=32`, `threshold=0.20`, `inference_edge=65`, 64³ patches. The edge exclusion zeros the outer 65 voxels on both Y/X borders, but does not clip any axial slices or require GT. It runs before hole filling and largest connected component filtering. CT-only comparison networks read only CT and return only AW; LungMask, CV and PE are not generated. No skeleton is generated.

[**Demo_inference_standalone_ver.ipynb**](src/Demo_inference_standalone_ver.ipynb) sets `CUDA_VISIBLE_DEVICES` in its first cell, plots superior-up coronal/sagittal and axial CT/AW projections with lung window (center −600 HU, width 1500 HU) and 16 uniformly spaced lung-cropped axial CT slices with green AW overlays, then saves the binary masks. It uses the example CT path `data/ATM26_Track1/imagesTr/ATM_615_0000.mha` and optional GT `data/ATM26_Track1/labelsTr/ATM_615.mha` (paths shown from the repository root).
Obtain the original files from the [ATM26 organizers](https://atm26.grand-challenge.org/registration/) and place them at those paths.
The original CT and GT are not distributed in this repo or its Releases.
GT is never used for inference.

For full dataset preprocessing, the `src/preprocessing/` scripts use the ATM26 Track 1 LungMask configuration (`LTRCLobes` + `R231` fill); the lightweight standalone demo intentionally uses `R231` only.

### Publish to PyPI (`pip install astra-airway`)

`src/astra-airway/pyproject.toml` is configured for the `astra-airway` distribution (import name: `astra_infer`). The public package contains only standalone inference code, not `src_T1` training networks. PyPI release must be done by the repository owner:

```bash
cd src/astra-airway
python -m pip install --upgrade build twine
python -m build
python -m twine check dist/*
python -m twine upload dist/*
```

Set up a PyPI project/account and an API token (or use GitHub Actions Trusted Publishing) before upload. On a new release, increment the version in `pyproject.toml` first. After publication, verify from a clean environment with `pip install astra-airway`. Use the PyPI project page to confirm availability; declaring the name in `pyproject.toml` alone does not publish the package. See the official [PyPI publishing guide](https://docs.pypi.org/publishing/publishing-to-pypi/).

**Validation scope:** pretrained checkpoint output accuracy requires testing with the published checkpoint and CT. Research evaluation applies additional dataset-specific processing; standalone airway masks are not claimed numerically identical without that comparison.

### Docker environment

```bash
docker pull guhong3648/guhong:v1
```

### Repo structure

| Path | Role |
|---|---|
| `src/Demo_inference_standalone_ver.ipynb` | Complete standalone inference demo |
| `src/astra-airway/` | Standalone source, checkpoints, and separate PyPI package (`from astra_infer import AWInferer`) |
| `src/src_T1/` | Paper implementation: training, inference, evaluation |
| `src/src_T2/` | Challenge multiclass reference; **not part of the paper** |
| `src/preprocessing/` | Dataset conversion for full benchmark replication |
| `src/MD/` | Experimental audit notes |

## Advanced setup: preprocessing, training, and evaluation

Install the dependencies from the repository root:

```bash
pip install -r src/requirements.txt
pip install pynrrd pydicom
```

The `pynrrd` package is used to read the Enhanced NRRD annotations; `pydicom` is needed for BAS/EXACT DICOM preprocessing. Preprocessing uses CUDA for lung segmentation and other operations. The released preprocessing implementation uses `skeletonize_3d`, so keep the compatible `scikit-image` version specified in the updated requirements.

**Dataset setup:** We recommend ATM26 for reproduction and evaluation, with ATM22 mainly used where required by the preparation pipeline. Download original CT and label files from the [ATM22 official site](https://atm22.grand-challenge.org/) or the [ATM26 challenge](https://atm26.grand-challenge.org/registration/).
They are not distributed in this repository or its GitHub Releases.
Project-created Enhanced NRRD masks and pretrained checkpoints are available in [ASTRA-Net Releases](https://github.com/guhong3648/ASTRA-Net/releases).
See [full dataset layout, metadata, and preprocessing order](src/MD/Airway_v26_PortableDataLayout.md).

### 1. Raw data layout

```text
ASTRA-Net-main/
├── src/
├── res/
└── data/
    ├── raw/
    │   ├── ATM22_TrainBatch1/
    │   │   ├── TrainBatch1_CaseInfo.csv
    │   │   ├── imagesTr/                 # original CT .nii.gz
    │   │   └── labelsTr/                 # original labels .nii.gz
    │   ├── ATM22_TrainBatch2/
    │   │   ├── TrainBatch2_CaseInfo.csv
    │   │   ├── imagesTr/
    │   │   └── labelsTr/
    │   ├── ATM22_ValidBatch/            # original validation CT .nii.gz
    │   ├── ATM26_Track1/
    │   │   ├── TrainBatch_CaseInfo.csv
    │   │   ├── imagesTr/                 # original CT .mha
    │   │   └── labelsTr/                 # original binary labels .mha
    │   ├── ATM26_Track2/labelsTr/       # original anatomical labels .mha
    │   └── ATM26_Enhanced/             # enhanced NRRD from GitHub Release
    ├── Dataset/                         # generated preprocessed cases
    └── Fig/                             # generated preprocessing figures
```

The preprocessing entry points change into `src/` before resolving the default relative paths `../data/raw/` and `../data/Dataset/`.
T1/T2 training and evaluation use `<REPO_ROOT>/data` and `<REPO_ROOT>/res`; runtime logs stay in `<REPO_ROOT>/src/logs`.
The default training and preprocessing paths are intentionally relative; edit the corresponding `hyper_parameters.py` settings to use different storage volumes.

**Working directory:** The repository contains sibling `src/`, `data/`, and `res/` folders  
Run commands and notebooks inside `src/`, where default input paths are `../data/` and output paths are `../res/`  
The parent directory may be named `ASTRA-Net` after cloning or `ASTRA-Net-main` after downloading the ZIP without changing any code  

**Metadata sheets:** The case-info CSV and Excel sheets needed for preprocessing belong in the indicated `data/raw/` folders
They are not included in the empty-directory release template
Use the official dataset materials, or on the project server run `src/SH/Airway_v26_CopyRawMetadata.sh` to copy the sheets without copying CT or label files

**Enhanced label mapping:** `Data_Processing_ATM22_Enhanced.py` and `Data_Processing_ATM26_Enhanced.py` both read the project-created Enhanced masks from `data/raw/ATM26_Enhanced/`.
These are annotation processing tasks, not independently released CT datasets.

### 2. Choose the preprocessing script

| Script under `preprocessing/` | Input / purpose | Main output under `Dataset/` |
| --- | --- | --- |
| `Data_Processing_ATM22_1.py` | ATM22 TrainBatch1 CT, labels, and case-info CSV | `ATM22_Shanghai/`, `ATM22_LIDC/` |
| `Data_Processing_ATM22_2.py` | ATM22 TrainBatch2 CT, labels, and case-info CSV | `ATM22_Shanghai/`, `ATM22_LIDC/` |
| `Data_Processing_ATM22_Valid.py` | ATM22 official validation CTs (without labels) | `ATM22_Valid/` |
| `Data_Processing_ATM26_Track_1.py` | ATM26 Track 1 CT, binary labels, and case-info CSV | `ATM26_Shanghai/`, `ATM26_LIDC/`, `ATM26_EXACT09/` |
| `Data_Processing_ATM26_Track_2.py` | ATM26 Track 2 anatomical labels and Track 1 case-info CSV | Adds `CK.h5` to corresponding ATM26 processed cases |
| `Data_Processing_ATM22_Enhanced.py` | Enhanced NRRD labels and previously processed ATM22 CT/context data | `ATM22_Enhanced/` |
| `Data_Processing_ATM26_Enhanced.py` | Same Enhanced NRRD labels, plus processed ATM22 and ATM26 CT/context data | `ATM26_Enhanced/` |
| `Data_Processing_BAS.py` | BAS DICOM CTs and NIfTI masks | `BAS/` |
| `Data_Processing_EXACT.py` | EXACT DICOM CTs and ATM22 TrainBatch2 labels/case-info | `EXACT09/` |
| `Data_Processing_PNUYH.py` | In-house CT and airway masks | `PNUYH/` |
| `Plot_Preprocessed.py` | Visualization of processed cases | Figures |

The two Enhanced pipelines are **annotation conversions**, not independent CT preprocessing pipelines:

- `ATM22_Enhanced`: loads the already-processed `ATM22_Shanghai/` or `ATM22_LIDC/` case and replaces the airway label with the Enhanced NRRD annotation. Original CT, lung mask, vessel context, positional metadata, and related files are reused.
- `ATM26_Enhanced`: loads the corresponding processed ATM26 case and the processed ATM22 CT, matches the ATM26 axial crop against the ATM22 CT, and crops the Enhanced NRRD mask to the ATM26 field of view. The script then regenerates the skeleton, weight map, and tree-parsing targets.

**Required order:** process ATM22 TrainBatch1/2 before `Data_Processing_ATM22_Enhanced.py`. For `Data_Processing_ATM26_Enhanced.py`, process **both** ATM22 TrainBatch1/2 **and** ATM26 Track 1 first. For T2, process ATM26 Track 1 before ATM26 Track 2, since the latter adds `CK.h5` to existing processed cases.

### 3. Run preprocessing

Start in the repository root and change into `src/` as shown below. All preprocessing scripts accept `--GPU_NUM`, `--idx_s` (starting zero-based index), and `--idx_n` (number of cases to process). The following commands illustrate a **20-case batch**, not the entire dataset:

```bash
# ATM26 Track 1: CT and binary airway masks
tput rmam
cd src
python -u preprocessing/Data_Processing_ATM26_Track_1.py --GPU_NUM 0 --idx_s 0 --idx_n 20

# ATM26 Track 2: anatomical labels, after Track 1 preprocessing
python -u preprocessing/Data_Processing_ATM26_Track_2.py --GPU_NUM 0 --idx_s 0 --idx_n 20

# Enhanced ATM22: after ATM22 source cases are processed
python -u preprocessing/Data_Processing_ATM22_Enhanced.py --GPU_NUM 0 --idx_s 0 --idx_n 20

# Enhanced ATM26: after both ATM22 and ATM26 Track 1 source cases are processed
python -u preprocessing/Data_Processing_ATM26_Enhanced.py --GPU_NUM 0 --idx_s 0 --idx_n 20
tput smam
```

Repeat the main-dataset commands with `--idx_s 20`, `40`, `60`, etc., until all cases are processed; the scripts clamp the final batch to the number of available entries. The Enhanced dataset contains 20 masks, so a single batch with `--idx_s 0 --idx_n 20` covers it. If preparing only Track 1, skip Track 2 label processing.

### 4. Verify the processed data

A typical labeled case is saved under `Dataset/<DATASET>/<PATIENT>/`:

```text
CT.h5                 CT volume
MK.h5                 Binary airway mask
LM.h5                 Lung/lobe mask
CV.h5                 Vessel context
SK.h5                 Skeleton target
SM.h5                 Soft airway label
MW.h5                 Training weight map
MK_parsing.h5         Parsed airway branches
Idx_SK.pt             Skeleton sampling indices
LM_pos.pt             Lung position metadata
MK_pos.pt             Airway position metadata
MK_gen.pt             Airway generation metadata
meta_info.pt          Spacing, affine, and other metadata
PM.nii.gz             Polygon lung mask
```

Some processors produce additional NIfTI masks (e.g., `GK.nii.gz`, `sk.nii.gz`) and plots. Unlabeled validation cases do not have the same label-related outputs. Track 2 adds `CK.h5` to the appropriate ATM26 patient directory; its training code also creates class sampling metadata when required.

`Preset1` in `src_T1/options/hyper_parameters.py` trains with `ATM26_Shanghai` and `ATM26_LIDC`. The Enhanced 20 patients (15 Shanghai, 5 LIDC) are held out from the original CT train/validation splits. The Shanghai and LIDC test IDs correspond to those same 20 patients. **Do not move Enhanced patients into train or validation** when reproducing the provided split. The Enhanced dataset names refer to separate annotations of those patients, not another independent set of CT acquisitions.

## Track 1: binary airway segmentation

ASTRA-Net uses 3D training patches of `64 × 64 × 64`, combining CT, lung and vessel context, and six positional encoding channels. Its training objectives are GUL, BCE, CTL, and BGR. The release uses a `0.20` probability threshold, sliding-window overlap of `32` voxels with simple probability averaging, and standalone postprocessing with in-plane edge exclusion (65 voxels per side), hole filling, and the 26-connected largest component.

Generate a single command at a time using `src/Training_Shell_Script.ipynb`, or run the entry points directly after changing into `src/` as shown below:

```bash
tput rmam
cd src
python -u src_T1/train.py --GPU_NUM 0 --name_dset Preset1 --name_model ASTRA_Net_phase_1 --epoch_load 0
python -u src_T1/inference.py --GPU_NUM 0 --name_dset Preset1 --name_model ASTRA_Net_phase_1 --epoch_load -1
python -u src_T1/evaluation.py --name_dset Preset1 --name_model ASTRA_Net_phase_1 --epoch_load -1
tput smam
```

Phase 1 uses 2,000 epochs with `alpha=10` training samples per patient per epoch. Phase 2 starts from the completed Phase 1 model, performs the teacher–student post-optimization described in the paper, and runs for 200 epochs with `alpha=10`. Use `--name_model ASTRA_Net_phase_2` to train or evaluate Phase 2 after Phase 1 is complete. The default public inference configuration is Phase 2.

### Comparison models

The six included baselines have separate network and runner files under `src_T1/`. They are **UNet, ResUNet, FANN, TSE-Net, UDADS-Net, and WingsNet** (`TSE_Net` is the code identifier for TSE). Each baseline uses GUL+BCE. UNet and ResUNet use fixed GUL parameters (`alpha=0.50`, `gamma=1.00`); FANN, TSE-Net, UDADS-Net, and WingsNet use the original epoch-wise alpha schedule (`0.10 → 0.20`, `gamma=0.70`). ASTRA-Net additionally uses CTL and BGR. To run a baseline, set `--name_model` to its code identifier (e.g., `UNet`). External nnU-Net and MEDPSeg implementations are not bundled.

## Track 2: anatomical labeling reference baseline

**Track 2 is not part of the ASTRA-Net paper.** It is provided only as a straightforward ATM26 baseline:

1. Adapt ASTRA-Net to 20 foreground anatomical classes (plus background), using the Track 1 probability/mask and other contextual inputs.
2. Train a supervised multiclass segmentation model against the anatomical label `CK`. The included network also has conditional attention and class-/branch-aware implementation details; these are not claimed as paper contributions.
3. **Paint the predicted classes inside the fixed Track 1 binary airway mask.** Track 2 does not change or recover the Track 1 airway geometry.

Before training T2, ensure `Data_Processing_ATM26_Track_2.py` has generated `CK.h5` and the Track 1 **Phase 2** checkpoint is present. Both `path_track1_checkpoint` (T1 cache generation) and `path_init_model` (T2 initialization) now default to `res/T1/Preset1_Ours_ASTRA_Net_phase_2/model/model_200.pt`. Cache generation now **clears the old T2 cache directory** and removes its T1-derived sampling indices before regeneration. Do not run it while training or evaluating T2. Then train T2 again if its original weights were initialized from Phase 1:

```bash
tput rmam
cd src
python -u src_T2/cache_track1.py --GPU_NUM 0
python -u src_T2/train.py --GPU_NUM 0 --name_dset Preset1 --name_model ASTRA_Net --epoch_load 0
python -u src_T2/inference.py --GPU_NUM 0 --name_dset Preset1 --name_model ASTRA_Net --epoch_load -2
python -u src_T2/evaluation.py --name_dset Preset1 --name_model ASTRA_Net --epoch_load -2
tput smam
```

The cache stores `probability.h5` and `mask.h5` under `HP_T2.path_cache/<DATASET>/<PATIENT>/`. Each run of cache generation deletes and rebuilds the cache. It also deletes `Idx_CK_SK_T1.pt` and `CK_count_T1.pt` for participating patients under the T2 training data path, so the T2 loader recalculates them using the new masks. T2 inference with `--epoch_load -2` uses its best saved model. T2 evaluation provides `mDSC`, `mIoU`, `mPRE`, `mSEN`, `ACC`, and `F1` and saves classwise results.

## Evaluation notebooks

Both `src/src_T1/Evaluation.ipynb` and `src/src_T2/Evaluation.ipynb` import their plotting and printing functions from `utils_history.py`. The notebooks contain explicitly listed model names, checkpoints, and dataset filters. `plot_history()` draws training curves; `print_kernel()` and `print_total()` display tabulated evaluation results with optional standard deviations. `target_dset` filters the output by dataset, `by_dset=True` shows separate tables, and `print_preset=True` prints train/test inclusion using `O` and `X`.

Evaluation results in the notebooks are populated after inference and evaluation have been run. The notebooks themselves do not perform preprocessing or model training.

## Citation

```bibtex
@article{lee2026astra,
  title={Discovery of Peripheral Airway Beyond Incomplete CT Annotations for Navigational Bronchoscopy},
  author={Lee, SiYeoul and Seo, MinKyung and Kim, Jiye and Lee, Hyeyun and Seong, EonSeung and Lee, DongEon and Kim, Do Hyung and Jeong, Yeon Joo and Seol, Hee Yun and Kim, MinWoo},
  journal={IEEE Transactions on Medical Imaging},
  year={2026},
  doi={10.1109/TMI.2026.3672178}
}
```
