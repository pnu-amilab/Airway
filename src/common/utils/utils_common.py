import os
import glob
import h5py
import numpy as np
import nibabel as nib
import SimpleITK as sitk
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

plt.style.use(plt.style.available[-3])
plt.rcParams['image.cmap'] = 'gray'


def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        os.chmod(path, 0o777)


def get_result_tag(epoch, alpha):
    return f'epoch_{epoch}_alpha_{alpha:g}'


def _skip_metadata(name):
    return name.startswith('.') or name == '@eaDir' or name.endswith(('@SynoEAStream', '@SynoResource'))


def globsort(path, skip_hidden=True):
    files = glob.glob(path)
    if skip_hidden:
        files = [file for file in files if not _skip_metadata(os.path.basename(os.path.normpath(file))) and '@eaDir' not in os.path.normpath(file).split(os.sep)]
    else:
        folder, name = os.path.split(os.fspath(path))
        if glob.has_magic(name): files += glob.glob(os.path.join(folder, '.' + name))

    return sorted(set(files))


def listdir(path, skip_hidden=True):
    files = os.listdir(path)
    if skip_hidden: files = [file for file in files if not _skip_metadata(file)]

    return sorted(files)


def load_h5(path, key):
    f = h5py.File(path, 'r')
    data = f[key][:]
    f.close()

    return data


def save_h5(path, list_arr, list_key):
    f = h5py.File(path, 'w')
    for arr, key in zip(list_arr, list_key):
        f.create_dataset(key, data=arr)
    f.close()


def load_nii(path, align=True):
    data_nii = nib.load(path)
    arr = data_nii.get_fdata()
    if align and arr.ndim == 3: arr = np.swapaxes(arr, 0, 2)[::-1]

    return data_nii, arr


def save_nii(path, arr, pixdim, affine, align=True):
    if align and arr.ndim == 3: arr = np.swapaxes(arr[::-1], 0, 2)
    data_nii = nib.Nifti1Image(arr, affine)
    data_nii.header['pixdim'] = pixdim
    nib.save(data_nii, path)


def load_mha(path, align=True):
    data_mha = sitk.ReadImage(path)
    arr = sitk.GetArrayFromImage(data_mha)
    if align: arr = arr[::-1]

    return data_mha, arr


def save_mha(path, arr, pixdim, affine, align=True):
    if align: arr = arr[::-1]
    spacing = tuple(np.float64(pixdim[1:4]))
    origin = tuple(np.float64(affine[:3, 3]))
    direction = tuple(np.float64(affine[:3, :3] / np.float64(pixdim[1:4])).reshape(-1))
    data_mha = sitk.GetImageFromArray(arr)
    data_mha.SetSpacing(spacing)
    data_mha.SetOrigin(origin)
    data_mha.SetDirection(direction)
    sitk.WriteImage(data_mha, path)


def GK_int_to_binary(GK, list_gen):
    GK_binary = []
    for gen in list_gen:
        gen_m, gen_M = np.min(gen), np.max(gen)
        GK_binary.append(((GK >= gen_m + 1) & (GK <= gen_M + 1)).astype(np.uint8))
    GK_binary = np.stack(GK_binary)

    return GK_binary


def GK_binary_to_int(GK):
    GK_int = np.zeros(GK.shape[1:], dtype=np.uint8)
    for g in range(GK.shape[0]):
        GK_int[GK[g] > 0] = g + 1

    return GK_int


def minMax_scaling(x, lb=0, m=None, M=None):
    if m==None: m = x.min()
    if M==None: M = x.max()
    if lb==0:

        return (x-m)/(M-m)
    if lb==-1:

        return (2*x-(M+m))/(M-m)


def custom_cmap(base_cmap='binary_r', T=100):
    cmap = plt.cm.get_cmap(base_cmap)
    cmap = cmap(np.arange(cmap.N))
    cmap[:T, -1] = 0
    cmap = ListedColormap(cmap)

    return cmap
