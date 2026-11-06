import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.draw import polygon
from scipy.spatial import ConvexHull
from scipy import ndimage

import torch
import torch.nn.functional as F

plt.style.use(plt.style.available[-3])
plt.rcParams['image.cmap'] = 'gray'

torch.set_default_dtype(torch.float32)


def read_case_info(path):
    try:
        meta = pd.read_excel(path)
    except Exception:
        try:
            meta = pd.read_csv(path, encoding='utf-8-sig')
        except Exception:
            try:
                meta = pd.read_csv(path, encoding='cp949')
            except Exception:
                meta = pd.read_csv(path, encoding='latin1', sep=None, engine='python')

    return meta


def imread(file_path):
    npArr = np.fromfile(file_path, dtype=np.uint8)

    return cv2.imdecode(npArr, cv2.IMREAD_GRAYSCALE)


def plot_projection(hk, jk):
    imgs = [(np.mean(jk, axis=0)>0).astype(np.float32), (np.mean(jk, axis=1)>0).astype(np.float32), (np.mean(jk, axis=2)>0).astype(np.float32),
            (np.mean(hk, axis=0)>0).astype(np.float32), (np.mean(hk, axis=1)>0).astype(np.float32), (np.mean(hk, axis=2)>0).astype(np.float32)]
    imgs = imgs + [imgs[0]-imgs[3], imgs[1]-imgs[4], imgs[2]-imgs[5]]
    titles = ['A_x', 'A_y', 'A_z', 'B_x', 'B_y', 'B_z', 'A-B', 'A-B', 'A-B']
    plt.figure(figsize=(30, 30))
    for i in range(len(imgs)):
        plt.subplot(3, 3, i+1)
        plt.title(titles[i])
        img = imgs[i]
        img[0, 0] = 0
        img[0, 1] = 1
        if i>5: img[0, 2] = -1
        cmap = 'gray' if i<=5 else 'jet'
        plt.imshow(img, cmap, interpolation=None)
        plt.grid(c='y', ls='--')


def euclidean_dist(dx, dy):
    return np.sqrt(np.power(dx, 2) + np.power(dy, 2))


def denoise_vessels(lung_contour, vessels):
    vessels_coords_x, vessels_coords_y = np.nonzero(vessels)  # get non zero coordinates
    for contour in lung_contour:
        x_points, y_points = contour[:, 0], contour[:, 1]
        for (coord_x, coord_y) in zip(vessels_coords_x, vessels_coords_y):
            for (x, y) in zip(x_points, y_points):
                d = euclidean_dist(x - coord_x, y - coord_y)
                if d <= 0.1:
                    vessels[coord_x, coord_y] = 0

    return vessels


def get_soft_label(y_data, kernel_size=5, sigma=1.0):
    x = torch.linspace(-sigma, sigma, kernel_size)
    y = torch.linspace(-sigma, sigma, kernel_size)
    x, y = torch.meshgrid(x, y)
    d = torch.sqrt(x*x + y*y)
    kernel = torch.exp(-(d**2 / (2.0*sigma**2)))
    kernel = kernel/kernel.sum()
    kernel = kernel.unsqueeze(0).unsqueeze(0)

    Res = []
    for n in range(len(y_data)):
        y = torch.tensor(y_data[n],  dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        res = F.conv2d(y, kernel, padding=kernel_size//2)
        Res.append(res)
    Res = np.array(torch.concat(Res, axis=0).squeeze())

    return Res

def polygon_single(img):
    points = np.argwhere(img==1)
    if len(points) < 3: return img
    if len(np.unique(points[:, 0])) < 2: return img
    if len(np.unique(points[:, 1])) < 2: return img

    try:
        hull = ConvexHull(points)
    except:
        return img

    hull_points = points[hull.vertices]

    res = np.zeros_like(img)
    rr, cc = polygon(hull_points[:, 0], hull_points[:, 1])
    rr = np.clip(rr, 0, res.shape[0]-1)
    cc = np.clip(cc, 0, res.shape[1]-1)
    res[rr, cc] = 1

    return res


def polygon_all(imgs):
    for i in range(len(imgs)):
        if imgs[i].sum()==0: continue
        imgs[i] = polygon_single(imgs[i])

    return imgs



def get_local_imbalance_weight(MK, kernel_size=7):
    kernel = np.ones([kernel_size]*3, dtype=np.float32)
    LW = ndimage.convolve(MK.astype(np.float32), kernel, mode='mirror')/kernel.sum()
    LW[LW==0] = 1
    LW = -np.log10(LW)
    LW = LW*MK

    return np.float16(LW)
