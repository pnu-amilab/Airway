from pathlib import Path
import os
import time

import cv2
import numpy as np
from scipy import ndimage
from tqdm import tqdm

import torch

from .network_ASTRA_Net import ASTRA_Net
from .release_assets import ensure_checkpoint


class AWInferer:
    MODEL_NAMES = ('ASTRA-Net_Phase1', 'ASTRA-Net_Phase2', 'WingsNet', 'UNet', 'UDADS-Net', 'FANN')

    def __init__(self, model_path=None, threshold=0.20, overlap=32, batch_size=32, device='cuda', align=True, lungmask_batch_size=16, lungmask_model_path=None, verbose=True, model_name='ASTRA-Net_Phase2', model_dir=None, inference_edge=65):
        self.patch_size = 64
        self.threshold = threshold
        self.overlap = overlap
        self.batch_size = batch_size
        self.inference_edge = inference_edge
        self.align = align
        self.verbose = verbose
        self.ct_min = -1200
        self.ct_max = 600
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model_name = model_name
        if self.model_name not in self.MODEL_NAMES: raise ValueError(f'Unsupported model_name={model_name}. Choose from {self.MODEL_NAMES}')
        self.use_context = self.model_name in ('ASTRA-Net_Phase1', 'ASTRA-Net_Phase2')
        self.dim_in = 9 if self.use_context else 1

        if model_path is None:
            path_root = Path(__file__).resolve().parent.parent
            default_model_dir = path_root/'model' if (path_root/'model').is_dir() else Path.home()/'.cache/astra-airway/model'
            model_dir = Path(model_dir or os.environ.get('ASTRA_MODEL_DIR', default_model_dir)).expanduser()
            model_path = model_dir/f'{self.model_name}_Final.pt'
        model_path = Path(model_path).expanduser()
        if not model_path.is_file(): ensure_checkpoint(model_path, self.model_name)

        t0 = time.perf_counter()
        self._log(f'Model configuration | model={self.model_name} | device={self.device} | patch={self.patch_size} | overlap={self.overlap} | stride={self.patch_size-self.overlap} | edge={self.inference_edge} | batch={self.batch_size} done', t0)
        self.model = self._build_model()
        state_dict = torch.load(model_path, map_location='cpu', weights_only=False)
        if isinstance(state_dict, dict) and 'model' in state_dict: state_dict = state_dict['model']
        if isinstance(state_dict, dict) and 'state_dict' in state_dict: state_dict = state_dict['state_dict']
        if any(key.startswith('module.') for key in state_dict): state_dict = {key.removeprefix('module.'):value for key, value in state_dict.items()}
        self.model.load_state_dict(state_dict)
        self.model.eval().to(self.device)
        self._log(f'{self.model_name} checkpoint loading | {model_path} done', t0)

        self.model_lm = None
        if self.use_context:
            from lungmask import LMInferer
            t0 = time.perf_counter()
            lm_kwargs = {'modelname':'R231', 'batch_size':lungmask_batch_size, 'tqdm_disable':not self.verbose}
            if lungmask_model_path is not None: lm_kwargs['modelpath'] = str(lungmask_model_path)
            self.model_lm = LMInferer(**lm_kwargs)
            self._log('LungMask R231 loading done', t0)

    def _build_model(self):
        if self.use_context:
            return ASTRA_Net(dim_in=9, dim_base=16, ega=True, ega_learnable=True, ega_recalibration=True)
        if self.model_name == 'UNet':
            from .network_others_UNet import UNet
            return UNet(dim_in=1)
        if self.model_name == 'WingsNet':
            from .network_others_WingsNet import WingsNet
            return WingsNet(size=64, dim_in=1, dim_base=16)
        if self.model_name == 'FANN':
            from .network_others_FANN import FANN
            return FANN(in_channel=1)
        if self.model_name == 'UDADS-Net':
            from .network_others_UDADS_Net import UDADS_Net
            return UDADS_Net()

    def _log(self, text, t0):
        if self.verbose: print(f'[{time.perf_counter()-t0:7.2f}s] [AWInferer] {text}', flush=True)

    def __call__(self, ct):
        t_total = time.perf_counter()
        ct = np.asarray(ct, dtype=np.float32)
        if ct.ndim != 3: raise ValueError(f'CT must be 3D, got shape={ct.shape}')

        t0 = time.perf_counter()
        ct_work = np.ascontiguousarray(ct[::-1] if self.align else ct)
        self._log(f'[0/4] CT orientation alignment | shape={ct_work.shape} done', t0)

        lm = cv = pe_axes = None
        if self.use_context:
            t0 = time.perf_counter()
            lm = (self.model_lm.apply(ct_work) > 0).astype(np.uint8)
            self._log(f'[1/4] LungMask inference | voxels={int(lm.sum()):,} done', t0)

            t0 = time.perf_counter()
            cv = self._make_cv(ct_work, lm)
            lm_pos = self._get_mask_position(lm)
            pe_axes = self._make_pe_axes(ct_work.shape, lm_pos)
            self._log(f'[2/4] CV and positional encoding | LM bbox=({lm_pos["idx_xm"]}:{lm_pos["idx_xM"]}, {lm_pos["idx_ym"]}:{lm_pos["idx_yM"]}, {lm_pos["idx_zm"]}:{lm_pos["idx_zM"]}) done', t0)
        else:
            self._log('[1/4] LungMask skipped (CT-only model) done', time.perf_counter())
            self._log('[2/4] CV and positional encoding skipped (CT-only model) done', time.perf_counter())

        centers = self._make_inference_centers(ct_work.shape)
        t0 = time.perf_counter()
        probability = self._sliding_window_inference(ct_work, lm, cv, pe_axes, centers)
        self._log(f'[3/4] {self.model_name} sliding-window inference done', t0)

        t0 = time.perf_counter()
        aw = self._postprocess(probability)
        self._log(f'[4/4] AW postprocessing | threshold={self.threshold} | edge={self.inference_edge} | voxels={int(aw.sum()):,} done', t0)

        t0 = time.perf_counter()
        output = {'LM':lm, 'CV':cv, 'AW':aw} if self.use_context else {'AW':aw}
        if self.align: output = {key:np.ascontiguousarray(value[::-1]) for key, value in output.items()}
        self._log(f'Output orientation restoration | channels={list(output)} | dtype=uint8 done', t0)
        self._log(f'CT-to-AW pipeline | shape={aw.shape} done', t_total)

        return output

    def _make_cv(self, ct, lm):
        ct_w = np.clip(ct, -1000, -300)
        cv = lm*ct_w
        cv[cv == 0] = -1000
        cv = (cv >= -500).astype(np.float32)

        iterator = range(len(cv))
        if self.verbose: iterator = tqdm(iterator, desc='[AWInferer] CV', unit='slice', leave=False)
        for k in iterator:
            lm_slice = np.ascontiguousarray(lm[k].astype(np.uint8)*255)
            contours, _ = cv2.findContours(lm_slice, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            boundary = np.zeros_like(lm_slice, dtype=np.uint8)
            cv2.drawContours(boundary, contours, -1, 1, thickness=3)
            cv[k][boundary > 0] = 0

        return cv.astype(np.uint8)

    def _get_mask_position(self, mask):
        if not np.any(mask):
            return {
                'idx_xm':0, 'idx_xM':mask.shape[0]-1,
                'idx_ym':0, 'idx_yM':mask.shape[1]-1,
                'idx_zm':0, 'idx_zM':mask.shape[2]-1}

        mask_binary = mask > 0
        projections = (
            np.any(mask_binary, axis=(1, 2)),
            np.any(mask_binary, axis=(0, 2)),
            np.any(mask_binary, axis=(0, 1)))
        bounds = []
        for projection in projections:
            indices = np.where(projection)[0]
            bounds.extend((int(indices.min()), int(indices.max())))

        return {
            'idx_xm':bounds[0], 'idx_xM':bounds[1],
            'idx_ym':bounds[2], 'idx_yM':bounds[3],
            'idx_zm':bounds[4], 'idx_zM':bounds[5]}

    def _make_pe_axes(self, shape, lm_pos):
        axes = []
        for axis, key_m, key_M in zip(range(3), ['idx_xm', 'idx_ym', 'idx_zm'], ['idx_xM', 'idx_yM', 'idx_zM']):
            denominator = lm_pos[key_M]-lm_pos[key_m]
            if denominator == 0: denominator = 1

            position = ((np.arange(shape[axis])-lm_pos[key_m])/denominator).astype(np.float32)
            theta = (2*position-1)*np.pi*0.5
            div_term = np.exp(np.arange(0, 2, 2)*-(np.log(10000.0)/2))
            sine = np.sin(theta[:, None]*div_term)[:, 0].astype(np.float32)
            cosine = np.cos(theta[:, None]*div_term)[:, 0].astype(np.float32)
            axes.append((sine, cosine))

        return axes

    def _make_inference_centers(self, shape):
        list_axis = [self._make_centers(length) for length in shape]

        return np.array([[x, y, z] for x in list_axis[0] for y in list_axis[1] for z in list_axis[2]], dtype=np.int32)

    def _make_centers(self, length):
        window = self.patch_size//2
        stride = self.patch_size-self.overlap
        centers = list(range(window, length-window+1, stride))
        last = length-window
        if len(centers) == 0: centers = [last]
        elif centers[-1] != last: centers.append(last)

        return centers

    def _make_patch(self, ct, lm, cv, pe_axes, center):
        x, y, z = center
        w = self.patch_size//2
        xs, xe = x-w, x+w
        ys, ye = y-w, y+w
        zs, ze = z-w, z+w

        patch = np.empty((self.dim_in, self.patch_size, self.patch_size, self.patch_size), dtype=np.float32)
        ct_patch = np.clip(ct[xs:xe, ys:ye, zs:ze], self.ct_min, self.ct_max)
        patch[0] = (ct_patch-self.ct_min)/(self.ct_max-self.ct_min)
        if not self.use_context: return patch

        patch[1] = lm[xs:xe, ys:ye, zs:ze]
        patch[2] = cv[xs:xe, ys:ye, zs:ze]
        patch[3] = pe_axes[0][0][xs:xe, None, None]
        patch[4] = pe_axes[0][1][xs:xe, None, None]
        patch[5] = pe_axes[1][0][None, ys:ye, None]
        patch[6] = pe_axes[1][1][None, ys:ye, None]
        patch[7] = pe_axes[2][0][None, None, zs:ze]
        patch[8] = pe_axes[2][1][None, None, zs:ze]

        return patch

    def _sliding_window_inference(self, ct, lm, cv, pe_axes, centers):
        probability = np.zeros(ct.shape, dtype=np.float32)
        count = np.zeros(ct.shape, dtype=np.float32)
        window = self.patch_size//2
        starts = range(0, len(centers), self.batch_size)
        if self.verbose: starts = tqdm(starts, total=(len(centers)+self.batch_size-1)//self.batch_size, desc=f'[AWInferer] {self.model_name}', unit='batch')

        with torch.inference_mode():
            for start in starts:
                batch_centers = centers[start:start+self.batch_size]
                batch = np.empty((len(batch_centers), self.dim_in, self.patch_size, self.patch_size, self.patch_size), dtype=np.float32)
                for i, center in enumerate(batch_centers): batch[i] = self._make_patch(ct, lm, cv, pe_axes, center)
                x = torch.from_numpy(batch).to(self.device, non_blocking=True)

                pred = self.model(x)[1][:, 0].float().cpu().numpy()

                for p, center in zip(pred, batch_centers):
                    x_, y_, z_ = center
                    xs, xe = x_-window, x_+window
                    ys, ye = y_-window, y_+window
                    zs, ze = z_-window, z_+window
                    probability[xs:xe, ys:ye, zs:ze] += p
                    count[xs:xe, ys:ye, zs:ze] += 1

        probability /= np.maximum(count, 1)

        return probability

    def _postprocess(self, probability):
        edge = self.inference_edge
        if edge > 0:
            probability[:, :edge, :] = 0
            probability[:, -edge:, :] = 0
            probability[:, :, :edge] = 0
            probability[:, :, -edge:] = 0
        prediction = probability > self.threshold
        prediction = ndimage.binary_fill_holes(prediction)

        return self._largest_connected_component(prediction)

    def _largest_connected_component(self, prediction):
        labels, n_label = ndimage.label(prediction > 0, structure=ndimage.generate_binary_structure(3, 3))
        if n_label == 0: return np.zeros_like(prediction, dtype=np.uint8)
        sizes = np.bincount(labels.ravel())
        sizes[0] = 0

        return (labels == sizes.argmax()).astype(np.uint8)


def infer_airway(ct, **kwargs):
    return AWInferer(**kwargs)(ct)
