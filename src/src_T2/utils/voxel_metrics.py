import numpy as np

BACKGROUND_LABEL = 0


def present_classes(gt_seg):
    return [int(c) for c in np.unique(gt_seg) if int(c) != BACKGROUND_LABEL]


def _dice(pred, gt, smooth=1e-5):
    intersection = float(np.count_nonzero(pred & gt))

    return (2.0*intersection+smooth)/(float(pred.sum())+float(gt.sum())+smooth)


def multi_class_dice(pred_seg, gt_seg, smooth=1e-5):
    per_class = {}
    for c in present_classes(gt_seg): per_class[c] = float(_dice(pred_seg == c, gt_seg == c, smooth))
    mean = float(np.mean(list(per_class.values()))) if per_class else 0.0

    return mean, per_class


def multi_class_overlap(pred_seg, gt_seg, smooth=1e-5):
    per_class = {'DSC_class':{}, 'IoU_class':{}, 'PRE_class':{}, 'SEN_class':{}}
    for c in present_classes(gt_seg):
        pred = pred_seg == c
        gt = gt_seg == c
        tp = float(np.count_nonzero(pred & gt))
        fp = float(np.count_nonzero(pred & ~gt))
        fn = float(np.count_nonzero(~pred & gt))
        per_class['DSC_class'][c] = (2*tp+smooth)/(2*tp+fp+fn+smooth)
        per_class['IoU_class'][c] = tp/(tp+fp+fn)
        per_class['PRE_class'][c] = tp/(tp+fp) if tp+fp else 0.0
        per_class['SEN_class'][c] = tp/(tp+fn)

    means = {key:float(np.mean(list(values.values()))) if values else 0.0 for key, values in per_class.items()}

    return means, per_class


def multi_class_cldice(pred_seg, gt_seg, gt_skeleton):
    from skimage.morphology import skeletonize

    per_class = {}
    for c in present_classes(gt_seg):
        gt = gt_seg == c
        sk_gt = (gt_skeleton > 0) & gt
        if not np.any(sk_gt): continue
        pred = pred_seg == c
        if not np.any(pred):
            per_class[c] = 0.0
            continue
        sk_pred = skeletonize(pred)
        tprec = np.count_nonzero(pred & sk_gt)/np.count_nonzero(sk_gt)
        tsens = np.count_nonzero(gt & sk_pred)/np.count_nonzero(sk_pred) if np.any(sk_pred) else 0.0
        per_class[c] = float(2*tprec*tsens/(tprec+tsens)) if tprec+tsens else 0.0

    return float(np.mean(list(per_class.values()))) if per_class else 0.0, per_class
