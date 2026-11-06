import numpy as np


def compute_acc_f1(gt, pred, valid_mask):
    g = gt[valid_mask]
    p = pred[valid_mask]
    acc = float(np.mean(p == g)) if g.size else 0.0
    classes = [int(c) for c in np.unique(g) if int(c) != 0]
    f1s = {}
    for c in classes:
        tp = np.sum((g == c) & (p == c))
        fp = np.sum((g != c) & (p == c))
        fn = np.sum((g == c) & (p != c))
        precision = tp/(tp+fp) if tp+fp else 0.0
        recall = tp/(tp+fn) if tp+fn else 0.0
        f1s[c] = float(2*precision*recall/(precision+recall)) if precision+recall else 0.0
    macro_f1 = float(np.mean(list(f1s.values()))) if f1s else 0.0

    return acc, macro_f1, f1s
