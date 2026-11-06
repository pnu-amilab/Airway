import torch
import torch.nn as nn
import torch.nn.functional as F


SIDE_GROUPS = [
    [0, 1, 2, 3, 4, 5, 6, 7, 19],
    [8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
    [18]]

LOBE_GROUPS = [
    [0, 1, 2, 3],
    [4, 5, 6, 7, 19],
    [8, 9, 10],
    [11, 12],
    [13, 14, 15, 16, 17],
    [18]]


class ForegroundCCE(nn.Module):

    def __init__(self, ignore_index=0, weight=None):
        super().__init__()
        self.ignore_index = ignore_index
        self.register_buffer('weight', weight)

    def forward(self, logits, target):
        mask = target != self.ignore_index
        if not torch.any(mask): return logits.sum()*0.0
        target = torch.clamp(target-1, min=0)
        loss = F.cross_entropy(logits, target, weight=self.weight, reduction='none')

        return loss[mask].mean()


class ForegroundHierarchicalCCE(nn.Module):

    def __init__(self, ignore_index=0):
        super().__init__()
        self.ignore_index = ignore_index

    def _group_loss(self, logits, target, groups):
        mask = target != self.ignore_index
        if not torch.any(mask): return logits.sum()*0.0
        coarse_logits = torch.stack([torch.logsumexp(logits[:, group], dim=1) for group in groups], dim=1)
        label_map = torch.zeros(logits.shape[1], dtype=torch.long, device=target.device)
        for class_idx, group in enumerate(groups): label_map[group] = class_idx
        coarse_target = label_map[torch.clamp(target-1, min=0)]
        loss = F.cross_entropy(coarse_logits, coarse_target, reduction='none')

        return loss[mask].mean()

    def forward(self, logits, target):
        return (self._group_loss(logits, target, SIDE_GROUPS)+self._group_loss(logits, target, LOBE_GROUPS))/2


class BranchCCE(nn.Module):

    def __init__(self, ignore_index=0, min_voxels=4):
        super().__init__()
        self.ignore_index = ignore_index
        self.min_voxels = min_voxels

    def forward(self, logits, target, branch_map):
        list_logits, list_target = [], []
        for batch_idx in range(logits.shape[0]):
            target_batch = target[batch_idx].reshape(-1)
            branch_batch = branch_map[batch_idx].long().reshape(-1)
            logits_batch = logits[batch_idx].reshape(logits.shape[1], -1).transpose(0, 1)
            valid = (target_batch != self.ignore_index) & (branch_batch > 0)
            if not torch.any(valid): continue
            branch_batch = branch_batch[valid]
            target_batch = target_batch[valid].long()-1
            logits_batch = logits_batch[valid]
            _, inverse = torch.unique(branch_batch, sorted=False, return_inverse=True)
            num_branch = int(inverse.max().item())+1
            count = torch.zeros(num_branch, device=logits.device, dtype=logits.dtype)
            count.index_add_(0, inverse, torch.ones_like(inverse, dtype=logits.dtype))
            branch_logits = torch.zeros(num_branch, logits.shape[1], device=logits.device, dtype=logits.dtype)
            branch_logits.index_add_(0, inverse, logits_batch)
            branch_logits = branch_logits/count[:, None].clamp_min(1)
            target_count = torch.zeros_like(branch_logits)
            target_count.index_add_(0, inverse, F.one_hot(target_batch, num_classes=logits.shape[1]).to(logits.dtype))
            keep = count >= self.min_voxels
            if not torch.any(keep): continue
            list_logits.append(branch_logits[keep])
            list_target.append(target_count.argmax(dim=1)[keep])
        if len(list_logits) == 0: return logits.sum()*0.0

        return F.cross_entropy(torch.concat(list_logits), torch.concat(list_target))


class BranchConsistency(nn.Module):

    def __init__(self, ignore_index=0, min_voxels=4):
        super().__init__()
        self.ignore_index = ignore_index
        self.min_voxels = min_voxels

    def forward(self, logits, target, branch_map):
        probability = torch.softmax(logits.float(), dim=1)
        list_loss = []
        for batch_idx in range(logits.shape[0]):
            target_batch = target[batch_idx].reshape(-1)
            branch_batch = branch_map[batch_idx].long().reshape(-1)
            p_batch = probability[batch_idx].reshape(probability.shape[1], -1).transpose(0, 1)
            valid = (target_batch != self.ignore_index) & (branch_batch > 0)
            if not torch.any(valid): continue
            branch_batch = branch_batch[valid]
            p_batch = p_batch[valid]
            _, inverse = torch.unique(branch_batch, sorted=False, return_inverse=True)
            num_branch = int(inverse.max().item())+1
            count = torch.zeros(num_branch, device=logits.device, dtype=torch.float32)
            count.index_add_(0, inverse, torch.ones_like(inverse, dtype=torch.float32))
            branch_probability = torch.zeros(num_branch, logits.shape[1], device=logits.device, dtype=torch.float32)
            branch_probability.index_add_(0, inverse, p_batch)
            branch_probability = branch_probability/count[:, None].clamp_min(1)
            keep = count[inverse] >= self.min_voxels
            if not torch.any(keep): continue
            target_probability = branch_probability[inverse].detach()
            list_loss.append(F.kl_div(torch.log(p_batch[keep]+1e-6), target_probability[keep], reduction='batchmean'))
        if len(list_loss) == 0: return logits.sum()*0.0

        return torch.stack(list_loss).mean().to(logits.dtype)


_CLASS_GROUP = {
    1:('trachea', 'L', 'LUL'), 2:('trachea', 'L', 'LUL'), 3:('trachea', 'L', 'LUL'), 4:('trachea', 'L', 'LUL'),
    5:('trachea', 'L', 'LLL'), 6:('trachea', 'L', 'LLL'), 7:('trachea', 'L', 'LLL'), 8:('trachea', 'L', 'LLL'), 20:('trachea', 'L', 'LLL'),
    9:('trachea', 'R', 'RUL'), 10:('trachea', 'R', 'RUL'), 11:('trachea', 'R', 'RUL'),
    12:('trachea', 'R', 'RML'), 13:('trachea', 'R', 'RML'),
    14:('trachea', 'R', 'RLL'), 15:('trachea', 'R', 'RLL'), 16:('trachea', 'R', 'RLL'), 17:('trachea', 'R', 'RLL'), 18:('trachea', 'R', 'RLL')}


def _class_path(label):
    if label == 19: return ('trachea',)

    return _CLASS_GROUP[label]+(label,)


def _hierarchical_distance(a, b):
    if a == b: return 0
    pa, pb = _class_path(a), _class_path(b)
    common = 0
    for x, y in zip(pa, pb):
        if x != y: break
        common += 1

    return len(pa)-common+len(pb)-common


class HierarchicalDistanceLoss(nn.Module):

    def __init__(self, num_classes=20, ignore_index=0):
        super().__init__()
        self.ignore_index = ignore_index
        matrix = torch.zeros(num_classes, num_classes, dtype=torch.float32)
        for target in range(1, num_classes+1):
            for prediction in range(1, num_classes+1): matrix[target-1, prediction-1] = _hierarchical_distance(target, prediction)
        matrix /= matrix.max()
        self.register_buffer('distance', matrix)

    def forward(self, logits, target):
        mask = target != self.ignore_index
        if not torch.any(mask): return logits.sum()*0.0
        probability = torch.softmax(logits.float(), dim=1).permute(0, 2, 3, 4, 1)[mask]
        target_idx = target[mask].long()-1
        expected_distance = torch.sum(probability*self.distance[target_idx], dim=1)

        return expected_distance.mean().to(logits.dtype)


class MultiClassDice(nn.Module):

    def __init__(self, num_classes=21, ignore_index=0, smooth=1e-5):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.smooth = smooth

    def forward(self, pred, target):
        values = []
        for class_idx in range(self.num_classes):
            if class_idx == self.ignore_index: continue
            y = target == class_idx
            if not torch.any(y): continue
            p = pred == class_idx
            values.append(2*torch.sum(p*y)/(torch.sum(p)+torch.sum(y)+self.smooth))
        if len(values) == 0: return pred.sum()*0.0

        return torch.stack(values).mean()


class ForegroundAccuracy(nn.Module):

    def __init__(self, ignore_index=0):
        super().__init__()
        self.ignore_index = ignore_index

    def forward(self, pred, target):
        mask = target != self.ignore_index
        if not torch.any(mask): return pred.sum()*0.0

        return (pred[mask] == target[mask]).float().mean()
