import numpy as np
from scipy import ndimage
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path


SEG_GROUP = {
    1:('L', 'LUL'), 2:('L', 'LUL'), 3:('L', 'LUL'), 4:('L', 'LUL'),
    5:('L', 'LLL'), 6:('L', 'LLL'), 7:('L', 'LLL'), 8:('L', 'LLL'), 20:('L', 'LLL'),
    9:('R', 'RUL'), 10:('R', 'RUL'), 11:('R', 'RUL'),
    12:('R', 'RML'), 13:('R', 'RML'),
    14:('R', 'RLL'), 15:('R', 'RLL'), 16:('R', 'RLL'), 17:('R', 'RLL'), 18:('R', 'RLL')}


def largest_component(seg):
    labeled, n = ndimage.label(seg > 0, structure=np.ones((3, 3, 3), dtype=np.uint8))
    if n < 2: return seg
    counts = np.bincount(labeled.ravel())
    counts[0] = 0

    return np.where(labeled == np.argmax(counts), seg, 0)


def project_to_nodes(seg, parsing, node_num, num_classes=21):
    mask = parsing > 0
    node = parsing[mask].astype(np.int64) - 1
    cls = seg[mask].astype(np.int64)
    counts = np.bincount(node*num_classes+cls, minlength=node_num*num_classes).reshape(node_num, num_classes)

    return counts.argmax(axis=1).astype(np.int64)


def reconstruct_edges(parsing, target):
    node_num = int(parsing.max())
    counts = np.bincount(parsing.ravel(), minlength=node_num+1)[1:]
    trachea = np.bincount(parsing[target == 19].ravel(), minlength=node_num+1)[1:]
    root = int(np.argmax(trachea if trachea.any() else counts))
    adjacent = [set() for _ in range(node_num)]
    for axis in range(3):
        aa = [slice(None)]*3
        bb = [slice(None)]*3
        aa[axis] = slice(None, -1)
        bb[axis] = slice(1, None)
        a = parsing[tuple(aa)]
        b = parsing[tuple(bb)]
        mask = (a > 0) & (b > 0) & (a != b)
        if not np.any(mask): continue
        pairs = np.stack([a[mask].astype(np.int64)-1, b[mask].astype(np.int64)-1], axis=1)
        for u, v in np.unique(pairs, axis=0):
            adjacent[u].add(int(v))
            adjacent[v].add(int(u))

    edges = []
    visited = set()
    for start in [root]+[i for i in range(node_num) if i != root]:
        if start in visited: continue
        visited.add(start)
        queue = [start]
        for parent in queue:
            for child in sorted(adjacent[parent]):
                if child in visited: continue
                visited.add(child)
                edges.append((parent, child))
                queue.append(child)

    return np.asarray(edges, dtype=np.int64).reshape(-1, 2).T


def _class_path(c):
    if c == 19: return ('trachea',)
    if c in SEG_GROUP:
        side, lobe = SEG_GROUP[c]

        return ('trachea', side, lobe, c)

    return None


def _class_distances(num_classes=21):
    paths = [_class_path(i) for i in range(num_classes)]
    D = np.full((num_classes, num_classes), 6.0, dtype=np.float64)
    np.fill_diagonal(D, 0)
    for i, a in enumerate(paths):
        if a is None: continue
        for j, b in enumerate(paths):
            if b is None: continue
            shared = 0
            for x, y in zip(a, b):
                if x != y: break
                shared += 1
            D[i, j] = len(a)+len(b)-2*shared

    return D


CLASS_DISTANCE = _class_distances()


def graph_scores(prediction, target, parsing, edges=None, td_fill=50.0):
    node_num = int(parsing.max())
    if edges is None:
        edges = reconstruct_edges(parsing, target)
        graph_source = 'reconstructed'
    else:
        graph_source = 'provided'
        edges = np.asarray(edges, dtype=np.int64)
        if edges.shape[0] != 2: edges = edges.T
    pred_node = project_to_nodes(largest_component(prediction), parsing, node_num)
    gt_node = project_to_nodes(target, parsing, node_num)
    valid = gt_node != 0
    g = gt_node[valid]
    p = pred_node[valid]
    acc = float(np.mean(g == p)) if g.size else 0.0
    f1_class = {}
    for c in np.unique(g):
        tp = np.count_nonzero((g == c) & (p == c))
        fp = np.count_nonzero((g != c) & (p == c))
        fn = np.count_nonzero((g == c) & (p != c))
        precision = tp/(tp+fp) if tp+fp else 0.0
        recall = tp/(tp+fn) if tp+fn else 0.0
        f1_class[int(c)] = 2*precision*recall/(precision+recall) if precision+recall else 0.0
    f1 = float(np.mean(list(f1_class.values()))) if f1_class else 0.0
    graph = csr_matrix((np.ones(edges.shape[1]*2), (np.r_[edges[0], edges[1]], np.r_[edges[1], edges[0]])), shape=(node_num, node_num))
    spd = shortest_path(graph, directed=False, unweighted=True)
    idx = np.where(valid)[0]
    distances = np.full(idx.size, td_fill, dtype=np.float64)
    for k, i in enumerate(idx):
        match = np.where(gt_node == pred_node[i])[0]
        if match.size:
            d = np.min(spd[i, match])
            if np.isfinite(d): distances[k] = d
    td = float(np.mean(distances)) if distances.size else 0.0
    tacc = float(np.mean(1-CLASS_DISTANCE[p, g]/6)) if g.size else 0.0
    children = [[] for _ in range(node_num)]
    for a, b in edges.T: children[int(a)].append(int(b))
    good = 0
    total = 0
    for i in range(node_num):
        if pred_node[i] == 19: continue
        members = []
        stack = [i]
        while stack:
            j = stack.pop()
            members.append(j)
            stack.extend(children[j])
        if len(members) <= 1: continue
        total += 1
        good += int(np.all(pred_node[members] == pred_node[i]))
    sc = good/total if total else 0.0

    return {'ACC':acc, 'F1':f1, 'SC':sc, 'TD':td, 'TAcc':tacc, 'F1_class':f1_class,
            'graph_source':graph_source, 'graph_nodes':node_num, 'graph_valid_nodes':int(valid.sum())}
