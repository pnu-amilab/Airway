def make_centers(length, size_patch, overlap):
    stride = size_patch-overlap
    if stride <= 0: raise ValueError('overlap must be smaller than size_patch')

    window = size_patch//2
    last = length-window
    centers = list(range(window, last+1, stride))
    if len(centers) == 0: centers = [last]
    elif centers[-1] != last: centers.append(last)

    return centers
