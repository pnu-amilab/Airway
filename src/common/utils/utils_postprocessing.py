import numpy as np
import SimpleITK as sitk
from scipy import ndimage


def postprocess_threshold(probability, threshold):
    prediction = (probability > threshold).astype(np.uint8)
    prediction_image = sitk.GetImageFromArray(prediction)
    prediction_image = sitk.BinaryFillhole(prediction_image, fullyConnected=True)

    return sitk.GetArrayFromImage(prediction_image).astype(np.uint8)


def largest_connected_component_26(prediction):
    labels, n_label = ndimage.label(prediction > 0, structure=ndimage.generate_binary_structure(3, 3))
    if n_label == 0: return np.zeros_like(prediction, dtype=np.uint8)
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0

    return (labels == sizes.argmax()).astype(np.uint8)
