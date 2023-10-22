import torch
import numpy as np


def gen_graph_from_labels(labels: np.ndarray, pixels: np.ndarray):
    # this function does not handle the case when there is one lonely pixel for a label
    
    assert len(labels.shape) == 2

    num_labels = labels.max()

    result_tuples = []
    result_values = []

    w1 = (labels[:, 1:] == labels[:, :-1])
    w2 = (labels[1:, :] == labels[:-1, :])
    for i in range(1, num_labels + 1):
        mi = labels == i

        m1 = w1 & mi[:, 1:]
        v1 = np.ravel_multi_index(
            np.argwhere(m1).T,
            labels.shape,
        )
        m2 = w2 & mi[1:, :]
        v2 = np.ravel_multi_index(
            np.argwhere(m2).T,
            labels.shape,
        )

        tuples = np.concatenate(
            (np.vstack((v1, v1 + 1)).T, np.vstack((v2, v2 + labels.shape[1])).T)
        )

        _, inverse_mapping = np.unique(tuples, return_inverse=True)
        remapped_tuples = inverse_mapping.reshape(tuples.shape)
        corresponding_pixels = pixels[mi]

        result_tuples.append(remapped_tuples)
        result_values.append(corresponding_pixels)

    return result_tuples, result_values
