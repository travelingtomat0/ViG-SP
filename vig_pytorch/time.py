import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential as Seq
from datasets import load_dataset
import numpy as np
import torchvision.transforms as T
from skimage.segmentation import slic
from cuda_slic.slic import slic as cuda_slic


dataset = load_dataset('imagenet-1k', split="train", use_auth_token=True, streaming=True)
iterator = iter(dataset)
img  = next(iterator)
imgdata = img["image"]
np_image = np.array(imgdata)

transform = T.Resize((224,224))
image_resized = transform(imgdata)
np_image_resized = np.array(image_resized)


from timeit import default_timer as timer

start = timer()

for i in range(10):
    labels = cuda_slic(np_image_resized, n_segments=100, compactness=10.0, multichannel=True)

end = timer()
print(f"cuda slic {end - start}")
start = timer()


for i in range(10):


    labels = slic(np_image_resized, n_segments=100, compactness=10.0)

end = timer()
print(f"slic {end - start}")




def kakapisse(labels: np.ndarray, pixels: np.ndarray):
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

print("start stefan")
start = timer()

for i in range(100):

    res_i, res_t = kakapisse(labels=labels, pixels=np_image_resized)

end = timer()
print(f"stefan {end - start}")


print("start slow")
start = timer()

for i in range(100):


    actual_num_superpixels = labels.max()

    nodes = np.empty(shape=(actual_num_superpixels+1, 224*224,3)) 
    from_edges = [[]for i in range(actual_num_superpixels+1)]
    to_edges = [[] for i in range(actual_num_superpixels+1)]

    map_pixel_to_node = np.full((224,224), -1)
    counter_per_graph = np.zeros(actual_num_superpixels+1, dtype=int)

    for iy, ix in np.ndindex(labels.shape):
            curr_pixel = np_image_resized[iy, ix]
            curr_label = labels[iy, ix]
            if map_pixel_to_node[iy, ix]==-1:
                map_pixel_to_node[iy, ix] = counter_per_graph[curr_label]
                nodes[curr_label, counter_per_graph[curr_label], :] = curr_pixel
                counter_per_graph[curr_label]+=1

            if iy > 0:
                lower_neighbor_pixel = np_image_resized[iy-1,ix]
                lower_neighbor_label = labels[iy-1, ix]
                if lower_neighbor_label == curr_label:
                    if map_pixel_to_node[iy-1, ix]==-1:
                        map_pixel_to_node[iy-1, ix] = counter_per_graph[curr_label]
                        nodes[curr_label, counter_per_graph[curr_label], :] = lower_neighbor_pixel
                        counter_per_graph[curr_label]+=1
                    from_edges[curr_label].append(map_pixel_to_node[iy, ix])
                    to_edges[curr_label].append(map_pixel_to_node[iy-1, ix])

            if ix > 0:
                left_neighbor_pixel = np_image_resized[iy,ix-1]
                left_neighbor_label = labels[iy, ix-1]
                if left_neighbor_label == curr_label:
                    if map_pixel_to_node[iy, ix-1]==-1:
                        map_pixel_to_node[iy, ix-1] = counter_per_graph[curr_label]
                        nodes[curr_label, counter_per_graph[curr_label], :] = left_neighbor_pixel
                        counter_per_graph[curr_label]+=1
                    from_edges[curr_label].append(map_pixel_to_node[iy, ix])
                    to_edges[curr_label].append(map_pixel_to_node[iy, ix-1])



    x=np.array(nodes[1, :counter_per_graph[1]])
    edge_t = np.stack((from_edges[1], to_edges[1]))

end = timer()

print(f"slow {end - start}")

