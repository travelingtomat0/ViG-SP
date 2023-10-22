import os
import struct
import time
from ctypes import *
from typing import List, Tuple

import numpy as np
import numpy.ctypeslib

libfastgraph = CDLL(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "libfastgraph.so")
)


def store_labels(labels: np.ndarray, filename: str) -> None:
    global libfastgraph

    assert len(labels.shape) == 2
    assert labels.dtype == int

    libfastgraph.store_labels.argtypes = [
        POINTER(c_long), c_int, c_int, c_char_p]

    libfastgraph.store_labels(
        labels.ctypes.data_as(POINTER(c_long)),
        labels.shape[1],
        labels.shape[0],
        c_char_p(filename.encode("ascii")),
    )


def load_labels(filename: str) -> np.ndarray:
    global libfastgraph

    with open(filename, "rb") as f:
        width, height = struct.unpack("ii", f.read(8))

    labels = np.empty((height, width), np.int64)

    libfastgraph.load_labels.argtypes = [c_char_p, POINTER(c_long)]

    libfastgraph.load_labels(
        c_char_p(filename.encode("ascii")
                 ), labels.ctypes.data_as(POINTER(c_long))
    )

    return labels


def create_graph(
    labels: np.ndarray,
    image: np.ndarray,
    crop_x: int,
    crop_y: int,
    crop_width: int,
    crop_height: int,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    global libfastgraph

    assert len(labels.shape) == 2
    assert labels.dtype == int

    assert len(image.shape) == 3
    assert image.dtype == np.uint8
    assert image.shape[2] == 3
    assert image.shape[:-1] == labels.shape

    assert crop_width >= 0 and crop_height >= 0
    assert crop_x >= 0 and crop_x + crop_width <= labels.shape[1]
    assert crop_y >= 0 and crop_y + crop_height <= labels.shape[0]

    num_graphs = c_int()

    sz_edges = POINTER(c_int)()
    edges = POINTER(POINTER(c_long))()

    sz_features = POINTER(c_int)()
    features = POINTER(POINTER(c_uint8))()

    libfastgraph.create_graph.argtypes = [POINTER(c_long), POINTER(c_uint8), c_int, c_int, c_int, c_int, c_int, c_int, POINTER(
        c_int), POINTER(POINTER(c_int)), POINTER(POINTER(POINTER(c_long))), POINTER(POINTER(c_int)), POINTER(POINTER(POINTER(c_uint8)))]

    libfastgraph.create_graph(
        labels.ctypes.data_as(POINTER(c_long)),
        image.ctypes.data_as(POINTER(c_uint8)),
        labels.shape[1],
        labels.shape[0],
        crop_x,
        crop_y,
        crop_width,
        crop_height,
        byref(num_graphs),
        byref(sz_edges),
        byref(edges),
        byref(sz_features),
        byref(features),
    )

    num_graphs = num_graphs.value

    graph_edges = []
    graph_features = []

    for i in range(num_graphs):
        graph_edges.append(
            np.ctypeslib.as_array(edges[i], shape=(sz_edges[i] // 2, 2)).copy()
        )
        graph_features.append(
            np.ctypeslib.as_array(features[i], shape=(
                sz_features[i] // 3, 3)).copy()
        )
        libfastgraph.free(edges[i])
        libfastgraph.free(features[i])

    libfastgraph.free(sz_edges)
    libfastgraph.free(edges)
    libfastgraph.free(sz_features)
    libfastgraph.free(features)

    return graph_edges, graph_features


def benchmark(labels: np.ndarray, pixels: np.ndarray):
    print("Running create benchmark...")
    start = time.time()
    for i in range(1000):
        create_graph(
            labels,
            pixels,
            labels.shape[1] // 2 - 112,
            labels.shape[0] // 2 - 112,
            224,
            224,
        )
    end = time.time()
    print(f"Create: {(end - start):.2f} ms")

    print("Running store benchmark...")
    start = time.time()
    for i in range(1000):
        store_labels(labels, "benchmark.bin")
    end = time.time()
    print(f"Store: {(end - start):.2f} ms")

    print("Running load benchmark...")
    start = time.time()
    for i in range(1000):
        lbl = load_labels("benchmark.bin")
    end = time.time()
    print(f"Load: {(end - start):.2f} ms")

    assert np.all(labels == lbl)
