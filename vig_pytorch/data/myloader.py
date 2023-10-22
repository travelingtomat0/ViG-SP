# 2021.06.15-Changed for implementation of TNT model
#            Huawei Technologies Co., Ltd. <foss@huawei.com>
""" Loader Factory, Fast Collate, CUDA Prefetcher

Prefetcher and Fast Collate inspired by NVIDIA APEX example at
https://github.com/NVIDIA/apex/commit/d5e2bb4bdeedd27b1dfaf5bb2b24d6c000dee9be#diff-cf86c282ff7fba81fad27a559379d5bf

Hacked together by / Copyright 2020 Ross Wightman
"""

import torch.utils.data
import torch.distributed as dist
import numpy as np

from timm.data.transforms_factory import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data.distributed_sampler import OrderedDistributedSampler
from timm.data.random_erasing import RandomErasing
from timm.data.mixup import FastCollateMixup
from timm.data.loader import fast_collate, PrefetchLoader, MultiEpochsDataLoader

from .rasampler import RASampler

import skimage
import skimage.graph as graph
from skimage.util import img_as_float

from sp_graph_utils import gen_graph_from_labels

from fast_graph import create_graph

import time

import torch
from torch_geometric.transforms import ToUndirected
from torch_geometric.data import Data
from torch_geometric.data import Batch


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def create_loader(
        dataset,
        input_size,
        batch_size,
        is_training=False,
        use_prefetcher=True,
        no_aug=False,
        re_prob=0.,
        re_mode='const',
        re_count=1,
        re_split=False,
        scale=None,
        ratio=None,
        hflip=0.5,
        vflip=0.,
        color_jitter=0.4,
        auto_augment=None,
        num_aug_splits=0,
        interpolation='bilinear',
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD,
        num_workers=1,
        distributed=False,
        crop_pct=None,
        collate_fn=None,
        pin_memory=False,
        fp16=False,
        tf_preprocessing=False,
        use_multi_epochs_loader=False,
        repeated_aug=False,
        slic_compactness=10.0,
        slic_segments=100
):
    re_num_splits = 0
    if re_split:
        # apply RE to second half of batch if no aug split otherwise line up with aug split
        re_num_splits = num_aug_splits or 2
    dataset.transform = create_transform(
        input_size,
        is_training=is_training,
        use_prefetcher=use_prefetcher,
        no_aug=no_aug,
        scale=scale,
        ratio=ratio,
        hflip=hflip,
        vflip=vflip,
        color_jitter=color_jitter,
        auto_augment=auto_augment,
        interpolation=interpolation,
        mean=mean,
        std=std,
        crop_pct=crop_pct,
        tf_preprocessing=tf_preprocessing,
        re_prob=re_prob,
        re_mode=re_mode,
        re_count=re_count,
        re_num_splits=re_num_splits,
        separate=num_aug_splits > 0
    )



    class test_transform:

        def __call__(self, pic):
            print(f"test transform with {type(pic)}")
            return pic

        def __repr__(self):
            return self.__class__.__name__ + '()'


    class break_transform:

        def __call__(self, pic):
            print(f"break transform with {type(pic)}")
            return torch.zeros(size=(10,10))

        def __repr__(self):
            return self.__class__.__name__ + '()'

    class float_transform:

        def __call__(self, pic):
            print(f"type of image is {type(pic)}")
            print(f"dtype of img is {pic.dtype}")
            pic = np.swapaxes(pic, 0, 2)
            np_img_float64 = img_as_float(pic)
            np_img_uint8=pic
            return [np_img_float64, np_img_uint8]

        def __repr__(self):
            return self.__class__.__name__ + '()'

    class slic_transform:
        def __init__(self, n_segments=100, compactness=10.0):
            self.n_segments = n_segments
            self.compactness = compactness

        def __call__(self, np_img):
            #np_img = np.swapaxes(np_img, 0, 2)

            #print(f"img shape {np_img.shape}")

            #start = time.time()
            # import os
            # path = '/itet-stor/mateodi/net_scratch/no_preprocessing_wandb_conv/mod_vig_pytorch/vig_pytorch/sample.npy'
            # if not os.path.exists(path) :
            #     np.save(file=path, arr=np_img)

            # labels = self.create_labeled_patches(np_img)
            # TODO: enable BELOW IF SLIC SEGMENTATION ENABLED!!
            
            labels = skimage.segmentation.slic(np_img, n_segments=self.n_segments, compactness=self.compactness, enforce_connectivity=True, channel_axis=0)
            #labels = self.merger(labels, self.n_segments, np_img)
            if np.max(labels) > 196:
                labels = self.merger(labels, self.n_segments, np_img)
            # pathl = '/itet-stor/mateodi/net_scratch/no_preprocessing_wandb_conv/mod_vig_pytorch/vig_pytorch/labels.npy'
            # if not os.path.exists(pathl) :
            #     np.save(file=pathl, arr=labels)
            #print(f"lables shape {labels.shape}")
            #end = time.time()
            #print(f"labels took {end-start}")

            return [np_img, labels]

        #print(f"num segments before {num_segments}, using unique: {len(np.unique(segments))}")
        #print(f"num segments after {num_segments}, using unique: {len(np.unique(segments))}")
        def merger(self, labels, n_segments, image):
            image = np.transpose(image, (1, 2, 0))
            # Create the region adjacency graph (RAG) based on mean color similarity
            rag = graph.rag_mean_color(image, labels)
            for n in rag.nodes():
                if 'pixel count' not in rag.nodes[n]:
                    rag.nodes[n]['pixel count'] = 0

            # Merge superpixels until we have exactly n_segments regions
            print(f"num segments before {rag.number_of_nodes()}")
            while rag.number_of_nodes() > n_segments:
                # Find the edge with the smallest mean color difference
                min_edge = min(rag.edges(data=True), key=lambda x: x[2]['weight'])
                merge_node1, merge_node2 = min_edge[0], min_edge[1]

                # Merge the two nodes into a single node
                rag.merge_nodes(merge_node1, merge_node2)
            print("Done")
            # Obtain the final merged segments as a 1D array
            final_segments = np.zeros_like(labels)
            for i, (segment, data) in enumerate(rag.nodes(data=True)):
                final_segments[labels == segment] = i

            return final_segments

        def create_labeled_patches(self, np_img):
            height, width = np_img.shape[1:]
            patch_size = 16
            num_patches_h = height // patch_size
            num_patches_w = width // patch_size

            labeled_patches = np.zeros((num_patches_h, num_patches_w), dtype=int)
            current_label = 0

            for i in range(num_patches_h):
                for j in range(num_patches_w):
                    labeled_patches[i, j] = current_label
                    current_label += 1

            # Repeat the labels for each patch
            labeled_patches = np.tile(labeled_patches, (patch_size, patch_size))

            return labeled_patches

        def __repr__(self):
            return self.__class__.__name__ + '()'

    class subgraphs_transform:
        def __init__(self):
            self.transform = ToUndirected()
            m_grid = np.mgrid[0:224, 0:224]
            self.indices_array = np.stack([m_grid[0], m_grid[1], np.zeros((224,224))], axis=-1).astype(np.uint8)

        def __call__(self, data):
            np_img  = data[0] #dtype is uint8
            labels = data[1]
            # actual_num_superpixels = labels.max()

            #np_i_s = np_img.shape
            #x_img = np.swapaxes(np_img, 0, 2)

            #print("shapes")
            #print(x_img.shape)
            #print(labels.shape)
            # start = time.time()
            result_tuples, result_values = create_graph(labels=labels, image = self.indices_array, crop_x = 0, crop_y = 0, crop_width=224, crop_height=224)
            result_tuples_undir = [np.concatenate((r_t, np.flip(r_t, axis=1))) for r_t in result_tuples]
            
            #result_tuples, result_indices = create_graph_mod(labels=labels, image=np_image)
            # end = time.time()
            # print(f"create graph took {end-start}")

            #assert(np_i_s == np_img.shape)
            # output_graphs = []

            # for i in range(actual_num_superpixels):
            #     values = result_values[i]
            #     tuples = result_tuples[i]

            #     edge_index = torch.from_numpy(tuples.T)
            #     x = torch.from_numpy(values).float()

            #     data = Data(x=x, edge_index=edge_index)
            #     data = self.transform(data)
            #     #data.validate(raise_on_error=True)

            #     output_graphs.append(data)


            #print("created_graphs")
            # start = time.time()
            # index_array = torch.empty(224*224, dtype=torch.long)
            # #num_pixels_list = torch.empty(labels.max())
            # num_pixels_list = []
            # count = 0
            # idx_ia = 0
            # idx_npi = 0

            # for sp in result_values:
            #     num_pixels = len(sp)
            #     #num_pixels_list[idx_npi] = count + num_pixels
            #     #num_pixels_list.append(count+ num_pixels)
            #     #count += num_pixels
            #     num_pixels_list.append(num_pixels)
            #     idx_npi += 1
            #     for pix in sp:
            #         y_ind = pix[0]
            #         x_ind = pix[1]
            #         index_array[idx_ia] = y_ind*224 + x_ind
            #         idx_ia += 1
 
            # #print(f"len of res tuples {len(result_tuples)}")
            # #print(f"type of f {type(result_tuples[0])}")
            # end = time.time()
            # print(f"index array took {end-start}")

            # start = time.time()
            
            # index_arr = [pix[0]*224 + pix[1] for sp in result_values for pix in sp]
            # num_p = [len(sp) for sp in result_values]
            # index_arr = torch.as_tensor(index_arr)

            
            # end = time.time()

            # print("mateoschhh")
            # print(end - start)

            # start = time.time()

            index_arr2 = np.concatenate([np.ravel_multi_index(sp.T, (224, 224, 1)) for sp in result_values])
            num_p2 = [len(sp) for sp in result_values]
            index_arr2 = torch.as_tensor(index_arr2)

            # end = time.time()
            # print("schdefan")
            # print(end - start)


            # end = time.time()
            # print(f"index array new took {end-start}")

            # assert num_pixels_list == num_p
            # eq = torch.eq(index_arr, index_arr2)
            # assert torch.all(eq==True) == True
            # print("asserted")


            return [np_img, result_tuples_undir, index_arr2, num_p2]

        def __repr__(self):
            return self.__class__.__name__ + '()'


    dataset.transform.transforms.append(slic_transform(slic_segments, slic_compactness))
    dataset.transform.transforms.append(subgraphs_transform())

    print("created transform")



    #APPEND cupy slic
    #APPEND graphs creation
    

    print(dataset.transform)




    sampler = None
    if distributed:
        if is_training:
            if repeated_aug:
                print('using repeated_aug')
                num_tasks = get_world_size()
                global_rank = get_rank()
                sampler = RASampler(
                    dataset, num_replicas=num_tasks, rank=global_rank, shuffle=True
                )
            else:
                sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        else:
            # This will add extra duplicate entries to result in equal num
            # of samples per-process, will slightly alter validation results
            sampler = OrderedDistributedSampler(dataset)
    else:
        if is_training and repeated_aug:
            print('using repeated_aug')
            num_tasks = get_world_size()
            global_rank = get_rank()
            sampler = RASampler(
                    dataset, num_replicas=num_tasks, rank=global_rank, shuffle=True
                )

    #if collate_fn is None:
    #    collate_fn = fast_collate if use_prefetcher else torch.utils.data.dataloader.default_collate


    def fast_collate_mod(batch):
        #print("fast collate start")
        """ A fast collation function optimized for uint8 images (np array or torch) and int64 targets (labels)"""
        #print("FAST COLLATE MOD START")
        #print(f"type of elem is {type(batch[0])}")
        #print(len(batch[0][0]))
        #print(f"shape of sample image is {batch[0][0][0].shape}")
        #print(batch[0][1])

        batch_size = len(batch)
        targets = torch.tensor([b[1] for b in batch], dtype=torch.int64)
        assert len(targets) == batch_size

        img_tensor = torch.zeros((batch_size,3, 224,224), dtype=torch.uint8)
        for i in range(batch_size):
            img_tensor[i] += torch.from_numpy(batch[i][0][0])

        result_tuples = [b[0][1] for b in batch]

        #print(f"collate result tuples len {len(result_tuples)} type {(type(result_tuples[0][0]))}")

        index_arrays = [b[0][2] + i * 224*224 for i, b in enumerate(batch)]
        #num_pixel_lists = [b[0][3] + i * 224 * 224 for i, b in enumerate(batch)]

        catted_index_arrays = torch.cat(index_arrays)


        #num_pixel_lists = [ x + i * 224 * 224 for i, b in enumerate(batch) for x in b[0][3]]
        num_pixel_lists = [ x  for b in batch for x in b[0][3]]

        #print(f"len of num pixel list is {len(num_pixel_lists)}")

        num_sp_list = [len(b) for b in result_tuples]
        #print(f"num sp list {num_sp_list}")


        #print("collated")
        return [img_tensor, result_tuples, catted_index_arrays, num_pixel_lists, num_sp_list], targets


    def collate_pyg_graphs_fn(batch):



        data_list = [subgraph for tuple_graph_target in batch for subgraph in tuple_graph_target[0]]
        num_sp_list = [len(tuple_graph_target[0]) for tuple_graph_target in batch]

        graph_batch = Batch.from_data_list(data_list)
        graph_batch.num_sp_list = num_sp_list

        target_list = torch.tensor([tuple_graph_target[1] for tuple_graph_target in batch])

        #print("collated batch")

        return graph_batch, target_list



    #collate_fn = torch.utils.data.dataloader.default_collate
    collate_fn = fast_collate_mod

    loader_class = torch.utils.data.DataLoader


    if use_multi_epochs_loader:
        loader_class = MultiEpochsDataLoader

    
    print(f"loader class is {loader_class}")
    loader = loader_class(
        dataset,
        batch_size=batch_size,
        shuffle=sampler is None and is_training,
        num_workers=num_workers,
        sampler=sampler,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        drop_last=is_training,
    )



    if use_prefetcher:
        print("prefetcher is used")
        prefetch_re_prob = re_prob if is_training and not no_aug else 0.
        loader = PrefetchLoader(
            loader,
            mean=mean,
            std=std,
            fp16=fp16,
            re_prob=prefetch_re_prob,
            re_mode=re_mode,
            re_count=re_count,
            re_num_splits=re_num_splits
        )

    return loader
