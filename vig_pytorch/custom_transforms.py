import torchvision.transforms.functional as TF
import random

def my_segmentation_transforms(image, segmentation):
    if random.random() > 0.5:
        angle = random.randint(-30, 30)
        image = TF.rotate(image, angle)
        segmentation = TF.rotate(segmentation, angle)
    # more transforms ...
    return image, segmentation

def slic_transform(image_t):
    dx = to_dlpack(image_t)
    cx = cupy.from_dlpack(dx)
    labels = mod_cuda_slic(image=cx, n_segments=100, multichannel=True)

    labels_t = torch.from_numpy(labels)

    return image, segmentation