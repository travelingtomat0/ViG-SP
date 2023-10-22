# ViG-SP
This repository contains an ImageNet-1k Classifier experimenting with a segmentation-driven approach to image classification. The model was built during the Distributed Laboratory Course and experimented with the applicability of irregular image segmentation techniques (such as SLIC) for image classification. The backbone of the model is inspired by the Vision GNN paper and builds on graph neural networks. We added a custom feature extraction method combining a CNN (U-Net architecture) with GNN's for message passing between nodes.

The final model only managed to acheive a top-1 accuracy of 68% on ImageNet-1k, although results on ImageNet-100 were promising suring model development.

To evaluate the model call:

```
python -m torch.distributed.launch --nproc_per_node=<#GPUs> ~/vig_pytorch/train_copy.py <Path to Imagenet> --model sp_vig -b 22 -j <#GPUs> --segments 196 --num-classes 1000 --resume <Path to model> --pretrain_path <Path to model> --evaluate
```
* The _nproc_per_node_ argument as _j_ need to be set with the number of GPUs used during evaluation.
* The _Path to Imagenet_ should look something similar to this: ~/imagenet-1k/ILSVRC/Data/CLS-LOC/
* The _Path to Model_ should look something like this: ~/imagenet1k_slic196/checkpoint-XX.pth.tar

To train from scratch adapt the _scripts/6_final_from_scratch.sh_ file with the corresponding file paths in your file system. Then use sbatch to call
```
./6_final_from_scratch.sh <num workers>
```

To resume from a checkpoint use
```
./6_final_resume.sh <path to the .pth.tar file of the checkpoint> <num Workers>
```

Make sure you have connected a wandb authorization token and adapted the logging (in train_copy.py) when running the code.
