# Dragonn

## Overview

Dragonn is a randomized hashing algorithm for gradient sparsification in data-parallel distributed training to minimize the compression overhead. 
DRAGONN can significantly reduce the compression time by up to 70% compared to state-of-the-art GS approaches, and achieve up to 3.52x speedup in total training throughput.


## Dependencies

### install dependencies
```shell script
pip install -r requirements

# install Dragonn
cd extensions/cuda
python setup.py install
```

### download pretrain weights

```shell script
# ViT
wget https://storage.googleapis.com/vit_models/imagenet21k/ViT-B_16.npz

# MLP-Mixer
wget https://storage.googleapis.com/mixer_models/imagenet1k/Mixer-B_16.npz

# XML
gdown https://drive.google.com/uc?id=14g1ZG1S_Emq2Hu0MBZYKccMYJL1QzI6l
```

## Run 

### Examples to run ViT
```shell script
# run ViT on 4 GPUs without compresion. The dataset is cifar10
horovodrun -np 4 /usr/bin/python3.7 main_vit.py --pcie --lr 1e-5 --batch_size 32 --epochs 1 --print_every 10 --pretrained --dataset cifar10

# run ViT on 8 GPUs with DGC as the compressor and the compression ratio is 0.01
horovodrun -np 8 /usr/bin/python3.7 main_vit.py --pcie --compress --compressor dgc --memory none --comm allgather --compress-ratio 0.01 --lr 1e-5 --batch_size 32 --epochs 1 --print_every 10 --pretrained --dataset cifar10
```

### Examples to run MLP-Mixer
```shell script
# run MLP-Mixer on 8 GPUs with DRAGONN as the compressor and the compression ratio is 0.001
horovodrun -np 8 python3 main_mixer.py --pcie --compress --compressor atopk --memory none --comm allgather --compress-ratio 1e-3 --lr 1e-5 --batch_size 32 --batching --threshold 204800 --epochs 1 --print_every 1 --pretrained --dataset cifar10
```

**Note**: make sure the directory of dataset and pretrained model is correct.

### Training with multiple machines
Refer to [Horovod](https://horovod.readthedocs.io/en/stable/running_include.html) to run Dragonn on multiple machines.

## How to run Dragonn in your code
It is easy to apply Dragonn to your customized training code.
Let's take [main_mixer.py](./main_mixer.py) as an example to showcase how to add a few lines of code to enable Dragonn.

```shell script
# Line 24: import required Draggon functions 
from mergeComp_dl.torch.helper import add_parser_arguments, wrap_compress_optimizer

# Line 120: add Dragonn specified arguments, such as compression algorithms and compression ration
parser = add_parser_arguments(parser)

# Line 202: wrap the optimizer with Dragonn's optimizer to support compression
optimizer = wrap_compress_optimizer(model, optimizer, args)
```
