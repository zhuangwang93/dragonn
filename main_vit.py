import os
import torch
import numpy as np
import torch.optim as optim
from vit import *
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler
import torch.nn as nn 
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.utils.data as data_utils
import math
import horovod.torch as hvd
from time import time as time_
import sys
import argparse
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
import tarfile
import time

from mergeComp_dl.torch.helper import add_parser_arguments, wrap_compress_optimizer
from time import sleep


# CONFIGS = {
#     'Mixer-B_16': get_mixer_b16_config(),
#     'Mixer-L_16': get_mixer_l16_config(),
#     'Mixer-B_16-21k': get_mixer_b16_config(),
#     'Mixer-L_16-21k': get_mixer_l16_config()
# }

def get_loader(dataset_name,img_size,train_batch_size,eval_batch_size):
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop((img_size, img_size), scale=(0.05, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    transform_test = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    # assert dataset_name=="cifar10"
    if dataset_name=="cifar10":
        trainset = datasets.CIFAR10(root="/home/zx22/data",
                                    train=True,
                                    download=True,
                                    transform=transform_train)
        testset = datasets.CIFAR10(root="/home/zx22/data",
                                    train=False,
                                    download=True,
                                    transform=transform_test) 
    elif dataset_name=="cifar100":
        trainset = datasets.CIFAR100(root="/home/zx22/data",
                                    train=True,
                                    download=True,
                                    transform=transform_train)
        testset = datasets.CIFAR100(root="/home/zx22/data",
                                    train=False,
                                    download=True,
                                    transform=transform_test) 

    # train_sampler = RandomSampler(trainset) 
    train_sampler = DistributedSampler(
        trainset, num_replicas=hvd.size(), rank=hvd.rank())
    # test_sampler = SequentialSampler(testset)
    test_sampler = DistributedSampler(
        testset, num_replicas=hvd.size(), rank=hvd.rank())
    train_loader = DataLoader(trainset,
                              sampler=train_sampler,
                              batch_size=train_batch_size,
                              num_workers=4,
                              pin_memory=True)
    test_loader = DataLoader(testset,
                             sampler=test_sampler,
                             batch_size=eval_batch_size,
                             num_workers=4,
                             pin_memory=True) 

    return train_loader, test_loader



parser = argparse.ArgumentParser(description='PyTorch Synthetic Benchmark',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser = add_parser_arguments(parser)
parser.add_argument('--dataset', type = str, default = "cifar10")
parser.add_argument('--net_type', type = str, default = "ViT-B_16")
parser.add_argument('--batch_size', type=int, default=32, metavar='N',help='input batch size for training (default: 1)')
parser.add_argument('--test_size', type=int, default=64, metavar='N',help='input batch size for testing (default: 1)')
parser.add_argument('--epochs', type=int, default=20, metavar='N',help='number of epochs to train (default: 1)')
parser.add_argument('--lr', type=float, default= '0.01', metavar='LR', help='learning rate (default: 0.1)')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
parser.add_argument('--print_every', type=int, default=200, metavar='N',help='print log interval')
parser.add_argument('--max_steps', type=int, default=2500000000, metavar='N',help='maximum steps to terminate the training')
parser.add_argument('--pretrained', action='store_true')
args = parser.parse_args()
torch.manual_seed(args.seed)


def main():
    hvd.init()
    torch.cuda.set_device(hvd.local_rank())
    cudnn.benchmark = True
    work_dir = "logs/vit/"+args.dataset
    if not os.path.exists(work_dir) and hvd.rank() == 0:
        os.makedirs(work_dir)
    dir_path= os.path.join(work_dir, str(hvd.size()), time.strftime('%Y%m%d-%H%M%S'))
    
    if not os.path.exists(dir_path) and hvd.rank() == 0:
        os.makedirs(dir_path)
    if hvd.rank() == 0:
        print('Experiment dir : {}'.format(dir_path))
        with open(os.path.join(dir_path,"param.txt"),"w") as f:
            f.write('    - GPUs : {}\n'.format(hvd.size()))
            for k, v in args.__dict__.items():
                f.write('    - {} : {}\n'.format(k, v))

    trainlogfile=os.path.join(dir_path,"train.txt")
    testlogfile=os.path.join(dir_path,"test.txt")
    # model_checkfile=os.path.join(dir_path,"model.pth.tar")

    if args.net_type=="ViT-B_16":
        config = CONFIGS[args.net_type]

    if args.dataset=="cifar10":
        num_classes=10
    elif args.dataset=="cifar100":
        num_classes=100

    model=VisionTransformer(config, num_classes=num_classes, zero_head=True)
    # model=VisionTransformer(config)

    if args.pretrained and args.net_type=="ViT-B_16":
        pretrained_dir="/home/zx22/data/pretrain/ViT-B_16.npz"
        model.load_from(np.load(pretrained_dir))
    model=model.cuda()
    num_tensors, tensor_sizes = count_tensors(model)
    num_params = count_parameters(model)
    if hvd.rank() == 0:
        # print(model)
        print("Number of tensors: {}, Number of parameters: {} M".format(num_tensors, num_params))

    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-2)
    hvd.broadcast_parameters(model.state_dict(), root_rank=0)
    hvd.broadcast_optimizer_state(optimizer, root_rank=0)
    optimizer = wrap_compress_optimizer(model, optimizer, args)

    train_loader, test_loader = get_loader(args.dataset, 224,args.batch_size, args.test_size)
    for epoch in range(args.epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        time_passed = 0
        # comm_time_passed=0
        zero = torch.tensor([0]).cuda()
        for i, data in enumerate(train_loader, 0):
            if i > args.max_steps:
                break
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs=inputs.cuda()
            labels=labels.cuda()

            hvd.allreduce(zero, name="test")
            ### start optmization
            torch.cuda.synchronize()
            start=time.time()

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)[0]
            loss = criterion(outputs, labels)
            loss.backward()
            # torch.cuda.synchronize()
            # comm_time = time_()
            # time_passed+=comm_time-start
            optimizer.step()
            torch.cuda.synchronize()
            end=time.time()
            time_passed+=end-start
            # comm_time_passed+=end - comm_time

            # print statistics
            running_loss += loss.item()
            if i % args.print_every == args.print_every-1:    # print every 2000 mini-batches
                if hvd.rank() == 0:
                    speed = args.print_every / time_passed * args.batch_size * hvd.size() 
                    print('[%d, %5d] loss: %.3f,Epoch time: %.2f, speed: %.2fimg/s' %(epoch + 1, i + 1, running_loss / args.print_every, time_passed, speed), file = open(trainlogfile, "a"))
                    print('[%d, %5d] loss: %.3f,Epoch time: %.2f, speed: %.2fimg/s' %(epoch + 1, i + 1, running_loss / args.print_every, time_passed, speed))
                    # print('' % (time_passed))
                    # print('Epoch time: %.2f, communication time  %.2f' % (time_passed,comm_time_passed))
                    
                    # comm_time_passed=0
                time_passed=0
                running_loss = 0.0

                correct = 0
                total = 0
                # since we're not training, we don't need to calculate the gradients for our outputs
                with torch.no_grad():
                    for data in test_loader:
                        images, labels = data
                        images=images.cuda()
                        labels=labels.cuda()
                        # calculate outputs by running images through the network
                        outputs = model(images)[0]
                        # the class with the highest energy is what we choose as prediction
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()
                if hvd.rank() == 0:
                    print('Accuracy: %.2f %%' % (100 * correct / total), file = open(testlogfile, "a"))
                    print('Accuracy: %.2f %%' % (100 * correct / total))
        if hvd.rank() == 0:
            print('[%d, %5d] loss: %.3f,Epoch time: %.2f' %(epoch + 1, i + 1, running_loss / args.print_every,time_passed), file = open(trainlogfile, "a"))
            print('[%d, %5d] loss: %.3f,Epoch time: %.2f' %(epoch + 1, i + 1, running_loss / args.print_every,time_passed))
            # print('' % (time_passed))
            # print('Epoch time: %.2f, communication time  %.2f' % (time_passed,comm_time_passed))
            
            # comm_time_passed=0
        time_passed=0
        running_loss = 0.0

    print('Finished Training')


if __name__ == '__main__':
    main()
    sleep(1)
    os.system("pkill -9 python3")
