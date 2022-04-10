import torch
import os
import numpy as np 
import torch.nn as nn 
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.utils.data as data_utils
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler
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
from lazy_parser import MultiLabelDataset, ValidDataset
from adam_base import Adam

from mergeComp_dl.torch.helper import add_parser_arguments, wrap_compress_optimizer
from time import sleep
# +

#### run command
# horovodrun -np 4 python main_last.py --compress --compressor atopk --memory none --comm allgather --compress-ratio 1e-3 --test_every 5 --epoch 50 --dataset wiki10 --hidden_dim 4096 --lr 1e-3 --batching
# horovodrun -np 4 python main_last.py --compress --compressor dgc --memory none --comm allgather --compress-ratio 1e-3 --test_every 5 --epoch 50 --dataset wiki10 --hidden_dim 4096 --lr 1e-3
# horovodrun -np 4 python main_last.py --compress --compressor atopk --memory none --comm allgather --compress-ratio 1e-3 --test_every 10 --epoch 20 --dataset eurlex4k --hidden_dim 10240 --lr 1e-2 --batching
# horovodrun -np 4 python main_last.py --compress --compressor dgc --memory none --comm allgather --compress-ratio 1e-3 --test_every 10 --epoch 20 --dataset eurlex4k --hidden_dim 10240 --lr 1e-2

DATASET = ['wiki10', 'amz13k','amz630k','amz3m','deli','wiki300', 'eurlex4k', 'text8_sub']

DATAPATH = "/home/zx22/data/pretrain"


parser = argparse.ArgumentParser(description='PyTorch Synthetic Benchmark',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser = add_parser_arguments(parser)
parser.add_argument('--dataset', type = str, default = "wiki10", choices = DATASET)
parser.add_argument('--batch_size', type=int, default=256, metavar='N',help='input batch size for training (default: 1)')
parser.add_argument('--test_size', type=int, default=256, metavar='N',help='input batch size for testing (default: 1)')
parser.add_argument('--hidden_dim', type=int, default=4096, metavar='N',help='input batch size for testing (default: 1)')
parser.add_argument('--epochs', type=int, default=30, metavar='N',help='number of epochs to train (default: 1)')
parser.add_argument('--lr', type=float, default= '0.001', metavar='LR', help='learning rate (default: 0.1)')
parser.add_argument('--test_every', type = int, default = 10)
args = parser.parse_args()
torch.manual_seed(0)


def count_tensors(model):
    cnt = 0
    sizes = []
    for p in model.parameters():
        if p.requires_grad:
            cnt += 1
            sizes.append(p.numel())
    
    return cnt, sizes


def count_parameters(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params/1000000


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    print("saved better model")


class Net(nn.Module):
    def __init__(self, OUT):
        super(Net, self).__init__()
        self.OUT = OUT
        self.smax = nn.Linear(args.hidden_dim, self.OUT+1, bias=False)
    def forward(self, x):
        return self.smax(x)


def train(args, model, loader, test_loader, optimizer, epoch,vallog):
    model.train()
    model.optimizer = optimizer

    time_passed = 0
    zero = torch.tensor([0]).cuda()
    for batch_idx, (data, labels) in enumerate(loader):
        step = epoch * len(loader)+ batch_idx
        # print(step)
        batch_size = labels.size(0)
        # print(batch_size)
        targets = Variable(torch.zeros(batch_size, model.OUT+1))
        # print(targets.shape)
        sizes = torch.sum(labels != -1, dim=1).int()
        for bdx in range(batch_size):
            num_labels = sizes[bdx].item()
            value = 1. / float(num_labels)
            for ldx in range(num_labels):
                targets[bdx, labels[bdx, ldx]] = value

        data, targets = data.cuda(), targets.cuda()

        # print(data.shape)
        hvd.allreduce(zero, name="test")
        ### start optmization
        torch.cuda.synchronize()
        start=time.time()
        optimizer.zero_grad()
        output_dist = F.log_softmax(model(data), dim=-1)
        loss = F.kl_div(output_dist, targets, reduction='sum') / batch_size

        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
        end=time.time()

        time_passed+=end-start
        # print(end-start)


        #print evaluate accuracy
        if batch_idx % args.test_every  == args.test_every -1 :
            # time_passed = time.time() - start_time
            top1_acc, top5_acc = evaluate(epoch, batch_idx, args,model,test_loader,k = 5,training = False)
            if hvd.rank() == 0:
                print("Step:{:.4f}, Time:{:.4f}, p@1:{:.4f}, p@5:{:.4f}".format(step, time_passed,top1_acc, top5_acc))
                print("Step:{:.4f}, Time:{:.4f}, p@1:{:.4f}, p@5:{:.4f}".format(step, time_passed,top1_acc, top5_acc), file = open(vallog, "a"))
            time_passed=0


# +
def evaluate(epoch, iter, args, model, loader, k=1,training=False,best=False):
    model.eval()
    N = 0.
    correct = 0.
    top1 = 0.

    full_data_list=[]
    full_label_list=[]
    with torch.no_grad():
        for batch_idx, (data,labels) in enumerate(loader):
            batch_size, ml = labels.size()
            sizes = torch.sum(labels != -1, dim=1)

            data = data.cuda()

            output = model.forward(data).cpu()
            if best:
                out_mat=output.data.cpu().numpy()
                full_data_list.append(out_mat)

                lab_mat=labels.data.cpu().numpy()
                full_label_list.append(lab_mat)

            values, indices = torch.topk(output, k=k, dim=1)
            for bdx in range(batch_size):
                label_set = labels[bdx,:sizes[bdx]].numpy().tolist()
                for idx in range(k):
                    N += 1
                    if indices[bdx, idx].item() in label_set:
                        correct+=1.
                        if idx == 0:
                            top1 += 1

            if( batch_idx == 50 and training ):
                break

    if best:
        final_data=np.concatenate(full_data_list)
        final_label=np.concatenate(full_label_list)
    top1_acc = top1/N * k
    topk_acc = correct/N
    model.train()
    return top1_acc, topk_acc


# -

def main():
    hvd.init()
    torch.cuda.set_device(hvd.local_rank())
    cudnn.benchmark = True
    if not os.path.exists("logs/xml/") and hvd.rank() == 0:
        os.makedirs("logs/xml/")
    work_dir = "logs/xml/"+args.dataset
    if not os.path.exists(work_dir) and hvd.rank() == 0:
        os.makedirs(work_dir)
    dir_path= os.path.join(work_dir, str(hvd.size())+'-'+time.strftime('%Y%m%d-%H%M%S'))
    
    if not os.path.exists(dir_path) and hvd.rank() == 0:
        os.makedirs(dir_path)
    if hvd.rank() == 0:
        print('Experiment dir : {}'.format(dir_path))
        with open(os.path.join(dir_path,"param.txt"),"w") as f:
            f.write('    - GPUs : {}\n'.format(hvd.size()))
            for k, v in args.__dict__.items():
                f.write('    - {} : {}\n'.format(k, v))

    # trainlogfile=os.path.join(dir_path,"train.txt")
    # vallogfile=os.path.join(dir_path,"val_in_middle.txt")
    testlogfile=os.path.join(dir_path,"test.txt")
    model_checkfile=os.path.join(dir_path,"model.pth.tar")

    
    best_acc1 = 0.0
    best_acc5 = 0.0

    file_path=os.path.join(DATAPATH,args.dataset+"_"+str(args.hidden_dim)+".npz")
    # print(file_path)
    file=np.load(file_path)
    train_data=file["train_data"]
    train_label=file["train_label"]
    test_data=file["test_data"]
    test_label=file["test_label"]
    # print("data loaded")


    train_dataset = data_utils.TensorDataset(torch.from_numpy(train_data),torch.from_numpy(train_label))
    train_sampler = DistributedSampler(train_dataset, num_replicas=hvd.size(), rank=hvd.rank())
    train_loader = data_utils.DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler,pin_memory=True, num_workers=4, shuffle=False)


    test_dataset = data_utils.TensorDataset(torch.from_numpy(test_data),torch.from_numpy(test_label))
    test_sampler = DistributedSampler(test_dataset, num_replicas=hvd.size(), rank=hvd.rank())
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, sampler=test_sampler,pin_memory=True, num_workers=4, shuffle=False)


    max_L=max(np.max(train_label),np.max(test_label))
    if hvd.rank() == 0:
        print("Statistics:", max_L)
    model = Net(max_L).cuda()
#     optimizer = Adam(model.parameters(), args.lr )
    num_tensors, tensor_sizes = count_tensors(model)
    num_params = count_parameters(model)
    if hvd.rank() == 0:
        # print(model)
        print("Number of tensors: {}, Number of parameters: {} M".format(num_tensors, num_params))

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # Horovod: broadcast parameters & optimizer state.
    
    hvd.broadcast_parameters(model.state_dict(), root_rank=0)
    hvd.broadcast_optimizer_state(optimizer, root_rank=0)
    optimizer = wrap_compress_optimizer(model, optimizer, args)

    for epoch in range(0, args.epochs):
        epoch_start_time = time.time()
        train(args, model, train_loader, test_loader, optimizer, epoch,testlogfile)


if __name__ == '__main__':
    main()
    sleep(1)
    os.system("pkill -9 python3")