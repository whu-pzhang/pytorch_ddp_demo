"""
(MNMC) Multiple Nodes Multi-GPU Cards Training
    with DistributedDataParallel and torch.distributed.launch
Try to compare with [snsc.py, snmc_dp.py & mnmc_ddp_mp.py] and find out the differences.


CUDA_VISIBLE_DEVICES=0,1,2,3 \
python -m torch.distributed.launch \
    --nproc_per_node=4 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=localhost \
    --master_port=22222 \
    mnmc.py
"""
import os

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
import torchvision
from torchvision import transforms as TF

BATCH_SIZE = 256
EPOCHS = 5


def main():

    # 0. set up distributed device
    rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(rank % num_gpus)
    dist.init_process_group(backend='nccl')
    device = torch.device('cuda', local_rank)

    print(f"[init] == local rank: {local_rank}, global rank: {rank} ==")

    # 1. define network
    device = 'cuda'
    model = torchvision.models.resnet18(num_classes=10)
    model.to(device)
    #
    model = nn.parallel.DistributedDataParallel(model,
                                                device_ids=[local_rank],
                                                output_device=local_rank)

    # 2. define dataloader
    train_dataset = torchvision.datasets.CIFAR10(
        root='./data',
        train=True,
        download=True,
        transform=TF.Compose([
            TF.RandomCrop(32, padding=4),
            TF.RandomHorizontalFlip(),
            TF.ToTensor(),
            TF.Normalize(mean=(0.4914, 0.4822, 0.4465),
                         std=(0.2023, 0.1994, 0.2010))
        ]))
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, shuffle=True)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=BATCH_SIZE,
                                               num_workers=4,
                                               pin_memory=True,
                                               sampler=train_sampler)

    # 3. define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=0.01,
                                momentum=0.9,
                                weight_decay=1e-4,
                                nesterov=True)

    if rank == 0:
        print(f"{'=' * 30}   Training   {'=' * 30}\n")

    # 4. start training
    model.train()
    for ep in range(1, EPOCHS + 1):
        train_loss = correct = total = 0
        # set sampler
        train_loader.sampler.set_epoch(ep)

        for idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)

            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            total += targets.size(0)
            correct += torch.eq(outputs.argmax(dim=1), targets).sum().item()

            if rank == 0 and (idx + 1) % 25 == 0 or (idx +
                                                     1) == len(train_loader):
                print(
                    f' == step: [{ep}/{EPOCHS}] [{idx+1:3}/{len(train_loader)}]'
                    f' | loss: {train_loss / (idx+1):.3f}'
                    f' | acc: {100.*correct/total:6.3f}%')

    if rank == 0:
        print(f"\n{'=' * 30}   Training Finished   {'=' * 30}")


if __name__ == '__main__':
    main()
