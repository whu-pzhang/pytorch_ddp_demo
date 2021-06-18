import torch
import torch.nn as nn
import torchvision
from torchvision import transforms as TF

BATCH_SIZE = 256
EPOCHS = 5


def main():

    # 1. define network
    device = 'cuda'
    model = torchvision.models.resnet18(num_classes=10)
    model.to(device)

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
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=BATCH_SIZE,
                                               shuffle=True,
                                               num_workers=4,
                                               pin_memory=True)

    # 3. define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=0.01,
                                momentum=0.9,
                                weight_decay=1e-4,
                                nesterov=True)

    print(f"{'=' * 30}   Training   {'=' * 30}\n")

    # 4. start training
    model.train()
    for ep in range(1, EPOCHS + 1):
        train_loss = correct = total = 0
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

            if (idx + 1) % 50 == 0 or (idx + 1) == len(train_loader):
                print(
                    f' == step: [{ep}/{EPOCHS}] [{idx+1:3}/{len(train_loader)}]'
                    f' | loss: {train_loss / (idx+1):.3f}'
                    f' | acc: {100.*correct/total:6.3f}%')

    print(f"\n{'=' * 30}   Training Finished   {'=' * 30}")


if __name__ == '__main__':
    main()
