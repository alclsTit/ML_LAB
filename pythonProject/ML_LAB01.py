import sys

import numpy

# for using pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


def main():

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load MNIST dateset
    path = "./DataSet"
    train_set = torchvision.datasets.MNIST(
        root=path,
        train=True,
        download=True,
        transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    )

    # Load Test dataset
    test_set = torchvision.datasets.MNIST(
        root=path,
        train=False,
        download=True,
        transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    )

    batch_size = 64

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size)

    for x, y in test_loader:
        print("Shape of X [N, C, H, W]: ", x.shape)
        print("Shape of y: ", y.shape, y.dtype)
        break

    class MyModel(nn.Module):
        def __init__(self):
            super(MyModel, self).__init__()

            out_size = 10
            unit_size = 256

            self.Layer1 = nn.Linear(28*28, unit_size)
            self.Layer2 = nn.Linear(unit_size, unit_size)
            self.Layer3 = nn.Linear(unit_size, out_size)

        def forward(self, x):
            x = torch.flatten(x, 1)
            x = self.Layer1(x)
            x = F.relu(x)
            x = self.Layer2(x)
            x = F.relu(x)
            x = self.Layer3(x)

            return x

    model = MyModel().to(device)
    print(model)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

    def train(dataloader, model, loss_fn, optimizer):
        size = len(dataloader.dataset)
        model.train()
        for batch, (x, y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)

            prev = model(x)
            loss = loss_fn(prev, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch % 100 == 0:
                loss, current = loss.item(), batch * len(x)
                print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")

    def test(dataloader, model, loss_fn):
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        model.eval()
        test_loss, correct = 0, 0
        with torch.no_grad():
            for x, y in dataloader:
                x, y = x.to(device), y.to(device)
                prev = model(x)
                test_loss += loss_fn(prev, y).item()
                correct += (prev.argmax(1) == y).type(torch.float).sum().item()
        test_loss /= num_batches
        correct /= size
        print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%  -------  AvgLoss: {test_loss:>8f} \n")

    epochs = 5
    for n in range(epochs):
        print(f"Epoch {n+1}\n --------------------------------")
        train(train_loader, model, loss_fn, optimizer)
        test(test_loader, model, loss_fn)

    print("Done!!!")


if __name__ == "__main__":
    print("")
    print("############# Start #############")

    main()
    import getpass
    getpass.getpass("\"Enter\" is Quit...")