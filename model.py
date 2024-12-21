from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from collections import defaultdict
import numpy as np

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Input Block
        self.conv1 = nn.Conv2d(1, 8, 3, padding=1)  # 28x28x1 > 28x28x8
        self.bn1 = nn.BatchNorm2d(8)
        self.drop1 = nn.Dropout2d(0.1)
        
        # CONV Block 1
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)  # 28x28x8 > 28x28x16
        self.bn2 = nn.BatchNorm2d(16)
        self.drop2 = nn.Dropout2d(0.1)
        
        # Transition Block 1
        self.pool1 = nn.MaxPool2d(2, 2)  # 28x28x16 > 14x14x16
        
        # CONV Block 2
        self.conv3 = nn.Conv2d(16, 16, 3, padding=1)  # 14x14x16 > 14x14x16
        self.bn3 = nn.BatchNorm2d(16)
        self.drop3 = nn.Dropout2d(0.1)
        
        self.conv4 = nn.Conv2d(16, 32, 3, padding=1)  # 14x14x16 > 14x14x32
        self.bn4 = nn.BatchNorm2d(32)
        self.drop4 = nn.Dropout2d(0.1)
        
        # Transition Block 2
        self.pool2 = nn.MaxPool2d(2, 2)  # 14x14x32 > 7x7x32
        
        # Output Block
        self.conv5 = nn.Conv2d(32, 16, 1)  # 7x7x32 > 7x7x16
        self.bn5 = nn.BatchNorm2d(16)
        self.conv6 = nn.Conv2d(16, 10, 1)  # 7x7x16 > 7x7x10
        self.gap = nn.AvgPool2d(7)  # 7x7x10 > 1x1x10

    def forward(self, x):
        x = self.drop1(self.bn1(F.relu(self.conv1(x))))
        x = self.drop2(self.bn2(F.relu(self.conv2(x))))
        x = self.pool1(x)
        x = self.drop3(self.bn3(F.relu(self.conv3(x))))
        x = self.drop4(self.bn4(F.relu(self.conv4(x))))
        x = self.pool2(x)
        x = self.bn5(F.relu(self.conv5(x)))
        x = self.conv6(x)
        x = self.gap(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=1)

# Training metrics collector
class MetricsCollector:
    def __init__(self):
        self.metrics = defaultdict(list)
    
    def add_metric(self, name, value):
        self.metrics[name].append(value)
    
    def get_latest(self, name):
        return self.metrics[name][-1]
    
    def get_best(self, name):
        return max(self.metrics[name])

metrics = MetricsCollector()

# !pip install torchsummary
from torchsummary import summary
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
model = Net().to(device)
summary(model, input_size=(1, 28, 28))
total_params = sum(p.numel() for p in model.parameters())
print(f'Total Parameters: {total_params}')

torch.manual_seed(1)
batch_size = 128

kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ])),
    batch_size=batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ])),
    batch_size=batch_size, shuffle=True, **kwargs)

from tqdm import tqdm
def train(model, device, train_loader, optimizer, epoch):
    model.train()
    pbar = tqdm(train_loader)
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        pbar.set_description(desc= f'loss={loss.item()} batch_id={batch_idx}')

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)

    print(f'Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)')
    return accuracy

model = Net().to(device)
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2, verbose=True)

best_accuracy = 0
for epoch in range(1, 20):
    print(f'\nEpoch: {epoch}')
    train(model, device, train_loader, optimizer, epoch)
    accuracy = test(model, device, test_loader)
    
    metrics.add_metric('accuracy', accuracy)
    metrics.add_metric('epoch', epoch)
    
    # Update learning rate based on accuracy
    scheduler.step(accuracy)
    
    # Save best model
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        torch.save(model.state_dict(), 'best_model.pth')
    
    # Early stopping if we reach target accuracy
    if accuracy >= 99.4:
        print(f'Reached target accuracy of 99.4% at epoch {epoch}')
        break

print(f'\nBest Test Accuracy: {best_accuracy:.2f}%')