import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable

# Hyper Parameter
num_epochs = 20
batch_size = 300
lerning_rate = 1e-3

# MNIST
train_dataset = dsets.MNIST(
    root = './data/',
    train = True,
    transform = transforms.ToTensor(),
    download = True
)
test_dataset  = dsets.MNIST(
    root = './data/',
    train = False,
    transform = transforms.ToTensor(),
    download = False
)

train_loader = torch.utils.data.DataLoader(
    dataset = train_dataset,
    batch_size = batch_size,
    shuffle = True
)
test_loader  = torch.utils.data.DataLoader(
    dataset = test_dataset,
    batch_size = batch_size,
    shuffle = False
)

# Model
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Linear(7 * 7 * 32, 10)
    
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

model = Model()
model.cuda()

# Loss and Optimazer
criterion = nn.CrossEntropyLoss()
optimazer = torch.optim.Adam(model.parameters(), lr=lerning_rate)

# Train
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = Variable(images).cuda()
        labels = Variable(labels).cuda()

        optimazer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimazer.step()

        if ((i+1) % 100 == 0):
            print("Epoch: {}/{}, Step: {}/{}, Loss: {}".format(epoch+1, num_epochs, i+1, len(train_dataset) // batch_size, loss.data[0]))

# Test
model.eval()
correct, total = 0, 0
for images, labels in test_loader:
    images = Variable(images).cuda()
    outputs = model(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted.cpu() == labels).sum()

print("Accuracy: {}".format(100 * correct / total))