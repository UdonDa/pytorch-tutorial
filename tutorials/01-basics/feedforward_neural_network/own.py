import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable

# Hyper Parameter
input_size = 784
hidden_size = 500
num_classes = 10
num_epochs = 20
batch_size = 100
learning_rate = 0.001

# MNIST
train_dataset = dsets.MNIST(
    root = './data',
    train = True,
    transform = transforms.ToTensor(),
    download = True
)
test_dataset  = dsets.MNIST(
    root = './data',
    train = False,
    transform = transforms.ToTensor()
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
class Net(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

model = Net(input_size, hidden_size, num_classes)
model.cuda()

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
#optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# Train
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = Variable(images.view(-1, 28 * 28).cuda())
        labels = Variable(labels.cuda())

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        if ((i+1) % 100 == 0):
            print("Epoch: {}/{}, Step: {}/{}, Loss: {}".format(epoch+1, num_epochs, i+1, len(train_dataset) // batch_size, loss.data[0]))

# Test
correct, total = 0, 0
for images, labels in test_loader:
    images = Variable(images.view(-1, 28 * 28).cuda())
    outputs = model(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted.cpu() == labels).sum()

print("Accuracy: {}".format(100 * correct / total))
