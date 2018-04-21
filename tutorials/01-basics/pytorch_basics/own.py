import torch, torchvision
import torch.nn as nn
import numpy as np
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from torch.autograd import Variable

##================== autograd ex1 =======================
x = Variable(torch.Tensor([1]), requires_grad=True)
w = Variable(torch.Tensor([2]), requires_grad=True)
b = Variable(torch.Tensor([3]), requires_grad=True)
#print("x : {}\nw : {}\nb : {}".format(x, w, b))
y = w * x + b

y.backward()

#print("x.grad : {}\nw.grad : {}\nb.grad : {}".format(x.grad, w.grad, b.grad))


##================== autograd ex2 =======================
x = Variable(torch.randn(5, 3))
y = Variable(torch.randn(5, 2))

# Linear layer
linear = nn.Linear(3, 2)
#print("w: {}\nb: {}".format(linear.weight, linear.bias))

# Loss and Optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(linear.parameters(), lr=0.01)

# Forward propagation
pred = linear(x)

# Compute Loss
loss = criterion(pred, y)
#rint("loss: {}".format(loss.data[0]))

# Back propagation
loss.backward()
#print("dL/dw : {}\ndL/db : {}".format(linear.weight.grad, linear.bias.grad))

optimizer.step()

pred = linear(x)
loss = criterion(pred, y)
#print("Loss after 1 step optimazation: {}".format(loss.data[0]))

##======================= Loading data from numpy ================
a = np.array([[1,2], [3, 4]])
b = torch.from_numpy(a) # numpy -> torch
c = b.numpy()# torch -> numpy
#print("type(b) : {}\ntype(c) : {}".format(type(b), type(c)))

##========================= Implementing the input pipline ====================
# Download
train_dataset = dsets.CIFAR10(
    root='../data/',
    train=True,
    transform=transforms.ToTensor(),
    download=True
)

image, label = train_dataset[0]
#print("image.size() : {}\nlabel : {}".format(image.size(), label))

train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    batch_size=100,
    shuffle=True,
    num_workers=10
)
data_iter = iter(train_loader)

# Mini-batch images and labels
images, labels = data_iter.next()