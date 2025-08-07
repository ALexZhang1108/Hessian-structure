import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import hessian_spectrum
import torch.nn.functional as F


from torchvision import datasets, transforms
from torch.optim.lr_scheduler import CosineAnnealingLR

# MNIST Network
class Net(nn.Module):
    def __init__(self, input_dim = 28*28, width = 8, n_class = 10):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_dim, width, bias=False)  # MNIST input size is 28x28
        #self.fc_hidden = nn.Linear(width, width, bias=False)
        self.fc2 = nn.Linear(width, n_class, bias=False)  # 10 classes for MNIST
        #self.fc1_linear = nn.Linear(input_dim, n_class, bias=False)
        self.input_dim = input_dim
    def forward(self, x):
        x = x.view(-1, self.input_dim)  # Flatten the input
        #x = self.fc1_linear(x)
        x = self.fc1(x)
        # x = F.relu(x)
        # x = self.fc_hidden(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x #F.log_softmax(x, dim=1)

def train(model, device, train_loader, optimizer, epoch, log_interval=100, criterion = F.cross_entropy):
    loss_list = []
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        if loss_type == 'CE':
            loss = criterion(output, target)
        elif loss_type == 'MSE':
            target_one_hot = F.one_hot(target, num_classes=n_class).float()
            loss = criterion(output, target_one_hot)
        loss.backward()
        optimizer.step()
        loss_list.append(loss.item())
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
    return loss_list
def test(model, device, test_loader, criterion = F.cross_entropy):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            if loss_type == 'CE':
                test_loss += criterion(output, target).item()
            elif loss_type == 'MSE':
                target_one_hot = F.one_hot(target, num_classes=n_class).float()
                test_loss += criterion(output, target_one_hot).item()
            
            
            # take softmax
            pred = F.softmax(output, dim=1)
            pred = pred.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


# Hyperparameters
width = 8#16
lr = 1e-4
dataset = 'cifar100'#'mnist'
visual_degree_full_hessian = 300 #  hyperparamter for visualization: heatmap_max = hessian_max / visual_degree
visual_degree_block_hessian = [3, 30]#  for fc1 and fc2.  hyperparamter for visualization: heatmap_max = hessian_max / visual_degree
n_epochs = 10
cosine_epochs = n_epochs
loss_type = 'CE'
cmap ='cividis'
comment = 'nn-' + dataset+loss_type+str(n_epochs)+'epochs'
if dataset == 'mnist':
    input_dim = 28*28
elif dataset == 'cifar10':
    input_dim = 32*32*3
    n_class = 10
elif dataset == 'cifar100':
    input_dim = 32*32*3
    n_class = 100
n_batch_for_hessian = 1

# Setup seed and CUDA
torch.manual_seed(0)
np.random.seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if dataset == 'mnist':
    # Load MNIST Dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    path = 'mnist/data'
    train_dataset = datasets.MNIST(path, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(path, train=False, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128 * 469, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)
elif dataset == 'cifar10':
    # Load CIFAR-10 Dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    path = 'cifar10/data'#'/home/yszhang/hessian_spectrum/cifar/data'
    train_dataset = datasets.CIFAR10(path, train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(path, train=False, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)

elif dataset == 'cifar100':
    # Load CIFAR-100 Dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    path = 'cifar10/data'
    train_dataset = datasets.CIFAR100(path, train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR100(path, train=False, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)


# Initialize model, optimizer and scheduler
model = Net(input_dim = input_dim, width = width, n_class = n_class).to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = CosineAnnealingLR(optimizer, T_max=cosine_epochs, eta_min=lr /10)




if loss_type == 'CE':
    criterion = nn.CrossEntropyLoss()  # Cross Entropy Loss
elif loss_type == 'MSE':
    criterion = nn.MSELoss()  # Mean Squared Error Loss

# Epochs at which to compute the Hessian (typically during LR decay)
hessian_epochs = [int(n_epochs * 0.5), n_epochs - 1]

# Sample a fixed mini-batch for Hessian computation
train_data = []
for batch_idx, (data, target) in enumerate(train_loader):
    if batch_idx >= n_batch_for_hessian:
        break
    if loss_type == 'CE':
        train_data.append({'X_train': data.to(device), 'Y_train': target.to(device)})
    elif loss_type == 'MSE':
        target_one_hot = F.one_hot(target, num_classes=n_class).float()
        train_data.append({'X_train': data.to(device), 'Y_train': target_one_hot.to(device)})


def compute_and_plot_hessian(epoch):
    hessian = hessian_spectrum.Hessian(
        model,
        train_data=train_data,
        batch_size=1,
        use_minibatch=False,
        gradient_accumulation_steps=1,
        device='cpu',
        comment=f'{comment}_epoch{epoch}',
        loss=criterion,
    )

    full_hessian = np.abs(hessian.get_full_hessian())
    max_value = np.max(full_hessian)
    plt.figure()
    plt.imshow(
        full_hessian,
        cmap=cmap,
        interpolation='nearest',
        vmax=max_value / visual_degree_full_hessian,
        vmin=0,
    )
    plt.colorbar()
    plt.savefig(f'figure_mnist/{comment}_fullhessian_epoch{epoch}.png')
    plt.close()


# train and test
loss_list = []
for epoch in range(n_epochs):
    loss_list.extend(train(model, device, train_loader, optimizer, epoch, criterion=criterion))
    scheduler.step()
    if epoch in hessian_epochs:
        compute_and_plot_hessian(epoch)


# plot loss list
plt.rcParams["axes.autolimit_mode"] = "round_numbers"
plt.rcParams["axes.xmargin"] = 0
plt.rcParams["axes.ymargin"] = 0
plt.rcParams["axes.labelsize"] = 20
plt.rcParams["axes.titlesize"] = 20
plt.rcParams["legend.fontsize"] = 10#25 #20
plt.rcParams["legend.frameon"] = True
plt.rcParams["legend.fancybox"] = True
plt.rcParams["legend.loc"] = 'upper right'
plt.rcParams["xtick.labelsize"] = 10#20
plt.rcParams["ytick.labelsize"] = 10#20
plt.rcParams["lines.markersize"] = 10
plt.rcParams["font.family"] = "serif"
plt.figure()
plt.plot(loss_list)
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.savefig(f'figure_mnist/{comment}_loss_T_{n_epochs}.png')
plt.close()
