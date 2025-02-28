import torchvision
import torch
import copy
import random
import numpy as np
from lenet import LeNet5

from otdd.pytorch.distance import DatasetDistance
from otdd.pytorch.datasets import load_torchvision_data

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

# Load datasets
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--rot', default=None, type=str)
parser.add_argument('--crop', default=None, type=str)
args = parser.parse_args()


def setup_seed(seed): 
    torch.manual_seed(seed) 
    torch.cuda.manual_seed_all(seed) 
    np.random.seed(seed) 
    random.seed(seed) 
    torch.backends.cudnn.deterministic = True 

setup_seed(args.seed)
train_transforms = []
test_transforms = [torchvision.transforms.Resize(32), 
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize((0.1307,), (0.3081,))]
import torchvision.transforms.functional as TF
class MyRotationTransform:
    """Rotate by one of the given angles."""

    def __init__(self, angles):
        self.angles = angles

    def __call__(self, x):
        return TF.rotate(x, self.angles)


if args.rot:
    if args.rot == 'rand':
        train_transforms.append(torchvision.transforms.RandomRotation((-180,180)))
    else:
        train_transforms.append(MyRotationTransform(int(args.rot)))

if args.crop:
    if args.crop == 'rand':
        train_transforms.append(torchvision.transforms.RandomResizedCrop(32))
    else:
        
        train_transforms.append(torchvision.transforms.Resize(32))
        train_transforms.append(torchvision.transforms.CenterCrop(28))
        train_transforms.append(torchvision.transforms.Resize(32))
        
else:
    train_transforms.append(torchvision.transforms.Resize(32))

train_transforms.extend([torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize((0.1307,), (0.3081,))])

print(train_transforms)
print(test_transforms)
loaders_src = load_torchvision_data('MNIST', transform=[torchvision.transforms.Compose(train_transforms), torchvision.transforms.Compose(test_transforms)])[0]
loaders_tgt = load_torchvision_data('USPS',  resize = 32)[0]

model = LeNet5()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)

model = model.cuda()
best_val = 0
# pretrain
for epoch in range(20):
    model.train()
    top1 = AverageMeter()
    for data, label in loaders_src['train']:
        data = data.cuda()
        label = label.cuda()
        output = model(data)
        loss = criterion(output, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        prec1 = accuracy(output.data, label)[0]
        top1.update(prec1.item(), data.size(0))

    print('Epoch: [{0}]'
                'Accuracy {top1.val:.3f} ({top1.avg:.3f})\t'.format(
                    epoch, len(loaders_src['train']), top1=top1))
    val_top1 = AverageMeter()
    model.eval()
    for data, label in loaders_src['valid']:
        data = data.cuda()
        label = label.cuda()
        output = model(data)
        prec1 = accuracy(output.data, label)[0]
        val_top1.update(prec1.item(), data.size(0))
    if val_top1.avg > best_val:
        best_val = val_top1.avg
        best_model_state_dict = copy.deepcopy(model.state_dict())

# load back

model.load_state_dict(best_model_state_dict)
test_top1 = AverageMeter()
for data, label in loaders_src['test']:
    data = data.cuda()
    label = label.cuda()
    output = model(data)
    prec1 = accuracy(output.data, label)[0]
    test_top1.update(prec1.item(), data.size(0))

print(f"Test Error: {100-test_top1.avg}")

for name, p in model.named_parameters():
    if not 'f' in name:
        p.requires_grad = False

for epoch in range(10):
    model.train()
    top1 = AverageMeter()
    for data, label in loaders_tgt['train']:
        data = data.cuda()  
        label = label.cuda()
        output = model(data)
        loss = criterion(output, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        prec1 = accuracy(output.data, label)[0]
        top1.update(prec1.item(), data.size(0))

    print('Epoch: [{0}]'
                'Accuracy {top1.val:.3f} ({top1.avg:.3f})\t'.format(
                    epoch, len(loaders_tgt['train']), top1=top1))
    val_top1 = AverageMeter()
    model.eval()
    for data, label in loaders_tgt['valid']:
        data = data.cuda()
        label = label.cuda()
        output = model(data)
        prec1 = accuracy(output.data, label)[0]
        val_top1.update(prec1.item(), data.size(0))
    if val_top1.avg > best_val:
        best_val = val_top1.avg
        best_model_state_dict = copy.deepcopy(model.state_dict())

# load back

model.load_state_dict(best_model_state_dict)
test_top1 = AverageMeter()
for data, label in loaders_tgt['test']:
    data = data.cuda()
    label = label.cuda()
    output = model(data)
    prec1 = accuracy(output.data, label)[0]
    test_top1.update(prec1.item(), data.size(0))

print(f"Test Error: {100-test_top1.avg}")


loaders_src = load_torchvision_data('MNIST', transform=[torchvision.transforms.Compose(train_transforms), torchvision.transforms.Compose(test_transforms)], maxsize=2000)[0]
loaders_tgt = load_torchvision_data('USPS',  resize = 32, maxsize=2000)[0]

dist = DatasetDistance(loaders_src['train'], loaders_tgt['train'],
                       inner_ot_method = 'exact',
                       debiased_loss = True,
                       p = 2, entreg = 1e-1,
                       device='cuda')

d = dist.distance(maxsamples = 1000)
print(f'OTDD(src,tgt)={d}')