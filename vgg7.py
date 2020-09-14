import argparse
import os
import shutil
import time
import math
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torch.optim
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models

import numpy as np 
import random 

os.environ["CUDA_VISIBLE_DEVICES"] = "0" 

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))
parser = argparse.ArgumentParser(description='PyTorch MNIST Training')
parser.add_argument('--dataset', default='MNIST', type=str, help='dataset = [MNIST]')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet18)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=100, type=int,
                    metavar='N', help='mini-batch size (default: 100)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=500, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-load', default='', type=str, metavar='PATH',
                    help='path to training mask (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--lr', '--learning-rate', default=0.0005, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('-j', '--workers', default=1, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--steps', default=100, type=int, metavar='N',
                    help='number of time steps to run')
parser.add_argument('--vth', default=2, type=float, metavar='vth',
                    help='threshold')
parser.add_argument('--leak', default=1, type=float, metavar='leak',
                    help='leaky parameter')
parser.add_argument('--hz', default=5, type=int, metavar='hz',
                    help='scale update hz')


best_prec1 = 0
change = 50
change2 = 75
change3 = 100

tp1 = [];
tp5 = [];
ep = [];
lRate = [];
device_num = 1
device = torch.device("cuda:0")

tp1_tr = [];
tp5_tr = [];
losses_tr = [];
losses_eval = [];

args = parser.parse_args()


sign = 1

scale11 = 1
scale12 = 1
scalep1 = 1
scale21 = 1
scale22 = 1
scale23 = 1
scalef0 = 1

def main():
    global args, best_prec1, device_num,sign

    batch_size = args.batch_size

    seed1 = 44
    seed2 = 56
    seed3 = 78
    torch.manual_seed(seed1)
    torch.cuda.manual_seed(seed2)
    torch.cuda.manual_seed_all(seed3)
    np.random.seed(seed1)
    random.seed(seed2)
    cudnn.benchmark = False
    cudnn.deterministic = True
    
    print('\n'+'='*15+'settings'+'='*15)
    print('lr: ', args.lr)
    print('change lr point:%d %d %d'%(change,change2,change3))
    print('batchsize:',batch_size)
    print('steps:', args.steps)
    print('vth:', args.vth)
    print('leak:{}'.format(args.leak))
    print('hz:{}'.format(args.hz))
    print('seed:%d %d %d'%(seed1,seed2,seed3))
    print('='*15+'settings'+'='*15+'\n')

    model = CNNModel()

    print(model)
    
    model = torch.nn.DataParallel(model)
    model.to(device)
    

    criterion = torch.nn.MSELoss(reduction='sum')
    criterion_en = torch.nn.CrossEntropyLoss()

    learning_rate = args.lr
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=5e-4)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    dataset_path = '/data/Zzzxd/cifar10-py'

    # Dataloader
    normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.557, 0.549, 0.5534])
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    train_data = torchvision.datasets.CIFAR10(dataset_path, train=True, download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.workers,
                                               pin_memory=True)

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    val_data = torchvision.datasets.CIFAR10(dataset_path, train=False, download=True, transform=transform_test)
    val_loader = torch.utils.data.DataLoader(val_data,  # val_data for testing
                                             batch_size=int(args.batch_size/2), shuffle=False,
                                             num_workers=args.workers,
                                             pin_memory=False)

    print('read dataset done')
    if args.evaluate:
        validate(val_loader, model, criterion, criterion_en, time_steps=args.steps, leak=args.leak)
        return

    for epoch in range(args.start_epoch, args.epochs):
        if epoch % args.hz == 0 and args.hz < args.epochs:
            sign = 1
        else:
            sign = 0

        start = time.time()
        adjust_learning_rate(optimizer, epoch)

        ep.append(epoch)

        # train for one epoch
        train(train_loader, model, criterion, criterion_en, optimizer, epoch, time_steps=args.steps, leak=args.leak)

        # evaluate on validation set
        modeltest = model.module
        prec1 = validate(val_loader, modeltest, criterion, criterion_en, time_steps=args.steps, leak=args.leak)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer': optimizer.state_dict(),
        }, is_best)
        time_use = time.time() - start
        print('time used this epoch: %d h%dmin%ds' %(time_use//3600,(time_use%3600)//60,time_use%60))

        if sign == 1:
            print('\n'+'='*15+'scale'+'='*15)
            print('scale11: ', scale11)
            print('scale12: ', scale12)
            print('scalep1: ', scalep1)
            print('scale21: ', scale21)
            print('scale22: ', scale22)
            print('scale23: ', scale23)
            print('scalef0: ', scalef0)
            print('='*15+'scale'+'='*15+'\n')

    for k in range(0, args.epochs - args.start_epoch):
        print('Epoch: [{0}/{1}]\t'
              'LR:{2}\t'
              'Prec@1 {top1:.3f} \t'
              'Prec@5 {top5:.3f} \t'
              'En_Loss_Eval {losses_en_eval: .4f} \t'
              'Prec@1_tr {top1_tr:.3f} \t'
              'Prec@5_tr {top5_tr:.3f} \t'
              'En_Loss_train {losses_en: .4f}'.format(
            ep[k], args.epochs, lRate[k], top1=tp1[k], top5=tp5[k], losses_en_eval=losses_eval[k], top1_tr=tp1_tr[k],
            top5_tr=tp5_tr[k], losses_en=losses_tr[k]))
    print('best_acc={}'.format(best_prec1))


def print_view(v):
    v = v.view(v.size(0), -1)
    j = 0 
    for i in v[0]:
        print(i)
        j = j + 1
    print(j)

def grad_cal(scale, IF_in):
    out = scale * IF_in.gt(0).type(torch.cuda.FloatTensor)
    return out

def ave(output, input):
    c = input >= output
    if input[c].sum() < 1e-3:
        return 1
    return output[c].sum()/input[c].sum()

def ave_p(output, input):
    if input.sum() < 1e-3:
        return 1
    return output.sum()/input.sum()


def train(train_loader, model, criterion, criterion_en, optimizer, epoch, time_steps, leak):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    top1_tr = AverageMeter()
    top5_tr = AverageMeter()
    losses_en = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    start_end = end 
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        input, target = input.to(device), target.to(device)
        labels = target.clone()

        optimizer.zero_grad()  # Clear gradients w.r.t. parameters

        output = model(input, steps=time_steps, l=leak)
        
        targetN = output.data.clone().zero_().to(device)
        targetN.scatter_(1, target.unsqueeze(1), 1)
        targetN = Variable(targetN.type(torch.cuda.FloatTensor))

        loss = criterion(output.cpu(), targetN.cpu())
        loss_en = criterion_en(output.cpu(), labels.cpu())

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        prec1_tr, prec5_tr = accuracy(output.data, target, topk=(1, 5))
        losses_en.update(loss_en.item(), input.size(0))
        top1_tr.update(prec1_tr.item(), input.size(0))
        top5_tr.update(prec5_tr.item(), input.size(0))

        # compute gradient and do SGD step
        loss.backward(retain_graph=False)

        


        optimizer.step()


        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
    
    
    print('Epoch: [{0}] Prec@1 {top1_tr.avg:.3f} Prec@5 {top5_tr.avg:.3f} Entropy_Loss {loss_en.avg:.4f}'
          .format(epoch, top1_tr=top1_tr, top5_tr=top5_tr, loss_en=losses_en))
    time_use = end - start_end
    print('train time: %d h%dmin%ds' %(time_use//3600,(time_use%3600)//60,time_use%60))

    losses_tr.append(losses_en.avg)
    tp1_tr.append(top1_tr.avg)
    tp5_tr.append(top5_tr.avg)


def validate(val_loader, model, criterion, criterion_en, time_steps, leak):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    losses_en_eval = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
	        # measure data loading time
            data_time.update(time.time() - end)
            input_var = input.to(device)
            labels = Variable(target.to(device))
            target = target.to(device)
            output = model.tst(input=input_var, steps=time_steps, l=leak)

            targetN = output.data.clone().zero_().to(device)
            targetN.scatter_(1, target.unsqueeze(1), 1)
            targetN = Variable(targetN.type(torch.cuda.FloatTensor))
            loss = criterion(output.cpu(), targetN.cpu())
            loss_en = criterion_en(output.cpu(), labels.cpu())
	        # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))
            top5.update(prec5.item(), input.size(0))
            losses_en_eval.update(loss_en.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

    print('Test: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Entropy_Loss {losses_en_eval.avg:.4f}'
          .format(top1=top1, top5=top5, losses_en_eval=losses_en_eval))

    tp1.append(top1.avg)
    tp5.append(top5.avg)
    losses_eval.append(losses_en_eval.avg)

    return top1.avg


def save_checkpoint(state, is_best, filename='checkpointT1_cifar10_v7.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_bestT1_cifar10_v9.pth.tar')


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


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (1 ** (epoch // change))

    for param_group in optimizer.param_groups:
        if epoch >= change3:
            param_group['lr'] = 0.2 * 0.2 * 0.2 * lr

        elif epoch >= change2:
            param_group['lr'] = 0.2 * 0.2 * lr

        elif epoch >= change:
            param_group['lr'] = 0.2 * lr

        else:
            param_group['lr'] = lr

    lRate.append(param_group['lr'])


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


class SpikingNN(torch.autograd.Function):
    def forward(self, input):
        self.save_for_backward(input)
        return input.gt(0).type(torch.cuda.FloatTensor)

    def backward(self, grad_output):
        input, = self.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input <= 0.0] = 0
        return grad_input


def LIF_sNeuron(membrane_potential, threshold, l, i):
    # check exceed membrane potential and reset
    ex_membrane = nn.functional.threshold(membrane_potential, threshold, 0)
    membrane_potential = membrane_potential - ex_membrane # hard reset
    # generate spike
    out = SpikingNN()(ex_membrane)
    # decay
    # note: the detach has no effects now
    membrane_potential = l * membrane_potential.detach() + membrane_potential - membrane_potential.detach()

    return membrane_potential, out


def Pooling_sNeuron(membrane_potential, threshold, i):
    # check exceed membrane potential and reset
    ex_membrane = nn.functional.threshold(membrane_potential, threshold, 0)
    membrane_potential = membrane_potential - ex_membrane
    # generate spike
    out = SpikingNN()(ex_membrane)

    return membrane_potential, out


class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.cnn11 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.cnn12 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.avgpool1 = nn.AvgPool2d(kernel_size=2)

        self.cnn21 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)
        self.cnn22 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)
        self.cnn23 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)

        self.avgpool2 = nn.MaxPool2d(kernel_size=2)

        self.fc0 = nn.Linear(128 * 8 * 8, 1024, bias=False)
        self.fc1 = nn.Linear(1024, 10, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                variance1 = math.sqrt(2.0 / n)
                m.weight.data.normal_(0, variance1)
                # define threshold
                m.threshold = args.vth

            elif isinstance(m, nn.Linear):
                size = m.weight.size()
                fan_in = size[1]  # number of columns
                variance2 = math.sqrt(2.0 / fan_in)
                m.weight.data.normal_(0.0, variance2)
                # define threshold
                m.threshold = args.vth 

    def forward(self, input, steps=100, l=1):
        global scale11,scale12,scale21,scale22,scale23,scalef0,scalep1,sign

        mem_11 = torch.zeros(input.size(0), 64, 32, 32, device = input.device)
        mem_12 = torch.zeros(input.size(0), 64, 32, 32, device = input.device)

        mem_1s = torch.zeros(input.size(0), 64, 16, 16, device = input.device)

        mem_21 = torch.zeros(input.size(0), 128, 16, 16, device = input.device)
        mem_22 = torch.zeros(input.size(0), 128, 16, 16, device = input.device)
        mem_23 = torch.zeros(input.size(0), 128, 16, 16, device = input.device)

        membrane_f0 = torch.zeros(input.size(0), 1024, device = input.device)
        
        Total_input = torch.zeros(input.size(0), 3, 32, 32, device = input.device)

        Total_11_output = torch.zeros(input.size(0), 64, 32, 32, device = input.device)
        IF_11_in = torch.zeros(input.size(0), 64, 32, 32, device = input.device)

        Total_12_output = torch.zeros(input.size(0), 64, 32, 32, device = input.device)
        IF_12_in = torch.zeros(input.size(0), 64, 32, 32, device = input.device)

        Total_p1_output = torch.zeros(input.size(0), 64, 16, 16,device = input.device)
        IF_p1_in = torch.zeros(input.size(0), 64, 16, 16, device = input.device)

        Total_21_output = torch.zeros(input.size(0), 128, 16, 16, device = input.device)
        IF_21_in = torch.zeros(input.size(0), 128, 16, 16, device = input.device)

        Total_22_output = torch.zeros(input.size(0), 128, 16, 16, device = input.device)
        IF_22_in = torch.zeros(input.size(0), 128, 16, 16, device = input.device)
        
        Total_23_output = torch.zeros(input.size(0), 128, 16, 16, device = input.device)
        IF_23_in = torch.zeros(input.size(0), 128, 16, 16, device = input.device)

        Total_f0_output = torch.zeros(input.size(0), 1024, device = input.device)
        IF_f0_in = torch.zeros(input.size(0), 1024, device = input.device)

        with torch.no_grad():
            for i in range(steps):
                # Poisson input spike generation
                rand_num = torch.rand(input.size(0), input.size(1), input.size(2), input.size(3), device = input.device)
                Poisson_d_input = (torch.abs(input) > rand_num)
                Poisson_d_input = torch.mul(Poisson_d_input.float(), torch.sign(input))
                Total_input = Total_input + Poisson_d_input

                # convolutional Layer
                in_layer = self.cnn11(Poisson_d_input)
                mem_11 = mem_11 + in_layer
                mem_11, out = LIF_sNeuron(mem_11, self.cnn11.threshold, l, i)
                IF_11_in = IF_11_in + in_layer
                Total_11_output = Total_11_output + out 


                in_layer = self.cnn12(out)
                mem_12 = mem_12 + in_layer
                mem_12, out = LIF_sNeuron(mem_12, self.cnn12.threshold, l, i)
                IF_12_in = IF_12_in + in_layer
                Total_12_output = Total_12_output + out


                # pooling Layer
                in_layer = self.avgpool1(out)
                mem_1s = mem_1s + in_layer
                mem_1s, out = Pooling_sNeuron(mem_1s, 0.75, i)
                IF_p1_in = IF_p1_in + in_layer
                Total_p1_output = Total_p1_output + out 

                # convolutional Layer
                in_layer = self.cnn21(out)
                mem_21 = mem_21 + in_layer
                mem_21, out = LIF_sNeuron(mem_21, self.cnn21.threshold, l, i)
                IF_21_in = IF_21_in + in_layer
                Total_21_output = Total_21_output + out 

                in_layer = self.cnn22(out)
                mem_22 = mem_22 + in_layer
                mem_22, out = LIF_sNeuron(mem_22, self.cnn22.threshold, l, i)
                IF_22_in = IF_22_in + in_layer
                Total_22_output = Total_22_output + out 
                
                in_layer = self.cnn23(out)
                mem_23 = mem_23 + in_layer
                mem_23, out = LIF_sNeuron(mem_23, self.cnn23.threshold, l, i)
                IF_23_in = IF_23_in + in_layer
                Total_23_output = Total_23_output + out 

                out = self.avgpool2(out)
                out = out.view(out.size(0), -1)

                # fully-connected Layer
                in_layer = self.fc0(out)
                membrane_f0 = membrane_f0 + in_layer
                membrane_f0, out = LIF_sNeuron(membrane_f0, self.fc0.threshold, l, i)
                IF_f0_in = IF_f0_in + in_layer
                Total_f0_output = Total_f0_output + out 


            if sign == 1:
                scalef0 = 0.6 * ave(Total_f0_output, IF_f0_in) + 0.4 * scalef0
                scale11 = 0.6 * ave(Total_11_output, IF_11_in) + 0.4 * scale11
                scale12 = 0.6 * ave(Total_12_output, IF_12_in) + 0.4 * scale12
                scalep1 = 0.6 * ave_p(Total_p1_output, IF_p1_in) + 0.4 * scalep1
                scale21 = 0.6 * ave(Total_21_output, IF_21_in) + 0.4 * scale21
                scale22 = 0.6 * ave(Total_22_output, IF_22_in) + 0.4 * scale22
                scale23 = 0.6 * ave(Total_23_output, IF_23_in) + 0.4 * scale23


            scale_f0 = grad_cal(scalef0, IF_f0_in)
            scale_11 = grad_cal(scale11, IF_11_in)
            scale_12 = grad_cal(scale12, IF_12_in)
            scale_p1 = grad_cal(scalep1, IF_p1_in)
            scale_21 = grad_cal(scale21, IF_21_in)
            scale_22 = grad_cal(scale22, IF_22_in)
            scale_23 = grad_cal(scale23, IF_23_in)
        
        with torch.enable_grad():
            cnn11_in = self.cnn11(Total_input.detach())
            tem = Total_11_output.detach()
            out = torch.mul(cnn11_in, scale_11)
            Total_11_output = out - out.detach() + tem            
            
            cnn12_in = self.cnn12(Total_11_output)
            tem = Total_12_output.detach()
            out = torch.mul(cnn12_in, scale_12)
            Total_12_output = out - out.detach() + tem            
            
            pool1_in = self.avgpool1(Total_12_output)
            tem = Total_p1_output.detach()
            out = torch.mul(pool1_in, scale_p1)
            Total_p1_output = out - out.detach() + tem
            
            cnn21_in = self.cnn21(Total_p1_output)
            tem = Total_21_output.detach()
            out = torch.mul(cnn21_in, scale_21)
            Total_21_output = out - out.detach() + tem            
            
            cnn22_in = self.cnn22(Total_21_output)
            tem = Total_22_output.detach()
            out = torch.mul(cnn22_in, scale_22)
            Total_22_output = out - out.detach() + tem            

            cnn23_in = self.cnn23(Total_22_output)
            tem = Total_23_output.detach()
            out = torch.mul(cnn23_in, scale_23)
            Total_23_output = out - out.detach() + tem            
            
            Total_p2_output = self.avgpool2(Total_23_output)
                        
            fc0_in = self.fc0(Total_p2_output.view(Total_p2_output.size(0),-1))
            tem = Total_f0_output.detach()
            out = torch.mul(fc0_in, scale_f0)
            Total_f0_output = out - out.detach() + tem
            
            fc1_in = self.fc1(Total_f0_output)

        return fc1_in/self.fc1.threshold/steps


    def tst(self, input, steps=100, l=1):
        mem_11 = torch.zeros(input.size(0), 64, 32, 32,device = input.device)
        mem_12 = torch.zeros(input.size(0), 64, 32, 32,device = input.device)
        mem_1s = torch.zeros(input.size(0), 64, 16, 16,device = input.device)

        mem_21 = torch.zeros(input.size(0), 128, 16, 16,device = input.device)
        mem_22 = torch.zeros(input.size(0), 128, 16, 16,device = input.device)
        mem_23 = torch.zeros(input.size(0), 128, 16, 16,device = input.device)

        membrane_f0 = torch.zeros(input.size(0), 1024,device = input.device)
        membrane_f1 = torch.zeros(input.size(0), 10,device = input.device)

        for i in range(steps):
            # Poisson input spike generation
            rand_num = torch.rand(input.size(0), input.size(1), input.size(2), input.size(3), device = input.device)
            Poisson_d_input = (torch.abs(input) > rand_num).type(torch.cuda.FloatTensor)
            Poisson_d_input = torch.mul(Poisson_d_input, torch.sign(input))

            # convolutional Layer
            mem_11 = mem_11 + self.cnn11(Poisson_d_input)
            mem_11, out = LIF_sNeuron(mem_11, self.cnn11.threshold, l, i)

            mem_12 = mem_12 + self.cnn12(out)
            mem_12, out = LIF_sNeuron(mem_12, self.cnn12.threshold, l, i)

            # pooling Layer
            mem_1s = mem_1s + self.avgpool1(out)
            mem_1s, out = Pooling_sNeuron(mem_1s, 0.75, i)

            # convolutional Layer
            mem_21 = mem_21 + self.cnn21(out)
            mem_21, out = LIF_sNeuron(mem_21, self.cnn21.threshold, l, i)

            mem_22 = mem_22 + self.cnn22(out)
            mem_22, out = LIF_sNeuron(mem_22, self.cnn22.threshold, l, i)
            
            mem_23 = mem_23 + self.cnn23(out)
            mem_23, out = LIF_sNeuron(mem_23, self.cnn23.threshold, l, i)

            # pooling Layer
            out = self.avgpool2(out)
            out = out.view(out.size(0), -1)

            # fully-connected Layer
            membrane_f0 = membrane_f0 + self.fc0(out)
            membrane_f0, out = LIF_sNeuron(membrane_f0, self.fc0.threshold, l, i)

            membrane_f1 = membrane_f1 + self.fc1(out)
        return membrane_f1 / self.fc1.threshold / steps

if __name__ == '__main__':
    main()
