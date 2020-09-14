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
import torchvision.transforms as transforms
import torchvision.models as models
import torchvision.datasets as dsets
import numpy as np 
import random

import torch._utils
try:
    torch._utils._rebuild_tensor_v2
except AttributeError:
    def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks):
        tensor = torch._utils._rebuild_tensor(storage, storage_offset, size, stride)
        tensor.requires_grad = requires_grad
        tensor._backward_hooks = backward_hooks
        return tensor
    torch._utils._rebuild_tensor_v2 = _rebuild_tensor_v2

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
parser.add_argument('-b', '--batch-size', default=64, type=int,
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
parser.add_argument('--lr', '--learning-rate', default=0.00085, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('-j', '--workers', default=1, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=125, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--steps', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--vth', default=1, type=float, metavar='Vth',
                    help='voltage threshold')  
parser.add_argument('--leak', default=1, type=float, metavar='Leak',
                    help='leaky parameter')         
parser.add_argument('--hz', default=5, type=int, metavar='hz',
                    help='scale update hz') 
parser.add_argument('--seed', default=0, type=int, metavar='seed',
                    help='whether change the seed') 

best_prec1 = 0
change = 25
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

sign = 1

scale1 = 1
scale2 = 1
scale3 = 1
scale4 = 1
scale5 = 1
args = parser.parse_args()

def main():
    global args, best_prec1, device_num, sign 
    if args.seed:
        seed1 = random.randint(1,100)
        seed2 = random.randint(1,100)
        seed3 = random.randint(1,100)
    else:
        seed1 = 30
        seed2 = 22
        seed3 = 66
    batch_size = args.batch_size
    print('\n'+'='*15+'settings'+'='*15)
    print('lr: ', args.lr)
    print('change lr point:%d'%change)
    print('batchsize:',batch_size)
    print('lenet adapt version')
    print('random-seed = %d %d %d'%(seed1,seed2,seed3))
    print('steps:{}'.format(args.steps))
    print('vth:{}'.format(args.vth))
    print('leak:{}'.format(args.leak))
    print('scale hz:{}'.format(args.hz))
    # print('rand seed: %d'%seed)
    print('='*15+'settings'+'='*15+'\n')
    
    torch.manual_seed(seed1)
    torch.cuda.manual_seed(seed2)
    torch.cuda.manual_seed_all(seed3)
    np.random.seed(seed1)
    random.seed(seed2)
    

    model = CNNModel()
    print(model)
    model = torch.nn.DataParallel(model)
    model.to(device)

    criterion = torch.nn.MSELoss(reduction='sum')
    criterion_en = torch.nn.CrossEntropyLoss()

    learning_rate = args.lr
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=5e-4)

    cudnn.benchmark = False
    cudnn.deterministic = True


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

    

    '''STEP 1: LOADING DATASET'''
    dataset_path = '/data/diospada/mnist-python/data'
    train_data = dsets.MNIST(root=dataset_path, train=True, transform=transforms.ToTensor(), download=True)
    val_data = dsets.MNIST(root=dataset_path, train=False, transform=transforms.ToTensor())

    train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(dataset=val_data, batch_size=int(args.batch_size), shuffle=False)
    print('read dataset succeed')
    if args.evaluate:
        validate(val_loader, model, criterion, criterion_en, time_steps=args.steps, leak=args.leak)
        return

    prec1_tr = 0
    for epoch in range(args.start_epoch, args.epochs):
        if epoch % args.hz == 0 and args.hz < args.epochs:
            sign = 1
        else:
            sign = 0
        adjust_learning_rate(optimizer, epoch)
        ep.append(epoch)
        start_end = time.time()
        # train for one epoch
        prec1_tr = train(train_loader, model, criterion, criterion_en, optimizer, epoch, time_steps=args.steps, leak=args.leak)

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

        time_used = time.time() - start_end
        print('time used this epoch: %dmin %ds'%(time_used//60,time_used%60))
    for k in range(0, args.epochs - args.start_epoch):
        print('Epoch: [{0}/{1}]\t'
              'LR:{2}\t'
              'Prec@1 {top1:.3f} \t'
              'Prec@5 {top5:.3f} '.format(
            ep[k], args.epochs, lRate[k], top1=tp1[k], top5=tp5[k]))
    print('best:',best_prec1)


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
    for i, (inputdata, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        inputdata, target = inputdata.to(device), target.to(device)
        labels = target.clone()

        optimizer.zero_grad()  # Clear gradients w.r.t. parameters

        output = model(inputdata, steps=time_steps, l=leak)

        targetN = output.data.clone().zero_().to(device)
        targetN.scatter_(1, target.unsqueeze(1), 1)
        targetN = Variable(targetN.type(torch.cuda.FloatTensor))

        loss = criterion(output.cpu(), targetN.cpu())
        loss_en = criterion_en(output.cpu(), labels.cpu())

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.item(), inputdata.size(0))
        top1.update(prec1.item(), inputdata.size(0))
        top5.update(prec5.item(), inputdata.size(0))

        prec1_tr, prec5_tr = accuracy(output.data, target, topk=(1, 5))
        losses_en.update(loss_en.item(), inputdata.size(0))
        top1_tr.update(prec1_tr.item(), inputdata.size(0))
        top5_tr.update(prec5_tr.item(), inputdata.size(0))

        loss.backward(retain_graph=False)
        
        
        optimizer.step()


        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
    time_used = end - start_end
    print('train time: %dmin %ds'%(time_used//60,time_used%60))

    print('Epoch: [{0}] Prec@1 {top1_tr.avg:.3f} Prec@5 {top5_tr.avg:.3f} Entropy_Loss {loss_en.avg:.4f}'
          .format(epoch, top1_tr=top1_tr, top5_tr=top5_tr, loss_en=losses_en))

    losses_tr.append(losses_en.avg)
    tp1_tr.append(top1_tr.avg)
    tp5_tr.append(top5_tr.avg)

    return top1_tr.avg


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
        for i, (inputdata, target) in enumerate(val_loader):
            # measure data loading time
            data_time.update(time.time() - end)
            input_var = inputdata.to(device)
            target = target.to(device)
        
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
            losses.update(loss.item(), inputdata.size(0))
            top1.update(prec1.item(), inputdata.size(0))
            top5.update(prec5.item(), inputdata.size(0))
            losses_en_eval.update(loss_en.item(), inputdata.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

    print('Test: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Entropy_Loss {losses_en_eval.avg:.4f}'
          .format(top1=top1, top5=top5, losses_en_eval=losses_en_eval))

    tp1.append(top1.avg)
    tp5.append(top5.avg)
    losses_eval.append(losses_en_eval.avg)

    return top1.avg


def save_checkpoint(state, is_best, filename='checkpointT1_mnist1.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_bestT1_mnist1.pth.tar')


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
    lr = args.lr

    for param_group in optimizer.param_groups:
        if epoch >= change:
            param_group['lr'] = 0.2 * lr

        elif epoch < change:
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
    membrane_potential = membrane_potential - ex_membrane
    # generate spike
    out = SpikingNN()(ex_membrane)
    membrane_potential = l * membrane_potential.detach() + membrane_potential - membrane_potential.detach()

    return membrane_potential, out


def Pooling_sNeuron(membrane_potential, threshold, i):
    # check exceed membrane potential and reset
    ex_membrane = nn.functional.threshold(membrane_potential, threshold, 0)
    membrane_potential = membrane_potential - ex_membrane # hard reset
    # generate spike
    out = SpikingNN()(ex_membrane)

    return membrane_potential, out


class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=20, kernel_size=5, stride=1, padding=2, bias=False)
        self.avgpool1 = nn.AvgPool2d(kernel_size=2)

        self.cnn2 = nn.Conv2d(in_channels=20, out_channels=50, kernel_size=5, stride=1, padding=2, bias=False)
        self.avgpool2 = nn.AvgPool2d(kernel_size=2)

        self.fc0 = nn.Linear(50*7*7, 200, bias=False)
        self.fc1 = nn.Linear(200, 10, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                variance1 = math.sqrt(2. / n)
                m.weight.data.normal_(0, variance1)
                m.threshold = args.vth

            elif isinstance(m, nn.Linear):
                size = m.weight.size()
                fan_in = size[1]
                variance2 = math.sqrt(2.0 / fan_in)
                m.weight.data.normal_(0.0, variance2)
                m.threshold = args.vth

    def forward(self, inputdata, steps=100, l=1):

        global scale1, scale2, scale3, scale4, scale5, sign 

        mem_1 = torch.zeros(inputdata.size(0), 20, 28, 28, device = inputdata.device)
        mem_1s = torch.zeros(inputdata.size(0), 20, 14, 14, device =inputdata.device)

        mem_2 = torch.zeros(inputdata.size(0), 50, 14, 14, device = inputdata.device)
        mem_2s = torch.zeros(inputdata.size(0), 50, 7, 7, device = inputdata.device)

        membrane_f0 = torch.zeros(inputdata.size(0), 200, device = inputdata.device)

        Total_input = torch.zeros(inputdata.size(0), 1, 28, 28, device = inputdata.device)

        Total_1_output = torch.zeros(inputdata.size(0), 20, 28, 28, device = inputdata.device)
        IF_in_c1 = torch.zeros(inputdata.size(0), 20, 28, 28, device = inputdata.device)

        Total_2_output = torch.zeros(inputdata.size(0), 50, 14, 14, device = inputdata.device)
        IF_in_c2 = torch.zeros(inputdata.size(0), 50, 14, 14, device = inputdata.device)

        Total_p1_output = torch.zeros(inputdata.size(0), 20, 14, 14, device = inputdata.device)
        IF_in_p1 = torch.zeros(inputdata.size(0), 20, 14, 14, device = inputdata.device)

        Total_p2_output = torch.zeros(inputdata.size(0), 50, 7, 7, device = inputdata.device)
        IF_in_p2 = torch.zeros(inputdata.size(0), 50, 7, 7, device = inputdata.device)

        Total_f0_output = torch.zeros(inputdata.size(0), 200, device = inputdata.device)
        IF_in_f0 = torch.zeros(inputdata.size(0), 200, device = inputdata.device)

        with torch.no_grad():
            for i in range(steps):
                # Poisson input spike generation
                rand_num = torch.rand(inputdata.size(0), inputdata.size(1), inputdata.size(2), inputdata.size(3), device = inputdata.device)
                Poisson_d_input = (torch.abs(inputdata)/2) > rand_num
                Poisson_d_input = torch.mul(Poisson_d_input.float(), torch.sign(inputdata))
                Total_input = Total_input + Poisson_d_input

                # convolutional Layer
                in_layer = self.cnn1(Poisson_d_input)
                mem_1 = mem_1 + in_layer
                mem_1, out = LIF_sNeuron(mem_1, self.cnn1.threshold, l, i)
                IF_in_c1 = IF_in_c1 + in_layer
                Total_1_output = Total_1_output + out 

                # pooling Layer
                in_layer = self.avgpool1(out)
                mem_1s = mem_1s + in_layer
                mem_1s, out = Pooling_sNeuron(mem_1s, 0.75, i)
                IF_in_p1 = IF_in_p1 + in_layer
                Total_p1_output = Total_p1_output + out 

                # convolutional Layer
                in_layer = self.cnn2(out)
                mem_2 = mem_2 + in_layer
                mem_2, out = LIF_sNeuron(mem_2, self.cnn2.threshold, l, i)
                IF_in_c2 = IF_in_c2 + in_layer
                Total_2_output = Total_2_output + out 

                # pooling Layer
                in_layer = self.avgpool2(out)
                mem_2s = mem_2s + in_layer
                mem_2s, out = Pooling_sNeuron(mem_2s, 0.75, i)
                IF_in_p2 = IF_in_p2 + in_layer
                Total_p2_output = Total_p2_output + out 

                out = out.view(out.size(0), -1)

                # fully-connected Layer
                in_layer = self.fc0(out)
                membrane_f0 = membrane_f0 + in_layer
                membrane_f0, out = LIF_sNeuron(membrane_f0, self.fc0.threshold, l, i)
                IF_in_f0 = IF_in_f0 + in_layer
                Total_f0_output = Total_f0_output + out 

            if sign == 1:
                scale1 = 0.6 * ave(Total_1_output, IF_in_c1) + 0.4 * scale1
                scale2 = 0.6 * ave_p(Total_p1_output, IF_in_p1) + 0.4 * scale2
                scale3 = 0.6 * ave(Total_2_output, IF_in_c2) + 0.4 * scale3
                scale4 = 0.6 * ave_p(Total_p2_output, IF_in_p2) + 0.4 * scale4
                scale5 = 0.6 * ave(Total_f0_output, IF_in_f0) + 0.4 * scale5
                

            scale_1 = grad_cal(scale1, IF_in_c1)
            scale_2 = grad_cal(scale2, IF_in_p1)
            scale_3 = grad_cal(scale3, IF_in_c2)
            scale_4 = grad_cal(scale4, IF_in_p2)
            scale_5 = grad_cal(scale5, IF_in_f0)

        with torch.enable_grad():
            cnn1_in = self.cnn1(Total_input.detach())
            tem =  Total_1_output.detach()
            out = torch.mul(cnn1_in,scale_1)
            Total_1_output = out - out.detach() + tem


            pool1_in = self.avgpool1(Total_1_output)
            tem = Total_p1_output.detach()
            out = torch.mul(pool1_in,scale_2)
            Total_p1_output = out - out.detach() + tem 

            cnn2_in = self.cnn2(Total_p1_output)
            tem = Total_2_output.detach()
            out = torch.mul(cnn2_in, scale_3)
            Total_2_output = out - out.detach() + tem

            pool2_in = self.avgpool2(Total_2_output)
            tem = Total_p2_output.detach()
            out = torch.mul(pool2_in, scale_4)
            Total_p2_output = out - out.detach() + tem

            fc0_in = self.fc0(Total_p2_output.view(Total_p2_output.size(0),-1)) 
            tem = Total_f0_output.detach()
            out = torch.mul(fc0_in, scale_5)
            Total_f0_output = out - out.detach() + tem
            
            fc1_in = self.fc1(Total_f0_output)


        return fc1_in/self.fc1.threshold/steps


    def tst(self, input, steps=100, l=1):
        mem_1 = torch.zeros(input.size(0), 20, 28, 28, device = input.device)
        mem_1s = torch.zeros(input.size(0), 20, 14, 14, device = input.device)
        mem_2 = torch.zeros(input.size(0), 50, 14, 14, device = input.device)
        mem_2s = torch.zeros(input.size(0), 50, 7, 7, device = input.device)

        membrane_f0 = torch.zeros(input.size(0), 200, device = input.device)
        membrane_f1 = torch.zeros(input.size(0), 10, device = input.device)

        for i in range(steps):
            # Poisson input spike generation
            rand_num = torch.rand(input.size(0), input.size(1), input.size(2), input.size(3), device =input.device)
            Poisson_d_input = ((torch.abs(input)/2) > rand_num).type(torch.cuda.FloatTensor)
            Poisson_d_input = torch.mul(Poisson_d_input, torch.sign(input))

            # convolutional Layer
            mem_1 = mem_1 + self.cnn1(Poisson_d_input)
            mem_1, out = LIF_sNeuron(mem_1, self.cnn1.threshold, l, i)

            # pooling Layer
            mem_1s = mem_1s + self.avgpool1(out)
            mem_1s, out = Pooling_sNeuron(mem_1s, 0.75, i)

            # convolutional Layer
            mem_2 = mem_2 + self.cnn2(out)
            mem_2, out = LIF_sNeuron(mem_2, self.cnn1.threshold, l, i)

            # pooling Layer
            mem_2s = mem_2s + self.avgpool2(out)
            mem_2s, out = Pooling_sNeuron(mem_2s, 0.75, i)

            out = out.view(out.size(0), -1)

            # fully-connected Layer
            membrane_f0 = membrane_f0 + self.fc0(out)
            membrane_f0, out = LIF_sNeuron(membrane_f0, self.fc0.threshold, l, i)

            membrane_f1 = membrane_f1 + self.fc1(out)

        return membrane_f1 / self.fc1.threshold / steps

if __name__ == '__main__':
    main()
