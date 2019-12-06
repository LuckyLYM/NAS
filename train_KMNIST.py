import os
import sys
import time
import glob
import numpy as np
import torch
import utils
import logging
import argparse
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn

from torch.autograd import Variable
from model import NetworkMNIST
import genotypes


# --epochs 20
# --split  5000

parser = argparse.ArgumentParser("KMNIST")
parser.add_argument('--data', type=str, default='data', help='location of the data corpus')    # should modify
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=0.001, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=20, help='num of training epochs')        # use less epochs
parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
parser.add_argument('--layers', type=int, default=8, help='total number of layers')         # only 8 layers
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--auxiliary', action='store_true', default=False, help='use auxiliary tower')
parser.add_argument('--auxiliary_weight', type=float, default=0.4, help='weight for auxiliary loss')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
parser.add_argument('--arch', type=str, default='1', help='the previously stored genotype')
parser.add_argument('--split', type=int, default=5000, help='the number of training data')  # default 5000 images
args = parser.parse_args()

# how to utilize GPU here??

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

CIFAR_CLASSES = 10


def main():
  if not torch.cuda.is_available():
    logging.info('no gpu device available')
    sys.exit(1)




  np.random.seed(args.seed)
  torch.cuda.set_device(args.gpu)
  cudnn.benchmark = True
  torch.manual_seed(args.seed)
  cudnn.enabled=True
  torch.cuda.manual_seed(args.seed)
  criterion = nn.CrossEntropyLoss()
  criterion = criterion.cuda()


  split=args.split


  # how to load a model here??
  arch_path="archs/"+args.arch
  f=open(arch_path,'r')
  data=f.read()
  genotype=eval("genotypes.%s" %data)
  f.close()

  # path to store model acc
  exp_path='EXP/KMNIST'+str(split)
  if not os.path.exists(exp_path):
    os.makedirs(exp_path)
  exp_path=os.path.join(exp_path,args.arch) 

  f=open(exp_path,"w")

  # path to store the model
  model_path="models/KMNIST"+str(split)
  if not os.path.exists(model_path):
    os.makedirs(model_path)
  model_path=os.path.join(model_path,args.arch+'.pt')


  # change the model here
  model = NetworkMNIST(args.init_channels, CIFAR_CLASSES, args.layers, args.auxiliary, genotype)
  model = model.cuda()
  logging.info("param size = %fMB", utils.count_parameters_in_MB(model))



  optimizer = torch.optim.SGD(
      model.parameters(),
      args.learning_rate,
      momentum=args.momentum,
      weight_decay=args.weight_decay)


  #******************should change the data_transforms_cifar10**********************# 
  train_transform, valid_transform = utils._data_transforms_KMNIST(args)           #
  #*********************************************************************************#

  # download dataset using torchvision.datasets 
  #**************************   *****************************************************#
  train_data = dset.KMNIST(root=args.data, train=True, download=True, transform=train_transform)
  valid_data = dset.KMNIST(root=args.data, train=False, download=True, transform=valid_transform)
  #********************************  ************************************************#


  num_train = len(train_data)
  indices = list(range(num_train))

  # 在training data上做训练和验证
  train_queue = torch.utils.data.DataLoader(
      train_data, batch_size=args.batch_size,
      sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
      pin_memory=True, num_workers=2)

  valid_queue = torch.utils.data.DataLoader(                   
      valid_data, batch_size=args.batch_size,shuffle=False,pin_memory=True, num_workers=2)


  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, float(args.epochs), eta_min=args.learning_rate_min)

 
  for epoch in range(args.epochs):
    logging.info('epoch %d lr %e', epoch+1, scheduler.get_lr()[0])
    model.drop_path_prob = args.drop_path_prob * epoch / args.epochs

    train_acc, train_obj = train(train_queue, model, criterion, optimizer)
    logging.info('epoch %d train_acc %f', epoch+1,train_acc)

    scheduler.step()

    # save these materials
    #utils.save(model, os.path.join(args.save, 'weights.pt'))
    # we don't save the model parameters
    if (epoch+1)%5==0:
        valid_acc, valid_obj = infer(valid_queue, model, criterion)
        logging.info('epoch %d valid_acc %f', epoch+1, valid_acc)
        f.write('epoch: %d valid_obj: %f valid_acc: %f \n'%(epoch+1,valid_obj, valid_acc))

  f.close()

  #store model parameters
  utils.save(model, model_path)



def train(train_queue, model, criterion, optimizer):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()
  model.train()

  for step, (input, target) in enumerate(train_queue):
    input = Variable(input).cuda()
    target = Variable(target).cuda()

    optimizer.zero_grad()
    logits, logits_aux = model(input)
    loss = criterion(logits, target)
    
    if args.auxiliary:
      loss_aux = criterion(logits_aux, target)
      loss += args.auxiliary_weight*loss_aux
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
    optimizer.step()

    prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
    n = input.size(0)
    objs.update(loss.data.item(), n)
    top1.update(prec1.data.item(), n)
    top5.update(prec5.data.item(), n)

    if step % args.report_freq == 0:
      logging.info('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

  return top1.avg, objs.avg


def infer(valid_queue, model, criterion):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()
  model.eval()

  for step, (input, target) in enumerate(valid_queue):
    with torch.no_grad():
        input = Variable(input).cuda()
        target = Variable(target).cuda()

    logits, _ = model(input)
    loss = criterion(logits, target)

    prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
    n = input.size(0)
    objs.update(loss.data.item(), n)
    top1.update(prec1.data.item(), n)
    top5.update(prec5.data.item(), n)

    if step % args.report_freq == 0:
      logging.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

  return top1.avg, objs.avg


if __name__ == '__main__':
  main() 