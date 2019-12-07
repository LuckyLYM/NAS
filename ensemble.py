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
import random
import math
import copy

from torch.autograd import Variable
from model import NetworkCIFAR
import genotypes

from itertools import combinations


parser = argparse.ArgumentParser("Ensemble")
parser.add_argument('--data', type=str, default='data', help='location of the data corpus')    # should modify
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=0.001, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=10, help='num of training epochs')        # use less epochs
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
args = parser.parse_args()


############################################################## MODIFY THE LAYERS #########################




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



  #************************ random sample and load archs************************#
  nModel=5
  archs=np.array(random.sample(range(1,101),nModel))

  performance=[]

############################################################## MODIFY THE EXP PATH ############################
  for arch in archs:
    exp_path='EXP/CIFAR50000/'+str(arch)
    f=open(exp_path,'r')
    acc=0
    while True:
        line=f.readline()
        if not line:
            performance.append(acc)
            #print("arch: %d  valid acc: %f"%(arch,acc))
            break
        acc=float(line.split()[-1])
    f.close()
  performance=np.array(performance)


  # descending order
  index=np.argsort(-performance)
  archs=archs[index]
  performance=performance[index]




  #print individual arch performance
  for i in range(nModel):
    print('arch: %d valid acc: %f'%(archs[i],performance[i]))

  # load genotypes
  genos=[]
  for arch in archs:
    arch_path="archs/"+str(arch)
    f=open(arch_path,'r')
    data=f.read()
    genotype=eval("genotypes.%s" %data)
    genos.append(genotype)
    f.close()



  ################################################## MODIFY THE DATA TRANSFORMS ##########################
  train_transform, valid_transform = utils._data_transforms_cifar10(args)           
  ################################################## MODIFY THE DATA LOADER ##############################
  valid_data = dset.CIFAR10(root=args.data, train=False, download=True, transform=valid_transform)
  valid_queue = torch.utils.data.DataLoader(                   
      valid_data, batch_size=args.batch_size,shuffle=False,pin_memory=True, num_workers=2)



  # get the prediction of each arch on the validation set
  logit_list=[]
  for index,arch in enumerate(archs):
 
 ################################################ MODIFY THE MODEL ####################
    model = NetworkCIFAR(args.init_channels, CIFAR_CLASSES, args.layers, args.auxiliary, genos[index])
 ################################################ MODIFY THE MODEL PATH ####################
    model_path="models/CIFAR50000/"+str(arch)+"_4_40.pt"
    utils.load(model,model_path)
    model.drop_path_prob = args.drop_path_prob
    model = model.cuda()
    logits=get_logit(valid_queue,model,criterion)
    logit_list.append(logits)
  



  # simple soft voting
  # enumerate all possible combinations
  # brute force version, the slowest implementation
  pool= list(range(nModel))
  for n in range(2,nModel+1):   #ensemble 的个数
    print('****************Assemble %d models******************'%n)
    combs=combinations(pool,n)

    for comb in combs:
      print("Assemble models: ",comb)
      top1 = utils.AvgrageMeter()
      stem=copy.deepcopy(logit_list[comb[0]])

      for step, (input, target) in enumerate(valid_queue):
          for index in range(1,n):
            stem[step]=logit_list[comb[index]][step]+stem[step]

          prec1, _ = utils.accuracy(stem[step], target, topk=(1, 5))
          batchsize = input.size(0)
          top1.update(prec1.data.item(), batchsize)

      logging.info('Simple Soft Voting: valid acc %f', top1.avg)

  # strategy 1: ensemble by simple soft voting and weihted voting
  '''
  stem=copy.deepcopy(logit_list[0])
  weighted_stem=copy.deepcopy(logit_list[0])

  for n in range(2,nModel+1):   #ensemble 的个数
    print('****************Ensemble %d models******************'%n)

    temp=logit_list[n-1]
    
    top1 = utils.AvgrageMeter()
    weighted_top1 = utils.AvgrageMeter()

    for step, (input, target) in enumerate(valid_queue):
        stem[step]=stem[step]+temp[step]
        prec1, _ = utils.accuracy(stem[step], target, topk=(1, 5))
        batchsize = input.size(0)
        top1.update(prec1.data.item(), batchsize)

        c1,c2=cal_coefficients(performance,n)
        weighted_stem[step]=c1*weighted_stem[step]+c2*temp[step]  # need weights here
        weighted_prec1, _ = utils.accuracy(weighted_stem[step], target, topk=(1, 5))
        weighted_top1.update(weighted_prec1.data.item(), batchsize)


    logging.info('Soft Plurality Voting: valid acc %f', top1.avg)
    logging.info('Soft Weighted Voting: valid acc %f', weighted_top1.avg)

  '''


  # soft weighted voting
  '''
  for n in range(2,nModel+1):   #ensemble 的个数    
    print('****************Ensemble %d models******************'%n)

    w=cal_weights(performance,n)


    stem=copy.deepcopy(logit_list[0])
    for step, _ in enumerate(valid_queue):

      stem[step]=stem[step]*w[0]
      
      for index in range(1,n):
        stem[step]=stem[step]+logit_list[index][step]*w[index]


    top1 = utils.AvgrageMeter()
    for step, (input, target) in enumerate(valid_queue):
        prec1, _ = utils.accuracy(stem[step], target, topk=(1, 5))
        batchsize = input.size(0)
        top1.update(prec1.data.item(), batchsize)

    logging.info('Soft Weighted Voting: valid acc %f', top1.avg)
  '''

  

  # strategy 2: ensemble by votes
  # initialize all the predicted label
  '''
  label_list=[]
  for i in range(nModel):
    logit=logit_list[i]
    for step,values in enumerate(logit):
      _, pred= values.topk(1,1,True,True)
      pred=pred.numpy()

      
      if len(label_list)<step+1:
        label_list.append(pred)
      else:
        label_list[step]=np.column_stack((label_list[step],pred))  # 每一列都是一个model 输出的label


  for num in range(2,nModel+1):   #ensemble 的个数
    print('****************Ensemble %d models******************'%num)

    top1 = utils.AvgrageMeter()

    for step, ( _, target) in enumerate(valid_queue):

        n = target.size(0)

        # how to get pred, the majority vote#
        votes=np.zeros(n)
        labels=label_list[step]
        for i in range(n):
          vote=max_count(labels[i][:num])
          votes[i]=vote

        votes=torch.from_numpy(votes)
        #***********************************#
        correct = votes.eq(target.view(1, -1))
        prec1=correct.view(-1).float().sum(0).mul_(100.0/n)
        top1.update(prec1.data.item(), n)

        #if step % args.report_freq == 0:
        #  logging.info('valid %03d %f %f', step, top1.avg, top5.avg)
    logging.info('Plurality Voting: valid acc %f', top1.avg)

    # calcualte weight of each base classifier
    w=cal_weights(performance,num)
    for step, ( _, target) in enumerate(valid_queue):

        n = target.size(0)

        # how to get pred, the majority vote#
        votes=np.zeros(n)
        labels=label_list[step]

        for i in range(n):
          vote=weighted_max_count(labels[i][:num],w)
          votes[i]=vote

        votes=torch.from_numpy(votes)
        #***********************************#
        correct = votes.eq(target.view(1, -1))
        prec1=correct.view(-1).float().sum(0).mul_(100.0/n)
        top1.update(prec1.data.item(), n)

        #if step % args.report_freq == 0:
        #  logging.info('valid %03d %f %f', step, top1.avg, top5.avg)
    logging.info('Weighted Voting: valid acc %f', top1.avg)
    '''

def cal_coefficients(p,num):
  w=[]
  s=0
  for i in range(num):
    v=p[i]/100
    v=math.log(v/(1-v))
    s=s+v
    w.append(v)
  for i in range(num):
    w[i]=w[i]/s

  c2=w[num-1]
  c1=1-c2
  return c1,c2

def cal_weights(p,num):
  w=[]
  s=0
  for i in range(num):
    v=p[i]/100
    v=math.log(v/(1-v))
    s=s+v
    w.append(v)
  for i in range(num):
    w[i]=w[i]/s
  return w


def weighted_max_count(lt,w):
    d = {}
    for index,label in enumerate(lt):
        if label not in d:
            d[label] = w[index]
        else:
            d[label] = d[label]+w[index]
    return max(d,key=d.get)

def max_count(lt):
    d = {}
    max_key = None
    for i in lt:
        if i not in d:
            count = np.sum(lt==i)
            d[i] = count
            if count > d.get(max_key, 0):
                max_key = i
    return max_key




def get_logit(valid_queue, model, criterion):
  model.eval()

  logit=[]

  for step, (input, target) in enumerate(valid_queue):
    with torch.no_grad():
        input = Variable(input).cuda()
        target = Variable(target).cuda()

    logits, _ = model(input)  
    logits=logits.cpu().data #.data makes this part works well
    logit.append(logits)  
    # if I append all the data here, then out of memory why

  return logit


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

    # logits is similar to the softmax vector
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