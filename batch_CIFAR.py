#!/usr/bin/env python
import os
import sys
import multiprocessing

##--epochs=80 --split=60000
gpu_list=[3,4,5,6,7]
    

def run(index,gpu):
    start=index*20+1
    end=(index+1)*20+1

    for arch in range(start,end):
        cmdLine = "python train_CIFAR.py --arch=%d --gpu=%d --layer=4 --split=50000 --epochs=40"%(arch,gpu)
        print(cmdLine)
        os.system(cmdLine)

def batchRun(nprocess=8):
    pool = multiprocessing.Pool(processes = nprocess)

    for index,gpu in enumerate(gpu_list):  
        pool.apply_async(run,(index,gpu))
                      
    pool.close()
    pool.join()
    
    
if __name__ == "__main__":
    nprocess = 8
    batchRun(nprocess)
    
