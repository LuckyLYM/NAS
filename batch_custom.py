#!/usr/bin/env python
import os
import sys
import multiprocessing

##--epochs=80 --split=60000
gpu_list=[0,1,2,3,4,5,6,7]
dataset=['CIFAR_5Class']
    

def run(index,gpu):
    '''
    dataset_index=0
    if index<4:
        dataset_index=0
    else:
        dataset_index=1
        index=index-4
    '''

    start=index*13+1
    end=(index+1)*13+1

    for arch in range(start,end):
        cmdLine = "python train_custom.py --dataset=%s --arch=%d --gpu=%d --layer=8 --epochs=20"%(dataset[0],arch,gpu)
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
    
