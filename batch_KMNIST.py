#!/usr/bin/env python
import os
import sys
import multiprocessing


trainNum =[5000,10000,20000,30000]  
gpu_list=[0,1,2,3,4,5,6,7]
    

def run(index,gpu):
    start=index*13+1
    end=(index+1)*13+1

    for num in trainNum:
        for arch in range(start,end):
            cmdLine = "python train_KMNIST.py --arch=%d --gpu=%d --split=%d"%(arch,gpu,num)
            print(cmdLine)
            os.system(cmdLine)

def batchRun(nprocess=8):
    pool = multiprocessing.Pool(processes = nprocess)

    for index,gpu in enumerate(gpu_list):  # enumerate all methods
        pool.apply_async(run,(index,gpu))
                      
    pool.close()
    pool.join()
    
    
if __name__ == "__main__":
    nprocess = 8
    batchRun(nprocess)
    
