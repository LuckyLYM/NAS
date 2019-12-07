import os
import os.path
import numpy as np
import sys
import random

if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

train_list = ['data_batch_1','data_batch_2','data_batch_3','data_batch_4','data_batch_5']
test_list = ['test_batch']
meta = {
    'filename': 'batches.meta',
    'key': 'label_names',
    'md5': '5ff9c542aee3614f3951f8cda6e48888',
}
data=[]
targets=[]

# currely, we only consider the case where the #classes=10
# we consider distribution
class CIFARGenerator():
    base_folder='cifar-10-batches-py'
    train_list = ['data_batch_1','data_batch_2','data_batch_3','data_batch_4','data_batch_5']
    test_list = ['test_batch']
    meta = {
        'filename': 'batches.meta',
        'key': 'label_names',
        'md5': '5ff9c542aee3614f3951f8cda6e48888',
    }

    def __init__(self,root):
        self.root=root
        self.train_data=[]
        self.train_targets=[]
        self.train_indexs=[]
        self.train_dataset=dict()
        self.train_meta=dict()

        self.test_data=[]
        self.test_targets=[]
        self.test_indexs=[]
        self.test_dataset=dict()
        self.test_meta=dict()
        
        self.load(self.train_data,self.train_targets,self.train_indexs,train=True)
        self.load(self.test_data,self.test_targets,self.test_indexs,train=False)

    def load(self,datahandle,targethandle,indexhandle,train=True):

        file_list=[]
        if train==True:
            file_list=train_list
        else:
            file_list=test_list        


        for file_name in file_list:
            file_path=os.path.join(self.root,self.base_folder,file_name)
            with open(file_path,'rb') as f:
                if sys.version_info[0] == 2:
                    entry = pickle.load(f)
                else:
                    entry = pickle.load(f, encoding='latin1')
                self.entry=entry
                datahandle.append(entry['data'])
                if 'labels' in entry:
                    targethandle.extend(entry['labels'])
                else:
                    targethandle.extend(entry['fine_labels'])
        if train==True:
            self.train_data=np.vstack(datahandle)
        else:
            self.test_data=np.vstack(datahandle)

        self.num_class=len(np.unique(targethandle))

        for cla in range(self.num_class):
            index=[i for i,x in enumerate(targethandle) if x==cla]
            indexhandle.append(index)



    def generateDistribution(self,filename):
        self.train_dataset=dict()
        self.train_meta=dict()
        self.filename=filename
        data_index=[]
        label=[]
        instance=[]
        nInstance=0

        r=np.random.rand(self.num_class)

        for cla,portion in enumerate(r):
            num=int(portion*len(self.train_indexs[cla]))
            s=random.sample(self.train_indexs[cla],num)
            data_index.extend(s)
            label.extend([cla]*num)

            instance.append(num)
            nInstance=nInstance+num

        randnum=random.randint(0,100)
        random.seed(randnum)
        random.shuffle(data_index)
        random.seed(randnum)
        random.shuffle(label)

        self.train_dataset['data']=self.train_data[data_index]
        self.train_dataset['labels']=label

        print("-------------------Generate Dataset %s------------------"%self.filename)
        print("#Classes %d"%self.num_class)
        print("#Instance %d"%nInstance)
        for cla in range(self.num_class):
            print('class %d: %d'%(cla,instance[cla]))

        self.meta['num_class']=self.num_class
        self.meta['num_per_class']=instance
        self.dump(self.root)


    def selectClass(self,chosen_class,datasethandle,datahandle,indexhandle,metahandle,train=True):
        if train==True:
            self.filename='train'
        else:
            self.filename='test'

        data_index=[]
        label=[]
        instance=[]
        nInstance=0

        num_class=len(chosen_class)

        for i,cla in enumerate(chosen_class):
            num=len(indexhandle[cla])
            data_index.extend(indexhandle[cla])
            label.extend([i]*num)
            instance.append(num)
            nInstance=nInstance+num

        randnum=random.randint(0,100)
        random.seed(randnum)
        random.shuffle(data_index)
        random.seed(randnum)
        random.shuffle(label)

        datasethandle['data']=datahandle[data_index]
        datasethandle['labels']=label

        if train==True:
            print("-------------------Generate Dataset %s------------------"%self.filename)
            print("#Classes %d"%num_class)
            print("#Instance %d"%nInstance)
            for cla in range(num_class):
                print('class %d: %d'%(cla,instance[cla]))

            metahandle['num_class']=self.num_class
            metahandle['num_per_class']=instance

        #dump training data
        dir_path=os.path.join(self.root,'CIFAR_'+str(num_class)+'Class')
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)

        self.dump(root=dir_path,train=train)


    def generateClass(self,num_class):    # num_class 表示模板class的个数
        chosen_class=random.sample(list(range(self.num_class)),num_class)
        print('original class ',chosen_class)
        self.train_dataset=dict()
        self.test_dataset=dict()
        self.train_meta=dict()
        self.test_meta=dict()

        self.selectClass(chosen_class,self.train_dataset,self.train_data,self.train_indexs,self.train_meta,train=True)
        self.selectClass(chosen_class,self.test_dataset,self.test_data,self.test_indexs,self.test_meta,train=False)



    def dump(self,root,train=True,meta=False):
        file_path=os.path.join(root,self.filename)       
        fw=open(file_path,'wb')
        if train==True:
            pickle.dump(self.train_dataset,fw,-1)
        else:
            pickle.dump(self.test_dataset,fw,-1)

        if meta==True:
            file_path=os.path.join(root,self.filename+'.meta')
            fw=open(file_path,'wb')
            pickle.dump(self.meta,fw,-1)      


gen=CIFARGenerator('data')
for i in range(4,9):
    gen.generateClass(i)







'''
print(generator.entry)
print(generator.data)
print(type(generator.data))
print("len: ",len(generator.data))
print("object 0: ",generator.data[0])
print(type(generator.data[0]))
print(generator.data[0].shape)
'''
# data is a numpy array 10000*3072
# label is a list
#print(generator.data.shape[1])
#print(generator.targets)