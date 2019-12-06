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
        self.data=[]
        self.targets=[]
        self.indexs=[]

        self.dataset=dict()
        self.meta=dict()
        # read all training data
        for file_name in train_list:
            file_path=os.path.join(self.root,self.base_folder,file_name)
            with open(file_path,'rb') as f:
                if sys.version_info[0] == 2:
                    entry = pickle.load(f)
                else:
                    entry = pickle.load(f, encoding='latin1')
                self.entry=entry
                self.data.append(entry['data'])
                if 'labels' in entry:
                    self.targets.extend(entry['labels'])
                else:
                    self.targets.extend(entry['fine_labels'])
        self.data = np.vstack(self.data)

        #split data into correspoding categories
        self.num_class=len(np.unique(self.targets))

        for cla in range(self.num_class):
            index=[i for i,x in enumerate(self.targets) if x==cla]
            self.indexs.append(index)



        # todo sample
        #self.data = self.data.transpose((0, 2, 3, 1))
        # self.data         50000*3072 numpy array
        # self.targets      50000 list

    def generate(self,filename):
        self.filename=filename
        data_index=[]
        label=[]
        instace=[]
        nInstance=0

        r=np.random.rand(self.num_class)

        for cla,portion in enumerate(r):
            num=int(portion*len(self.indexs[cla]))
            s=random.sample(self.indexs[cla],num)
            data_index.extend(s)
            label.extend([cla]*num)

            instace.append(num)
            nInstance=nInstance+num

        randnum=random.randint(0,100)
        random.seed(randnum)
        random.shuffle(data_index)
        random.seed(randnum)
        random.shuffle(label)

        self.dataset['data']=self.data[data_index]
        self.dataset['labels']=label

        print("-------------------Generate Dataset %s------------------"%self.filename)
        print("#Classes %d"%self.num_class)
        print("#Instance %d"%nInstance)
        for cla in range(self.num_class):
            print('class %d: %d'%(cla,instace[cla]))

        self.meta['num_class']=self.num_class
        self.meta['num_per_class']=instace

        self.dump()

        self.dataset=dict()
        self.meta=dict()

    def dump(self):
        file_path=os.path.join(self.root,self.filename)
        fw=open(file_path,'wb')
        pickle.dump(self.dataset,fw,-1)

        file_path=os.path.join(self.root,self.filename+'.meta')
        fw=open(file_path,'wb')
        pickle.dump(self.meta,fw,-1)      


gen=CIFARGenerator('data')
for i in range(1,6):
    gen.generate('CIFAR'+str(i))







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
