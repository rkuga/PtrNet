from chainer import serializers
from chainer import Variable
import numpy as np
import chainer.functions as F
from chainer import cuda
import os
import xml.dom.minidom
import scipy.io
import cv2
from PIL import Image
import json
import h5py
import cPickle
import gzip

class BaseNetwork(object):

    def __init__(self, epochs,  save_every_m):
        self.save_every_m = save_every_m
        self.epochs=epochs
    
    def my_state(self):
        return '%s_'%(self.net)

    
    def save_params(self, epoch):
        print "==> saving state %s" % self.out_model_dir
        serializers.save_hdf5("%s/net_model_classifier_%d.h5"%(self.out_model_dir, epoch),self.network)
    

    def load_state(self,path,epoch):
        print "==> loading state %s epoch %s"%(path,epoch)
        serializers.load_hdf5('./states/%s/net_model_classifier_%s.h5'%(path,epoch), self.network)
        return int(epoch)


    def read_batch(self, perm, batch_index, data_raw):

        data = np.zeros((self.batchsize, self.in_channel, self.input_height, self.input_width), dtype=np.float32)
        label = np.zeros((self.batchsize), dtype=np.int32)

        for j_,j in enumerate(perm[batch_index:batch_index+self.batchsize]):
            data[j_,:,:,:] = data_raw[j][0].astype(np.float32)
            label[j_] = int(data_raw[j][1])

        return data, label
    
    
    def step(self,perm,batch_index, mode, epoch): 
        if mode =='train':
            data, label=self.read_batch(perm,batch_index,self.train_data)
        else:
            data, label=self.read_batch(perm,batch_index,self.test_data)

        data = Variable(cuda.to_gpu(data))
        yl = self.network(data)

        label=Variable(cuda.to_gpu(label))

        L_network = F.softmax_cross_entropy(yl, label)
        A_network = F.accuracy(yl, label)

        if mode=='train':
            self.o_network.zero_grads()
            L_network.backward()
            self.o_network.update()


        return {"prediction": yl.data.get(),
                "current_loss": L_network.data.get(),
                "current_accuracy": A_network.data.get(),
        }

  
    def get_dataset(self):
        '''
        I assume the train and test data contains only 10 citys problem.
        Variable:train,valid,test is list of tuple respectively.
        Tuple consists of (np.array([x,y]),np.array([2,4,5,1,3,10,7,9,8,6])) for example,
        where x and y are the coodinates of citys and later is order of traveling.
        '''
        path = "./train_10citys.pkl.gz"
        test_data_path = './test_10citys.pkl.gz'

        data_dir, data_file = os.path.split(path)
        if data_dir == "" and not os.path.isfile(path):
            path = os.path.join(
                os.path.split(__file__)[0],
                "..",
                "data",
                path
            )
        if self.mode == 'train':
            if path.endswith(".gz"):
                f = gzip.open(path, 'rb')
            else:
                f = open(path, 'rb')

            train, valid, test = cPickle.load(f)
            f.close()
        else:
            f = gzip.open(test_data_path, 'rb')
            train, valid, test = cPickle.load(f)
            f.close()


        self.out_model_dir = './states/'+self.my_state()
        if not os.path.exists(self.out_model_dir):
            os.makedirs(self.out_model_dir)

        if self.mode=='train':
            print "==> %d training examples" % len(train)
            print "out_model_dir ==> %s " % self.out_model_dir
            print "==> %d valid examples" % len(valid)
        else:
            print "==> %d test examples" % len(test)

        return train, valid, test