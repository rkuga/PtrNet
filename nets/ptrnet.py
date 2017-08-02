import numpy as np
import math
import chainer
from chainer import cuda
from chainer import optimizers
from chainer import serializers
from chainer import Variable
import chainer.functions as F
import chainer.links as L
import matplotlib
matplotlib.use('Agg')
import matplotlib.pylab as pylab
from base_network import BaseNetwork
from progressbar import ProgressBar
import matplotlib.pyplot as plt
from utils.beam_search import beam_search

class Encoder(chainer.Chain):
    def __init__(self, in_size, nh):
        initializer = chainer.initializers.HeNormal()
        super(Encoder, self).__init__(
            #encoder
            l1_x=L.Linear(in_size, 4*nh, initialW=initializer),
            l1_h=L.Linear(nh, 4*nh, nobias=True, initialW=initializer),
        )
    def __call__(self, cur_city, state):
        h1_in = self.l1_x(cur_city) + self.l1_h(state['h1'])
        c1, h1 = F.lstm(state['c1'], h1_in)
        state = {'c1': c1, 'h1': h1}
        return state,h1
 
class Decoder(chainer.Chain):
    def __init__(self, in_size, nh):
        initializer = chainer.initializers.HeNormal()
        super(Decoder, self).__init__(     
            # decoder
            l1_x=L.Linear(in_size, 4*nh, initialW=initializer),
            l1_h=L.Linear(nh, 4*nh, nobias=True, initialW=initializer),

        )

    def __call__(self,feature, state):
        h1_in = self.l1_x(feature) + self.l1_h(state['h1'])
        c1, h1 = F.lstm(state['c1'], h1_in)
        state = {'c1': c1, 'h1': h1}
        return state,h1

class Pointer(chainer.Chain):
    def __init__(self, nh):
        initializer = chainer.initializers.HeNormal()
        super(Pointer, self).__init__(     
            # pointer net
            vt=L.Linear(nh, 1, nobias=True, initialW=initializer),
            W1 = L.Linear(nh, nh, nobias=True, initialW=initializer),
            W2 = L.Linear(nh, nh, nobias=True, initialW=initializer),
        )

    def __call__(self,enc_outs,y):
        ptrs = [F.tanh(self.W1((ptr))+self.W2((y))) for ptr in enc_outs]
        out = F.concat(tuple(self.vt(ptrs[i]) for i in range(len(ptrs))))

        return out



class Network(BaseNetwork):
    def __init__(self,gpu,batchsize,net,mode,epochs,save_every_m,load,nz,nc,lr,nh,beam_width, **kwargs):
        super(Network, self).__init__(epochs,save_every_m)
        print "==> not used params in this network:", kwargs.keys()
        print "building ..."
        self.nz=nz
        self.nc=nc
        self.net = net
        self.mode=mode
        self.lr=lr
        self.batchsize=batchsize
        self.nh = nh
        self.beam_width=beam_width
        self.train_data, self.valid_data, self.test_data=self.get_dataset(dataset)

        self.enc = Encoder(self.nz,self.nh)
        self.dec = Decoder(self.nz,self.nh)
        self.ptr = Pointer(self.nh)

        self.xp = cuda.cupy
        cuda.get_device(gpu).use()

        self.enc.to_gpu()
        self.dec.to_gpu()
        self.ptr.to_gpu()

        self.o_enc = self.make_optimizer(self.enc,lr)
        self.o_dec = self.make_optimizer(self.dec,lr)
        self.o_ptr = self.make_optimizer(self.ptr,lr)

    def make_optimizer(self,model,lr):
        optimizer = optimizers.RMSpropGraves(lr=lr)
        optimizer.setup(model)
        # optimizer.add_hook(chainer.optimizer.WeightDecay(0.0001))
        return optimizer

    def my_state(self):
        return '%s'%(self.net)

    def read_batch(self, perm, batch_index,data_raw):
        data = np.zeros((self.batchsize, self.nz, self.nc+1), dtype=np.float32)
        t = np.zeros((self.batchsize,self.nc+1), dtype=np.int32)
        for j_,j in enumerate(perm[batch_index:batch_index+self.batchsize]):
            # insert the first dummy city 
            data[j_,:,1:] = data_raw[j][0].T
            t[j_,1:] = data_raw[j][1]
        
        return data, t

    def step(self,perm,batch_index,mode,epoch): 
        if mode=='train':
            data,t=self.read_batch(perm,batch_index,self.train_data)
        else :
            data,t=self.read_batch(perm,batch_index,self.valid_data)

        loss=Variable(cuda.cupy.asarray(0.0).astype(np.float32))
        acc=0.0

        state = {name: Variable(self.xp.zeros((self.batchsize, self.nh),dtype=self.xp.float32)) for name in ('c1', 'h1')}
        
        enc_outs=[]
        for cur_city in data.transpose(1,0,2).T:
            state,h = self.enc(Variable(cuda.to_gpu(cur_city)), state)
            enc_outs.append(h)

        for cur_city,next_city in zip(t.T[0:-1],t.T[1:]):
            cur_city = np.array([data[i,:,cur_city[i]] for i in range(self.batchsize)]).astype(np.float32)
            state,y = self.dec(Variable(cuda.to_gpu(cur_city)),state)
            ptr = self.ptr(enc_outs, y) 
            loss += F.softmax_cross_entropy(ptr, Variable(cuda.to_gpu(next_city))) 
            acc += F.accuracy(ptr, Variable(cuda.to_gpu(next_city))).data.get()


        if mode=='train':
            self.enc.cleargrads()
            self.dec.cleargrads()
            self.ptr.cleargrads()
            loss.backward()
            self.o_enc.update()
            self.o_dec.update()
            self.o_ptr.update()


        return {"prediction": 0,
                "current_loss": loss.data.get()/(t.T.shape[0]-1),
                "current_accuracy": acc/(t.T.shape[0]-1),
        }


    def test(self):
        p = ProgressBar()
        sum_accuracy = 0.0
        batchsize=self.batchsize
        xs=[]
        ys=[]
        routes=[]
        truths=[]
        for i_  in p(xrange(0,len(self.test_data),batchsize)): 
            data = np.zeros((batchsize, self.nz, self.nc+1), dtype=np.float32)
            t = np.zeros((batchsize,self.nc+1), dtype=np.int32)
            for j in xrange(batchsize):
                data[j,:,1:] = self.test_data[i_+j][0].T
                t[j,1:] = self.test_data[i_+j][1]
        
            state = {name: Variable(self.xp.zeros((batchsize, self.nh),dtype=self.xp.float32)) for name in ('c1', 'h1')}

            enc_outs=[]
            for cur_city in data.transpose(1,0,2).T:
                state,h = self.enc(Variable(cuda.to_gpu(cur_city)), state)
                enc_outs.append(h)
            
            if self.beam_width > 1:
                route = beam_search(self.dec,self.ptr,state,enc_outs,t,data,self.beam_width)
            else:
                route=[[] for i in range(batchsize)]
                ptr=np.zeros((batchsize,self.nc+1))
                ptr[:,0]+=1

                for cur_city,next_city in zip(t.T[0:-1],t.T[1:]):
                    y = np.argmax(ptr, axis=1)
                    cur_city = np.array([data[i,:,int(y[i])] for i in range(batchsize)]).astype(np.float32)
                    state,y = self.dec(Variable(cuda.to_gpu(cur_city)),state)
                    ptr = self.ptr(enc_outs, y) 
                    sum_accuracy += F.accuracy(ptr, Variable(cuda.to_gpu(next_city))).data.get()
                    ptr = ptr.data.get()
                    pred_next_city = np.argmax(ptr,axis=1)
                    # if next city has already selected, go to next next city
                    for j in range(batchsize):
                        if pred_next_city[j] in route[j]:
                            ptr[j,np.array(route[j])]=-1000000
                            pred_next_city = np.argmax(ptr,axis=1)
                        route[j].append(pred_next_city[j])

            for j in xrange(batchsize):
                xs.append(np.array(data[j])[0,:])
                ys.append(np.array(data[j])[1,:])
                truths.append(np.array(t[j][1:]))
                routes.append(route[j])
        print 'test accuracy = ', sum_accuracy/float(self.nc*len(self.test_data)/batchsize)
        
        sum_pred_distance=0
        sum_truth_distance=0
        pred_distance_list=[]
        truth_distance_list=[]
        for j in range(len(self.test_data)):
            pred_x_list = [xs[j][i] for i in routes[j]]
            pred_y_list = [ys[j][i] for i in routes[j]]
            truth_x_list = [xs[j][i] for i in truths[j]]
            truth_y_list = [ys[j][i] for i in truths[j]]
            # return to the first city
            pred_x_list.append(xs[j][routes[j][0]])
            pred_y_list.append(ys[j][routes[j][0]])
            truth_x_list.append(xs[j][truths[j][0]])
            truth_y_list.append(ys[j][truths[j][0]])

            pred_distance = sum([math.sqrt(pow(pred_x_list[i+1]-pred_x_list[i],2) + pow(pred_y_list[i+1]-pred_y_list[i],2)) for i in range(len(pred_x_list)-1)])
            truth_distance = sum([math.sqrt(pow(truth_x_list[i+1]-truth_x_list[i],2) + pow(truth_y_list[i+1]-truth_y_list[i],2)) for i in range(len(truth_x_list)-1)])
            sum_pred_distance+=pred_distance
            sum_truth_distance+=truth_distance

            # for the percentile
            pred_distance_list.append(pred_distance/truth_distance)
            truth_distance_list.append(truth_distance/truth_distance)
        
        pred_distance_list.sort()
        pred_distance_array = np.array(pred_distance_list)
        pred_distance_list=[]
        truth_distance_array = np.array(truth_distance_list)
        truth_distance_list=[]
        for i in range(0,101):
            pred_distance_list.append(np.percentile(pred_distance_array,i))
            truth_distance_list.append(np.percentile(truth_distance_array,i))
        plt.plot(range(len(pred_distance_list)),truth_distance_list ,color='blue')
        plt.plot(range(len(pred_distance_list)),pred_distance_list ,color='red')
        plt.ylim(0.9,1.5)
        plt.xlim(0,100)
        plt.savefig('./percentile.png')
        print 'Predicted distance = ',sum_pred_distance/len(self.test_data)
        print 'Ground truth distance = ',sum_truth_distance/len(self.test_data)
        


    def save_params(self, epoch):
        print "==> saving state %s" % self.out_model_dir
        serializers.save_hdf5("%s/net_model_enc_%d.h5"%(self.out_model_dir, epoch),self.enc)
        serializers.save_hdf5("%s/net_model_dec_%d.h5"%(self.out_model_dir, epoch),self.dec)
        serializers.save_hdf5("%s/net_model_ptr_%d.h5"%(self.out_model_dir, epoch),self.ptr)


    def load_state(self,path,epoch):
        print "==> loading state %s epoch %s"%(path,epoch)
        serializers.load_hdf5('./states/%s/net_model_enc_%s.h5'%(path,epoch), self.enc)
        serializers.load_hdf5('./states/%s/net_model_dec_%s.h5'%(path,epoch), self.dec)
        serializers.load_hdf5('./states/%s/net_model_ptr_%s.h5'%(path,epoch), self.ptr)
        return int(epoch)

