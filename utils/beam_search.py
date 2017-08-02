import numpy as np
from chainer import cuda
from chainer import Variable
import chainer.functions as F


def beam_search(dec,ptrnet,state,enc_outs,t,data,beam_width):  
    beam_width=beam_width
    xp=cuda.cupy
    batchsize=data.shape[0]
    route = np.zeros((batchsize,beam_width,t.T.shape[0])).astype(np.int32)
    
    for (j,next_city) in enumerate(t.T[1:]):
        if j == 0:
            state,y = dec(Variable(cuda.to_gpu(data[:,:,0])), state)
            h=state['h1'].data
            c=state['c1'].data
            h=xp.tile(h.reshape(batchsize,1,-1), (1,beam_width,1))
            c=xp.tile(c.reshape(batchsize,1,-1), (1,beam_width,1))

            ptr = ptrnet(enc_outs, y) 
            ptr=ptr.data.get()

            pred_total_city = np.argsort(ptr)[:,::-1][:,:beam_width]
            pred_total_score = np.sort(ptr)[:,::-1][:,:beam_width]
            route[:,:,j] = pred_total_city
            pred_total_city=pred_total_city.reshape(batchsize,beam_width,1)
        else:
            pred_next_score=np.zeros((batchsize,beam_width,11))
            pred_next_city=np.zeros((batchsize,beam_width,11)).astype(np.int32)
            for b in range(beam_width):
                state={'c1':Variable(c[:,b,:]), 'h1':Variable(h[:,b,:])}
                cur_city = xp.array([data[i,:,int(pred_total_city[i,b,j-1])] for i in range(batchsize)]).astype(xp.float32)
                state,y = dec(Variable(cur_city),state)
                h[:,b,:]=state['h1'].data
                c[:,b,:]=state['c1'].data
                ptr = ptrnet(enc_outs, y) 
                ptr=ptr.data.get()
                pred_next_score[:,b,:]=ptr
                pred_next_city[:,b,:]=np.tile(np.arange(11),(batchsize,1))

            h=F.stack([h for i in range(11)], axis=2).data
            c=F.stack([c for i in range(11)], axis=2).data
            
            pred_total_city = np.tile(route[:,:,:j],(1,1,11)).reshape(batchsize,beam_width,11,j)
            pred_next_city = pred_next_city.reshape(batchsize,beam_width,11,1)
            pred_total_city = np.concatenate((pred_total_city,pred_next_city),axis=3)

            pred_total_score = np.tile(pred_total_score.reshape(batchsize,beam_width,1),(1,1,11)).reshape(batchsize,beam_width,11,1)
            pred_next_score = pred_next_score.reshape(batchsize,beam_width,11,1)
            pred_total_score += pred_next_score

            pred_total_score[:,:,0]=-np.inf
            for b in range(batchsize):
                for (j_,pred_city) in enumerate(pred_total_city[b,:,0,:j]):
                    for i_,i in enumerate(pred_city):
                        pred_total_score[b,j_,i] = -np.inf
            # pred_total_score=[pred_total_score[b,j_,i]*(-np.inf) for b in range(batchsize) for (j_,pred_city) in enumerate(pred_total_city[b,:,0,:j]) for i_,i in enumerate(pred_city)]

            idx = pred_total_score.reshape(batchsize,beam_width * 11).argsort(axis=1)[:,::-1][:,:beam_width]

            pred_total_city = pred_total_city[:,idx//11, np.mod(idx,11), :][np.diag_indices(batchsize,ndim=2)].reshape(batchsize,beam_width,j+1)
            pred_total_score = pred_total_score[:,idx//11, np.mod(idx,11), :][np.diag_indices(batchsize,ndim=2)].reshape(batchsize,beam_width,1)
            h = h[:,idx//11, np.mod(idx,11), :][np.diag_indices(batchsize,ndim=2)].reshape(batchsize,beam_width,256)
            c = c[:,idx//11, np.mod(idx,11), :][np.diag_indices(batchsize,ndim=2)].reshape(batchsize,beam_width,256)

            route[:,:,:j+1] = pred_total_city

    return route[:,0,:j+1].tolist()