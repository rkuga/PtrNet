import numpy as np
import argparse
import importlib

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='citys', help='dataset to train')
parser.add_argument('--batchsize', type=int, default=128, help='literary')
parser.add_argument('--gpu', type=int, default=0, help='run in  specific GPU')
parser.add_argument('--nz', type=int, default=2, help='dimension of city')
parser.add_argument('--nc', type=int, default=10, help='number of citys')
parser.add_argument('--nh', type=int, default=256, help='number of dimension of hidden state')
parser.add_argument('--beam_width', type=int, default=1, help='beam_width')
parser.add_argument('--epochs', type=int, default=200, help='number of epochs to train')
parser.add_argument('--save_every_m', type=int, default=10, help='save the model every n epochs')
parser.add_argument('--net', type=str, default='ptrnet', help='import the network')
parser.add_argument('--load', nargs=2, type=str, default='', help='loading network parameters')
parser.add_argument('--mode', type=str, default='train', help='train/test')
parser.add_argument('--fine', dest="fine", action='store_true', help='set start epoch to zero')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')

args = parser.parse_args()
print args

print "==> using network %s" % args.net
args_dict = dict(args._get_kwargs())

network_module = importlib.import_module("nets." + args.net)
network = network_module.Network(**args_dict)

def do_epoch(mode, epoch):
    if mode=='train':
        length=len(network.train_data)
        perm = np.random.permutation(length)
    elif mode=='valid':
        length=len(network.valid_data)
        perm = np.array(range(length))

    sum_loss = 0.0
    sum_accuracy = 0.0
    bs=args.batchsize
    batches_per_epoch=length//args.batchsize
    for batch_index in xrange(0, length-bs, bs):  
        step_data=network.step(perm,batch_index,mode,epoch)
        prediction = step_data["prediction"] 
        current_loss = step_data["current_loss"]
        current_accuracy = step_data["current_accuracy"]
        
        sum_loss += current_loss
        sum_accuracy += current_accuracy

    if mode == 'train':
        print "epoch %d end loss: %.10f"%(epoch, sum_loss/batches_per_epoch),
        print "train accuracy: %.10f"%(sum_accuracy/batches_per_epoch)
    elif mode=='valid':
        print "valid loss: %.10f"%(sum_loss/batches_per_epoch),
        print "valid accuracy: %.10f"%(sum_accuracy/batches_per_epoch)

start_epoch=0

if args.load != '':
    start_epoch=network.load_state(args.load[0], args.load[1] )

if args.fine:
    start_epoch=0

if args.mode == 'train':
    print "==> training"  
    start=time.time()
    for epoch in xrange(start_epoch+1,args.epochs+1):
        do_epoch('train', epoch)
        if epoch%1==0 :
            do_epoch('valid',epoch)
        if epoch % args.save_every_m == 0  and epoch != start_epoch:
            network.save_params(epoch)

elif args.mode == 'test':
    print "==> testing"   
    network.test()

else:
    raise Exception('unrecognized mode')
