# PtrNet

Chainer implementation of ptrnet for TSP.  
The dataset is not available here so please prepare by yourself. Discription is in /nets/base_network.py.  

## how to use  
  
### step1.  
Install the chainer.
```
pip install chainer=='1.24.0'
```
  
### step2.  
Create the dataset as discribed in base_network.py.  
I used 1048576 tours for train data.  
  
### step3.  
Run the code for training.  
```
python main.py
```
  
### step4.  
After training 10 epochs, run the code for testing   
```
python main.py --mode test --load ptrnet 10 --beam_width 10
```  
Beam search will be done if you assign the beam_width > 1.  



## result  
The original paper said the average tour length predicted from ptrnets is 2.88 compared with its ground truth length 2.87, and actually I got 2.88.

