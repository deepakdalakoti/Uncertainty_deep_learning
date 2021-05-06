import numpy as np
from tensorflow.python.keras import backend as K


class ReduceLROnPlateau():
    def __init__(self, optim, factor=0.1, patience=10, min_lr=1e-6):

        self.best = 1e20
        self.wait=0
        self.patience=patience
        self.min_lr = min_lr
        self.factor = factor
        self.optim = optim
        
    def on_epoch_end(self, loss, epoch):
        if(self.best > loss):
            self.wait=0
            self.best = loss
        else:
            self.wait = self.wait+1
            
            cur_lr = K.get_value(self.optim.lr)
            if(cur_lr > self.min_lr):
                if(self.wait > self.patience):
                    self.wait=0
                    new_lr = cur_lr*self.factor
                    new_lr = max(new_lr, self.min_lr)
                    K.set_value(self.optim.lr, new_lr)
                    
                    print("Epoch {}: ReduceLROnPlateau reducing learning rate to {}".format(epoch, new_lr))
        return

def do_normalization(data,data2,which):
    if(which=='range'):
        datanorm = (data)/(np.max(data2,0)-np.min(data2,0))
        return datanorm
    elif(which=='std'):
        datanorm = (data-np.mean(data2,0))/(np.std(data2,0))
        return datanorm
    elif(which=='level'):
        datanorm = (data-np.mean(data2,0))/(np.mean(data2,0))
        return datanorm
    elif(which=='vast'):
        datanorm = (data-np.mean(data2,0))/(np.std(data2,0))*np.mean(data2,0)
        return datanorm
    elif(which=='pareto'):
        datanorm = (data-np.mean(data2,0))/np.sqrt(np.std(data2,0))
        return datanorm
    elif(which=='minmax'):
        datanorm = (data-np.min(data2,0))/(np.max(data2,0)-np.min(data2,0))
        return datanorm
    elif(which=='none'):

        return np.copy(data)
    
def do_inverse_norm(data,datanorm,which):
    if(which=='range'):
        data_inv = datanorm*(np.max(data,0)-np.min(data,0))
        return data_inv
    if(which=='std'):
        data_inv = datanorm*(np.std(data,0))+np.mean(data,0)
        return data_inv
    if(which=='level'):
        data_inv = datanorm*(np.mean(data,0))+np.mean(data,0)
        return data_inv
    if(which=='vast'):
        data_inv = datanorm*(np.std(data,0))/np.mean(data,0)+np.mean(data,0)
        return data_inv
    if(which=='pareto'):
        data_inv = datanorm*np.sqrt(np.std(data,0))+np.mean(data,0)
        return data_inv
    if(which=='minmax'):
        data_inv = datanorm*(np.max(data,0)-np.min(data,0))+np.min(data,0)
        return data_inv
    if(which=='none'):
        return datanorm

