import numpy as np

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

