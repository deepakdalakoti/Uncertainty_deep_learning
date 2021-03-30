import numpy as np
import torch as th
from torch.autograd import Variable
#import cantera as ct
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense, Dropout
import tensorflow_probability as tfp
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.preprocessing import PowerTransformer, QuantileTransformer

tfd = tfp.distributions


class BNN_tensorflow():


    def __init__(self, inF, outF, units):

        self.batch_size = 1
        self.nbatches = 2000.0

        inputs = keras.layers.Input(shape=(inF,))

        act=keras.layers.LeakyReLU(alpha=0.2)
        #act=tf.nn.sigmoid

        hidden = tfp.layers.DenseFlipout(units,kernel_prior_fn=tfp.layers.default_multivariate_normal_fn,
                           kernel_divergence_fn=self.kernel_divergence_fn,activation=act)(inputs)

        hidden = tfp.layers.DenseFlipout(units, kernel_divergence_fn=self.kernel_divergence_fn,activation=act)(hidden)

        hidden = tfp.layers.DenseFlipout(units,kernel_divergence_fn=self.kernel_divergence_fn,activation=act)(hidden)

#        hidden = tfp.layers.DenseFlipout(units,bias_posterior_fn=tfp.layers.util.default_mean_field_normal_fn(),
#                           bias_prior_fn=tfp.layers.default_multivariate_normal_fn,
#                           kernel_divergence_fn=self.kernel_divergence_fn,
#                           bias_divergence_fn=self.bias_divergence_fn,activation=act)(hidden)

#        params = tfp.layers.DenseFlipout(outF,bias_posterior_fn=tfp.layers.util.default_mean_field_normal_fn(),
#                           bias_prior_fn=tfp.layers.default_multivariate_normal_fn,
#                           kernel_divergence_fn=self.kernel_divergence_fn,
#                           bias_divergence_fn=self.bias_divergence_fn,activation=tf.nn.softmax)(hidden)

        params = tfp.layers.DenseFlipout(outF, kernel_divergence_fn=self.kernel_divergence_fn,
                           activation=tf.nn.sigmoid)(hidden)


        #dist = tfp.layers.DistributionLambda(self.normal_sp)(params) 


        self.model = keras.models.Model(inputs=inputs, outputs=params)
        self.model.compile(keras.optimizers.Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=[tf.keras.metrics.AUC()]) 
#        self.model.compile(keras.optimizers.Adam(learning_rate=1e-3), loss='mean_squared_error', metrics=['accuracy']) 

        self.model_params = keras.models.Model(inputs=inputs, outputs=params)

    def NLL(self, y, distr): 
          return tf.math.reduce_mean(-distr.log_prob(y), axis=-1 ) 
        #return -distr.log_prob(y) 
        

    def normal_sp(self,params): 
        return tfd.Independent(tfd.Normal(loc=params[...,0:9], scale=0.001))      
   
        #scaledtanh=lambda x: 2.0*(1.0-tf.math.tanh(2.0*(abs(x)-1.0)/1.0))

    def kernel_divergence_fn(self, q, p, _):
        return  tfp.distributions.kl_divergence(q, p) / ( self.nbatches*1.0)

    def bias_divergence_fn(self, q, p, _):    
        return  tfp.distributions.kl_divergence(q, p) / (  self.nbatches*1.0)

    def do_training(self, xdataTrN, ydataTrN, xdataTeN, ydataTeN, batch_size, ep):
        self.batch_size = batch_size
        #self.nbatches = xdataTrN.shape[0]/batch_size
        #self.nbatches = xdataTrN.shape[0]

        #es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20,restore_best_weights=True, min_delta=1e-5)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.75, patience=10, min_lr=0.000001,verbose=1,mode='min')

        h=self.model.fit(xdataTrN, ydataTrN, batch_size=self.batch_size, epochs=ep, validation_data = (xdataTeN, ydataTeN), callbacks=[reduce_lr])
        return h

    def do_training2(self, xdataTrN, ydataTrN, batch_size, ep):
        self.batch_size = batch_size
        #self.nbatches = xdataTrN.shape[0]/batch_size
        #self.nbatches = xdataTrN.shape[0]

        es = EarlyStopping(monitor='loss', mode='min', verbose=1, patience=10,restore_best_weights=True, min_delta=1e-5)
        reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, min_lr=0.000001,verbose=1,mode='min')

        h=self.model.fit(xdataTrN, ydataTrN, batch_size=self.batch_size, epochs=ep, callbacks=[reduce_lr])
        return h

    def pred(self,xdata, samps):
        pred = np.zeros([xdata.shape[0],samps])
        for i in range(samps):
            pred[:,i:i+1] = self.model(xdata)
        return pred, np.mean(pred,axis=1)[:,None], np.std(pred,axis=1)[:,None]


def get_BNN(batch_size, xdataN, ydataN, units):
    nbatches = int(xdataN.shape[0]/batch_size)
    kernel_divergence_fn=lambda q, p, _: tfp.distributions.kl_divergence(q, p) / ( nbatches*1.0)
    bias_divergence_fn=lambda q, p, _: tfp.distributions.kl_divergence(q, p) / (  nbatches*1.0)


    inputs = keras.layers.Input(shape=(xdataN.shape[1],))

    act=keras.layers.LeakyReLU(alpha=0.1)

    hidden = tfp.layers.DenseFlipout(units,bias_posterior_fn=tfp.layers.util.default_mean_field_normal_fn(),
                           bias_prior_fn=tfp.layers.default_multivariate_normal_fn,
                           kernel_divergence_fn=kernel_divergence_fn,
                           bias_divergence_fn=bias_divergence_fn,activation=act)(inputs)

    hidden = tfp.layers.DenseFlipout(units,bias_posterior_fn=tfp.layers.util.default_mean_field_normal_fn(),
                           bias_prior_fn=tfp.layers.default_multivariate_normal_fn,
                           kernel_divergence_fn=kernel_divergence_fn,
                           bias_divergence_fn=bias_divergence_fn,activation=act)(hidden)

    hidden = tfp.layers.DenseFlipout(units,bias_posterior_fn=tfp.layers.util.default_mean_field_normal_fn(),
                           bias_prior_fn=tfp.layers.default_multivariate_normal_fn,
                           kernel_divergence_fn=kernel_divergence_fn,
                           bias_divergence_fn=bias_divergence_fn,activation=act)(hidden)

    params = tfp.layers.DenseFlipout(ydataN.shape[1],bias_posterior_fn=tfp.layers.util.default_mean_field_normal_fn(),
                           bias_prior_fn=tfp.layers.default_multivariate_normal_fn,
                           kernel_divergence_fn=kernel_divergence_fn,
                           bias_divergence_fn=bias_divergence_fn,activation="linear")(hidden)


    dist = tfp.layers.DistributionLambda(normal_sp)(params) 


    model_vi = keras.models.Model(inputs=inputs, outputs=dist)
    model_vi.compile(keras.optimizers.Nadam(learning_rate=1e-3), loss=NLL, metrics=['mse']) 

    model_params = keras.models.Model(inputs=inputs, outputs=params)
    return model_vi, model_params
    

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

class MC_dropout(th.nn.Module):
    def __init__(self, Nin, H, Nout, drp, act, lr):
        super().__init__()
        self.drp = drp
        self.model = th.nn.Sequential(
        th.nn.Linear(Nin, H),
        act,    
        th.nn.Dropout(drp),
        th.nn.Linear(H, H),
        act,
        th.nn.Dropout(drp),
        #th.nn.Linear(H, H),
        #th.nn.Dropout(drp),
        #act,
        #th.nn.Linear(H, H),
        #th.nn.Dropout(drp),
        #act,
        #th.nn.Linear(H, H),
        #th.nn.Dropout(drp),
        #act,
        th.nn.Linear(H, Nout),
        #th.nn.Sigmoid(),
        )

        self.optimizer = th.optim.Adam(self.model.parameters(), lr=lr)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.75, patience=5,
                verbose=True, threshold=0.01, threshold_mode='abs', cooldown=0, min_lr=0, eps=1e-07)

    def train(self,xtrain, ytrain, xval, yval,  batch_size, epochs):
        xtrain, ytrain = th.from_numpy(xtrain).float(), th.from_numpy(ytrain).float()
        xval, yval = th.from_numpy(xval).float(), th.from_numpy(yval).float()

        data = th.utils.data.TensorDataset(xtrain, ytrain)
        dataloader = th.utils.data.DataLoader(dataset=data, batch_size=batch_size, shuffle=True)
        #lossfn = th.nn.BCELoss(reduction='sum')
        lossfn = th.nn.MSELoss(reduction='sum')

        hist = []
        hist_val = []
        for ep in range(epochs):
            total_loss = 0
            #self.model.train()
            for x, y in dataloader:
                out = self.model(x)
                loss = lossfn(out,y)
                total_loss = total_loss+loss
            
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            #self.model.eval()
            val_loss = lossfn(self.model(xval),yval)
            #self.scheduler.step(val_loss)
            hist.append(total_loss/xtrain.shape[0])
            hist_val.append(val_loss/xval.shape[0])
            print("Epoch {}: Train loss: {} Val loss: {}".format(ep, total_loss/xtrain.shape[0], val_loss/xval.shape[0]))
        return hist, hist_val


    def pred(self, x, nsamp):
        preds = np.zeros([x.shape[0],nsamp])
        x = th.from_numpy(x).float()
        for i in range(nsamp):
            preds[:,i] = self.model(x).detach().numpy()[:,0]
        means = np.mean(preds,axis=1)
        std = np.std(preds,axis=1)
        return preds, means[:,None], std[:,None]

def NLL_ens(ytrue, ypred):
        var = tf.math.softplus(ypred[...,1:2]) + 1e-6
        NLL = tf.math.log(var)*0.5 + 0.5*tf.math.divide(tf.math.square(ytrue-ypred[...,0:1]),var)
        return tf.reduce_mean(NLL)
        #return -ypred.log_prob(ytrue)


class deep_ensemble():

    def __init__(self, inF, outF, H, lr=1e-4,  problem='regression'):
        self.inF= inF
        self.outF = outF
        self.H = H
        self.problem = problem
        self.lr = lr
        return

    def base_model(self):
        if(self.problem=='regression'):
            act = None
            loss = NLL_ens 
            metrics = []
            outF = self.outF*2
        else:
            act = 'softmax'
            loss = 'categorical_crossentropy'
            metrics = [tf.keras.metrics.AUC()]
            outF = self.outF

        inputs = Input(shape=(self.inF,))
        x = Dense(self.H, activation='relu')(inputs)
        x = Dense(self.H, activation='relu')(x)
        x = Dense(outF, activation=act)(x)
        #dist = tfp.layers.DistributionLambda(lambda t: tfd.Normal(loc=t[..., :1],
        #                   scale=1e-6 + tf.math.softplus( t[...,1:])))(x)

        model = keras.Model(inputs, x)
        #model_params = keras.Model(inputs, x)
        #model = tf.keras.Sequential([
        #    tf.keras.layers.Dense(self.H, input_shape=(1,), activation='relu'),
        #    tf.keras.layers.Dense(2),
        #    tfp.layers.DistributionLambda(lambda t: tfd.Normal(loc=t[..., :1],
        #                   scale=1e-6 + tf.math.softplus( t[...,1:])))


        #])

        model.compile(keras.optimizers.Adam(learning_rate=self.lr), loss=loss, metrics=metrics) 
        return model

    def _generate_ensembles(self, N):
        self.models = []
        for i in range(N):
            self.models.append(self.base_model())

        return

    def train(self, xtrain, ytrain, batch, epochs, validation_data=None):
        h = []
        callbacks = keras.callbacks.TensorBoard(log_dir='./logs')
        for i in range(len(self.models)):
            h1=self.models[i].fit(xtrain, ytrain, batch_size=batch, validation_data=validation_data, epochs=epochs, callbacks=[callbacks])
            h.append(h1)
        return h

    def pred(self, xdata):
        pred_all = np.zeros([len(self.models),xdata.shape[0], 2])
        for i in range(len(self.models)):
            pred_all[i,:,:] = self.models[i].predict(xdata)

        return pred_all, np.mean(pred_all,axis=0), np.std(pred_all,axis=0)

    def pred_regression(self, xdata, y, norm):
        pred_all = np.zeros([len(self.models),xdata.shape[0], 2])
        for i in range(len(self.models)):
            pred_all[i,:,:] = self.models[i].predict(xdata)

        #pred_all = do_inverse_norm(y, pred_all, norm)
        # Mean, std as in paper
        pred_mean = np.mean(pred_all[:,:,0],axis=0)
        pred_std = np.sqrt(np.mean(pred_all[:,:,0]**2 + tf.math.softplus(pred_all[:,:,1]).numpy()+1e-6,axis=0) - pred_mean**2 )
        #pred_mean = do_inverse_norm(y, pred_mean, norm)
        #pred_std = do_inverse_norm(y, pred_std, norm)
        #pred_std = pred_std*np.std(y)
        return pred_all, pred_mean, pred_std


class deep_ensemble2():

    def __init__(self, inF, outF, H, lr=1e-4,  problem='regression'):
        self.inF= inF
        self.outF = outF
        self.H = H
        self.problem = problem
        self.lr = lr
        return

    def base_model(self):
        if(self.problem=='regression'):
            act = None
            metrics = []
            outF = self.outF*2
        else:
            act = 'softmax'
            loss = 'categorical_crossentropy'
            metrics = [tf.keras.metrics.AUC()]
            outF = self.outF

        init = tf.keras.initializers.RandomNormal(stddev=0.5)
        init = tf.keras.initializers.glorot_normal()
        
        inputs = Input(shape=(self.inF,))
        x = Dense(self.H, activation=tf.nn.relu, kernel_initializer=init)(inputs)
        x = Dense(self.H, activation=tf.nn.relu, kernel_initializer=init)(x)
        x = Dense(self.H, activation=tf.nn.relu, kernel_initializer=init)(x)
        #x1 = Dense(10, activation='linear', kernel_initializer=init)(x)
        #x1 = Dense(10, activation='linear', kernel_initializer=init)(x1)
        mu = Dense(1, activation='linear', kernel_initializer=init)(x)

        #x2 = Dense(10, activation='linear', kernel_initializer=init)(x)
        #x2 = Dense(10, activation='linear', kernel_initializer=init)(x2)
        sigma = Dense(1, activation='softplus', kernel_initializer=init)(x)
        #sigma = sigma + 1e-6
        model = keras.Model(inputs=inputs, outputs=[mu, sigma])
        return model

    def _generate_ensembles(self, N):
        self.models = []
        for i in range(N):
            self.models.append(self.base_model())

        return

    def NLL(self, mu, sigma, ytrain):

        #var = tf.math.log(1.0+tf.math.exp(ypred[...,1:2])) + 1e-6
        var = sigma + 1e-6
        NLL = tf.math.log(var)*0.5 + 0.5*tf.math.divide(tf.math.square(ytrain-mu),var)  
        #NLL = tf.math.square(ytrain-ypred)
        return tf.reduce_mean(NLL)
    
    def train_step(self, model, xtrain, ytrain):
        with tf.GradientTape() as tape:
            mu, sigma = model(xtrain, training=True)
            loss = self.NLL(mu, sigma, ytrain)
        grad = tape.gradient(loss, model.trainable_variables)
        self.optimizer.apply_gradients(zip(grad, model.trainable_variables))
        return self.optimizer.iterations.numpy(), loss.numpy()

    def train(self, model, epochs, xtrain, ytrain):
        shuf = np.arange(0,xtrain.shape[0])
        np.random.shuffle(shuf)
        xtrain = xtrain[shuf,:]
        ytrain = ytrain[shuf,:]
        self.optimizer = tf.keras.optimizers.Adam(self.lr, beta_1=0.9, beta_2=0.999)
        for i in range(epochs):
            step, loss = self.train_step(model, xtrain, ytrain)
            print("Step {} loss {}".format(step, loss))
        return

    def train_ensemble(self, N, xtrain, ytrain, epochs):
        self.models = []
        for i in range(N):
            self.models.append(self.base_model())
            self.train(self.models[i], epochs, xtrain, ytrain)
        return

    def pred_regression(self, xdata, y, norm):
        pred_all = np.zeros([len(self.models),xdata.shape[0], 2])
        for i in range(len(self.models)):

            pred_all[i,:,:] = self.models[i].predict(xdata)

        #pred_all = do_inverse_norm(y, pred_all, norm)
        # Mean, std as in paper
        pred_mean = np.mean(pred_all[:,:,0],axis=0)
        pred_std = np.sqrt(np.mean(pred_all[:,:,0]**2 + tf.math.softplus(pred_all[:,:,1]).numpy()+1e-6,axis=0) - pred_mean**2 )
        #pred_mean = do_inverse_norm(y, pred_mean, norm)
        #pred_std = do_inverse_norm(y, pred_std, norm)
        #pred_std = pred_std*np.std(y)
        return pred_all, pred_mean, pred_std


