import numpy as np
from torch.autograd import Variable
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from util import do_inverse_norm, ReduceLROnPlateau

# Set default type for compatibility between CPU/GPU
tf.keras.backend.set_floatx('float32')
#tf.config.run_functions_eagerly(True)

class deep_ensemble():

    def __init__(self, inF, outF, H, lr=1e-4, Nmodels=5, problem='regression'):
        self.inF= inF
        self.outF = outF
        self.H = H
        self.problem = problem
        self.lr = lr
        self.model_fn = self.base_model_regression
        self.train_step = self.train_step_regression
        self.pred = self.pred_regression
        self.loss_fn = self.loss_reg
        if(problem=='classification'):
            self.model_fn = self.base_model_classification
            self.train_step = self.train_step_classification
            self.pred = self.pred_classification
            self.loss_fn = self.loss_class

        self.models = []
        self.Nmodels = Nmodels
        # Define all optimizers here 
        # so that in repeated training they maintain state
        self.optimizers = []
        for i in range(Nmodels):
            self.models.append(self.model_fn())
            opt=tf.keras.optimizers.Adam(self.lr, beta_1=0.9, beta_2=0.999)
            grad_vars = self.models[i].trainable_weights
            zero_grads = [tf.zeros_like(w) for w in grad_vars]
            opt.apply_gradients(zip(zero_grads, grad_vars)) 
            self.optimizers.append(opt)
 
        return


    def base_model_regression(self):

        init = tf.keras.initializers.glorot_normal()
        
        #inputs = Input(shape=(self.inF,))
        #x = Dense(self.H, activation=tf.nn.relu, kernel_initializer=init)(inputs)
        #x = Dense(self.H, activation=tf.nn.relu, kernel_initializer=init)(x)
        #x = Dense(self.H, activation=tf.nn.relu, kernel_initializer=init)(x)
        #mu = Dense(self.outF, activation='linear', kernel_initializer=init)(x)
        #sigma = Dense(self.outF, activation='softplus', kernel_initializer=init)(x)

        inputs = Input(shape=(self.inF,))

        xmu = Dense(self.H, activation=tf.nn.relu, kernel_initializer=init, name='mu_inp')(inputs)
        xmu = Dense(self.H, activation=tf.nn.relu, kernel_initializer=init, name='mu_d1')(xmu)
        xmu = Dense(self.H, activation=tf.nn.relu, kernel_initializer=init, name='mu_d2')(xmu)
        mu = Dense(self.outF, activation='linear', kernel_initializer=init, name='mu_d3')(xmu)

        xsig = Dense(self.H, activation=tf.nn.relu, kernel_initializer=init, name='sig_inp')(inputs)
        xsig = Dense(self.H, activation=tf.nn.relu, kernel_initializer=init, name='sig_d1')(xsig)
        xsig = Dense(self.H, activation=tf.nn.relu, kernel_initializer=init, name='sig_d2')(xsig)
        sigma = Dense(self.outF, activation='softplus', kernel_initializer=init, name='sig_d3')(xsig)

        model = keras.Model(inputs=inputs, outputs=[mu, sigma])
        model.build(input_shape=(self.inF))
        return model

    def base_model_classification(self):

        init = tf.keras.initializers.glorot_normal()
        
        inputs = Input(shape=(self.inF,))
        x = Dense(self.H, activation=tf.nn.relu, kernel_initializer=init)(inputs)
        x = Dense(self.H, activation=tf.nn.relu, kernel_initializer=init)(x)
        x = Dense(self.H, activation=tf.nn.relu, kernel_initializer=init)(x)
        x = Dense(self.outF, activation='softmax', kernel_initializer=init)(x)
        model = keras.Model(inputs=inputs, outputs=x)
        return model


    def _generate_ensembles(self, N):
        self.models = []
        for i in range(N):
            self.models.append(self.model_fn())

        return
    @tf.function
    def loss_reg(self, model, xtrain, ytrain, training):
        # NLL loss
        mu, sigma = model(xtrain, training=training)
        var = sigma + 1e-6
        NLL = tf.zeros([xtrain.shape[0],], dtype = tf.float32)
        # Add NLL for each component, considering independence
        for i in range(self.outF):
            NLL = NLL +   tf.math.log(var[...,i])*0.5 + \
                    0.5*tf.math.divide(tf.math.square(ytrain[...,i]-mu[...,i]),var[...,i])  
        return tf.reduce_mean(NLL,axis=-1)
    @tf.function
    def loss_class(self, model, xtrain, ytrain, training):
        # 
        ypred = model(xtrain, training=training)
        loss_fn = tf.keras.losses.CategoricalCrossentropy()
        return loss_fn(ytrain, ypred)
    @tf.function
    def train_step_regression(self, model, xtrain, ytrain, indx):
        with tf.GradientTape() as tape:
            loss = self.loss_fn(model,xtrain, ytrain, True)
        grad = tape.gradient(loss, model.trainable_variables)
        self.optimizers[indx].apply_gradients(zip(grad, model.trainable_variables))
        return loss
    @tf.function
    def train_step_classification(self, model, xtrain, ytrain, indx):
        with tf.GradientTape() as tape:
            loss = self.loss_fn(model, xtrain, ytrain, True)
        grad = tape.gradient(loss, model.trainable_variables)
        self.optimizers[indx].apply_gradients(zip(grad, model.trainable_variables))
        return loss

    def train(self, model, batch_size, epochs, xtrain, ytrain, indx, validation_data=None):
        train_dataset = tf.data.Dataset.from_tensor_slices((xtrain.astype(np.float32), ytrain.astype(np.float32)))
        train_dataset = train_dataset.shuffle(buffer_size=xtrain.shape[0], reshuffle_each_iteration=True).batch(batch_size)
        #self.optimizer = tf.keras.optimizers.RMSprop(self.lr)
        red_lr = ReduceLROnPlateau(self.optimizers[indx], 0.8, 10, 1e-5)

        if(validation_data):
            validation_data[0] = validation_data[0].astype(np.float32)
            validation_data[1] = validation_data[1].astype(np.float32)

        train_loss = []
        valid_loss = []
        epoch_loss_avg = tf.keras.metrics.Mean()
        for i in range(epochs):
            epoch_loss_avg.reset_states()
            for x, y in train_dataset:
                loss = self.train_step(model, x, y, indx)
                epoch_loss_avg.update_state(loss)
            train_loss.append(epoch_loss_avg.result().numpy())
            red_lr.on_epoch_end(train_loss[-1], i)
            if(validation_data):
                valid_loss.append(np.mean(self.loss_fn(model, validation_data[0], validation_data[1], False).numpy()))
                print("Step {} loss {} valid_loss {}".format(i, epoch_loss_avg.result(), valid_loss[i]))
            else:
                print("Step {} loss {}".format(i, epoch_loss_avg.result()))
        
        return train_loss, valid_loss

    def train_ensemble(self, xtrain, ytrain, epochs, batch_size, validation_data=None):
        #self.models = []
        hist = []
        hist_valid = []
        for i in range(self.Nmodels):
            #self.models.append(self.model_fn())
            h1, h2 = self.train(self.models[i], batch_size, epochs, xtrain, ytrain, i, validation_data)
            hist.append(h1)
            hist_valid.append(h2)
        return hist, hist_valid

    def pred_regression(self, xdata, y, norm):
        pred_all = np.zeros([len(self.models),xdata.shape[0], 2*self.outF])
        for i in range(len(self.models)):
            mu, sigma = self.models[i](xdata)
            pred_all[i,:,0:self.outF]  = mu
            pred_all[i,:,self.outF:]  = sigma


        # Mean, std as in paper
        
        pred_mean = np.mean(pred_all[:,:,0:self.outF],axis=0)
        pred_std = np.sqrt(np.mean(pred_all[:,:,0:self.outF]**2 + pred_all[:,:,self.outF:]+1e-6,axis=0) - pred_mean**2 )
        pred_mean = do_inverse_norm(y, pred_mean, norm)
        pred_std = pred_std*np.std(y)
         
        return pred_all, pred_mean, pred_std

    def pred_classification(self, xdata, y, norm):
        pred_all = np.zeros([len(self.models),xdata.shape[0], self.outF])
        for i in range(len(self.models)):
            pred_all[i,:,:] = self.models[i](xdata)

        pred_all = do_inverse_norm(y, pred_all, norm)
        # Mean, std as in paper
        pred_mean = np.mean(pred_all,axis=0)
        pred_std = np.std(pred_all,axis=0)
        return pred_all, pred_mean, pred_std

    def predict(self, xdata, y, norm):
        pred_all, pred_mean, pred_std = self.pred(xdata, y, norm)
        return pred_all, pred_mean, pred_std
